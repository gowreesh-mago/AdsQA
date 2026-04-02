"""
Model Evaluation using Qwen2.5-7B-Instruct
Local replacement for GPT-4o based evaluation
"""

import argparse
import json
import os
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from hallucination_verifier import LLMSemanticVerifier


# Enhanced evaluation prompt with robust formatting requirements
prompt_template = """
You are an expert evaluator comparing a model's answer against a ground truth answer.

###Question:
{question}

###Ground Truth Answer:
{golden_answer}

###Response to Evaluate:
{response}

###Evaluation Rules:
1. Compare ONLY the Response against the Ground Truth Answer
2. Check if Response contains ALL key information from Ground Truth
3. Check if Response contains ANY claims NOT present in Ground Truth

###Scoring:
- 1: Response contains ALL key information from Ground Truth
- 0.5: Response contains SOME key information from Ground Truth, but missing other key info
- 0: Response is incorrect (missing most/all key information)

Note: Hallucinations are tracked separately and do NOT affect the score. Score reflects only how much ground truth information is present.

###Output Format (EXACT format required):

Answer: [EXACTLY 0 or 0.5 or 1]

Missing Information: [SPECIFIC_INFO_1] | [SPECIFIC_INFO_2] | ... OR None

Hallucination: [SPECIFIC_CLAIM_1] | [SPECIFIC_CLAIM_2] | ... OR None

IMPORTANT:
- For Answer, write ONLY "Answer: 0", "Answer: 0.5", or "Answer: 1" (no other text)
- Use pipe separators (|) between multiple items
- Write exactly "None" if there are no issues (not "none", "None.", etc.)
- Do not add explanations or additional text after "None"
"""


def read_json(jpath):
    """Load JSON file"""
    with open(jpath, 'r') as ff:
        data = json.load(ff)
    return data


def evaluate_with_qwen(model, tokenizer, question, golden_answer, response):
    """
    Evaluate response using Qwen2.5-7B-Instruct

    Args:
        model: Qwen2.5-7B-Instruct model
        tokenizer: Corresponding tokenizer
        question: Question text
        golden_answer: Ground truth answer
        response: Model's answer to evaluate

    Returns:
        Dictionary with:
        - score: Score string in format "Answer: X" where X is 0, 0.5, or 1
        - missing_info: String describing missing information (pipe-separated)
        - hallucinations: String describing hallucinations (pipe-separated)
        - full_response: Complete model response
    """
    # Format prompt (without meta_info)
    prompt = prompt_template.format(
        question=question,
        golden_answer=golden_answer,
        response=response
    )

    # Create messages
    messages = [{"role": "user", "content": prompt}]

    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode
    generated = outputs[0][inputs["input_ids"].shape[-1]:]
    response_text = tokenizer.decode(generated, skip_special_tokens=True).strip()

    # Helper function to clean claims
    def clean_claims(text):
        """Filter 'None' variants from pipe-separated claims, handle 'OR None' malformations"""
        if not text or text.strip().lower() in ["none", "none.", ""]:
            return ""
        claims = [c.strip() for c in text.split('|') if c.strip()]

        # Filter explicit "None" variants
        claims = [c for c in claims if c.lower() not in ["none", "none.", ""]]

        # Handle malformed "X OR None" or "X or none" patterns
        cleaned_claims = []
        for claim in claims:
            # Remove " OR None" suffix (case-insensitive)
            claim_cleaned = re.sub(r'\s+(or|OR)\s+(none|None|NONE)\.?$', '', claim).strip()
            if claim_cleaned and claim_cleaned.lower() not in ["none", "none.", ""]:
                cleaned_claims.append(claim_cleaned)

        return ' | '.join(cleaned_claims) if cleaned_claims else ""

    # Parse response
    result = {
        "score": "",
        "missing_info": "",
        "hallucinations": "",
        "full_response": response_text
    }

    # Extract score with robust parsing
    score_match = re.search(r'Answer:\s*(0\.5|0|1)(?:\s|$)', response_text)
    if not score_match:
        print(f"WARNING: Could not parse score from: {response_text[:100]}")
        result["score"] = "Answer: 0"  # Conservative default
    else:
        result["score"] = f"Answer: {score_match.group(1)}"

    # Extract missing information (pipe-separated format)
    missing_match = re.search(r'Missing Information:\s*(.*?)(?=Hallucination:|$)', response_text, re.DOTALL)
    if missing_match:
        missing_text = missing_match.group(1).strip()
        result["missing_info"] = clean_claims(missing_text)

    # Extract hallucinations (pipe-separated format)
    halluc_match = re.search(r'Hallucination:\s*(.*?)$', response_text, re.DOTALL)
    if halluc_match:
        halluc_text = halluc_match.group(1).strip()
        result["hallucinations"] = clean_claims(halluc_text)

    return result


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Evaluate predictions using Qwen2.5-7B-Instruct")
    parser.add_argument('--eval_name', type=str, default='qwen3vl-8b.json',
                        help='Prediction file name (e.g., qwen3vl-8b.json)')
    parser.add_argument('--test_file', type=str, default='./testset_groundtruth.json',
                        help='Ground truth file path')
    parser.add_argument('--results_dir', type=str, default='./results/',
                        help='Directory containing prediction files')
    parser.add_argument('--model_path', type=str, default='Qwen/Qwen2.5-7B-Instruct',
                        help='Qwen2.5-7B-Instruct model path')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Evaluate only first N samples for testing')
    args = parser.parse_args()

    print(f"Loading ground truth from: {args.test_file}")
    raw_test_data = read_json(args.test_file)

    # Limit samples for testing if specified
    if args.max_samples:
        raw_test_data = raw_test_data[:args.max_samples]
        print(f"Testing mode: Evaluating only {args.max_samples} samples")

    # Load model
    print(f"Loading model from {args.model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model.eval()
    print("Model loaded successfully!")

    # Initialize LLM-based semantic verifier
    print("Initializing LLM-based semantic verifier...")
    verifier = LLMSemanticVerifier()
    print("Verifier initialized!")

    # Tracking metrics
    strict_acc_scores = {"Type_1": 0, "Type_2": 0, "Type_3": 0, "Type_4": 0, "Type_5": 0}
    strict_acc_counts = {"Type_1": 0, "Type_2": 0, "Type_3": 0, "Type_4": 0, "Type_5": 0}
    relax_acc_scores = {"Type_1": 0, "Type_2": 0, "Type_3": 0, "Type_4": 0, "Type_5": 0}
    relax_acc_counts = {"Type_1": 0, "Type_2": 0, "Type_3": 0, "Type_4": 0, "Type_5": 0}

    pred_nums = 0
    relaxed_acc = 0.
    strict_acc = 0.

    # Track perfect score cases
    perfect_score_cases = []

    # Evaluate each sample
    for ii, item in enumerate(tqdm(raw_test_data, desc="Evaluating")):

        question_id = item['question_id']
        meta_info = item['meta_info']
        gt_answer = item.get('answer', item.get('gt_answer', ''))
        question = item['question']
        question_types = item.get('question_type', [])

        pred_path = os.path.join(args.results_dir, question_id, args.eval_name)

        # Skip if prediction doesn't exist
        if not os.path.exists(pred_path):
            print(f"\nWarning: Prediction not found: {pred_path}")
            continue

        try:
            pred_item = read_json(pred_path)
        except json.decoder.JSONDecodeError:
            print(f"\nError: Invalid JSON: {pred_path}")
            continue

        # Check if already evaluated
        pred_score = pred_item[0].get('score', '')
        if pred_score != "" and pred_score is not None:
            # Already evaluated, parse existing score
            gptscore = pred_score.replace('Answer: ', '').strip()
            pred_answer = pred_item[0]['prediction']
            if pred_answer is None:
                continue

            pred_nums += 1

            # Update metrics
            if '1' in gptscore:
                strict_acc += 1
                relaxed_acc += 1
                # Save perfect score case
                perfect_score_cases.append({
                    'question_id': question_id,
                    'question': question,
                    'question_type': question_types,
                    'ground_truth': gt_answer,
                    'prediction': pred_answer,
                    'score': pred_score,
                    'missing_info': pred_item[0].get('missing_info', ''),
                    'hallucinations': pred_item[0].get('hallucinations', '')
                })
            elif '0.5' in gptscore:
                strict_acc += 0
                relaxed_acc += 0.5
            else:
                strict_acc += 0
                relaxed_acc += 0

            # Update per-type metrics
            for typee in question_types:
                if '1' in gptscore:
                    strict_acc_scores[typee] += 1
                    relax_acc_scores[typee] += 1
                elif '0.5' in gptscore:
                    strict_acc_scores[typee] += 0
                    relax_acc_scores[typee] += 0.5

                strict_acc_counts[typee] += 1
                relax_acc_counts[typee] += 1

            continue

        # Extract answer from prediction
        pred_answer = pred_item[0]['prediction']
        if pred_answer is None:
            continue

        # Extract <answer> content if present
        if "<answer>" in pred_answer:
            pred_answer = re.sub(r'(?s).*<answer>\s*(.*?)\s*</answer>.*', r'\1', pred_answer)

        # NO TRUNCATION - evaluate full predictions for semantic understanding

        # Evaluate with Qwen2.5-7B (without meta_info)
        try:
            eval_result = evaluate_with_qwen(
                model, tokenizer,
                question, gt_answer, pred_answer
            )

            # VERIFICATION STEP 1: Verify hallucinations using LLM
            halluc_verification = verifier.verify_hallucinations_llm(
                eval_result['hallucinations'],
                gt_answer,
                model,
                tokenizer
            )

            # VERIFICATION STEP 2: Verify missing information using LLM
            missing_verification = verifier.verify_missing_info_llm(
                eval_result['missing_info'],
                gt_answer,
                model,
                tokenizer
            )

            # Only keep verified hallucinations
            eval_result['hallucinations'] = ' | '.join(
                halluc_verification['verified_hallucinations']
            ) if halluc_verification['verified_hallucinations'] else ''

            # Only keep verified missing info
            eval_result['missing_info'] = ' | '.join(
                missing_verification['verified_missing']
            ) if missing_verification['verified_missing'] else ''

            # Score remains unchanged - no modifications after initial evaluation

            # Save verification details
            eval_result['hallucination_verification'] = halluc_verification
            eval_result['missing_info_verification'] = missing_verification

        except Exception as e:
            print(f"\nError during evaluation of {question_id}: {e}")
            continue

        # Save score and reasoning back to file
        pred_item[0]['score'] = eval_result['score']
        pred_item[0]['missing_info'] = eval_result['missing_info']
        pred_item[0]['hallucinations'] = eval_result['hallucinations']
        pred_item[0]['evaluation_reasoning'] = eval_result['full_response']
        pred_item[0]['hallucination_verification'] = eval_result.get('hallucination_verification', {})
        pred_item[0]['missing_info_verification'] = eval_result.get('missing_info_verification', {})

        with open(pred_path, 'w', encoding='utf-8') as ff:
            json.dump(pred_item, ff, indent=4, ensure_ascii=False)

        # Parse and update metrics
        try:
            pred_nums += 1
            gptscore_clean = eval_result['score'].replace('Answer: ', '').strip()

            if '1' in gptscore_clean:
                strict_acc += 1
                relaxed_acc += 1
                # Save perfect score case
                perfect_score_cases.append({
                    'question_id': question_id,
                    'question': question,
                    'question_type': question_types,
                    'ground_truth': gt_answer,
                    'prediction': pred_answer,
                    'score': eval_result['score'],
                    'missing_info': eval_result['missing_info'],
                    'hallucinations': eval_result['hallucinations']
                })
            elif '0.5' in gptscore_clean:
                strict_acc += 0
                relaxed_acc += 0.5
            else:
                strict_acc += 0
                relaxed_acc += 0

            # Update per-type metrics
            for typee in question_types:
                if '1' in gptscore_clean:
                    strict_acc_scores[typee] += 1
                    relax_acc_scores[typee] += 1
                elif '0.5' in gptscore_clean:
                    strict_acc_scores[typee] += 0
                    relax_acc_scores[typee] += 0.5

                strict_acc_counts[typee] += 1
                relax_acc_counts[typee] += 1

            print(f"\n{question_id}: {gptscore_clean}")
            if eval_result['missing_info']:
                print(f"  Missing: {eval_result['missing_info'][:100]}...")
            if eval_result['hallucinations']:
                print(f"  Hallucinations: {eval_result['hallucinations'][:100]}...")

        except Exception as e:
            print(f"\nError parsing score for {question_id}: {e}")
            print(f"Evaluation result: {eval_result}")

    # Print final results
    print("\n" + "="*50)
    print("EVALUATION COMPLETE")
    print("="*50)

    print("\nPer-type Strict Accuracy:")
    for typee in strict_acc_scores:
        if strict_acc_counts[typee] > 0:
            print(f"  {typee}: {strict_acc_scores[typee] / strict_acc_counts[typee]:.4f}")

    print("\nPer-type Relaxed Accuracy:")
    for typee in relax_acc_scores:
        if relax_acc_counts[typee] > 0:
            print(f"  {typee}: {relax_acc_scores[typee] / relax_acc_counts[typee]:.4f}")

    print(f"\nTotal evaluated: {pred_nums}")
    print(f"Total target: {len(raw_test_data)}")

    if len(raw_test_data) > 0:
        print(f"\nStrict accuracy: {strict_acc / len(raw_test_data):.4f}")
        print(f"Relaxed accuracy: {relaxed_acc / len(raw_test_data):.4f}")

    # Save perfect score cases
    if perfect_score_cases:
        perfect_scores_path = os.path.join(args.results_dir, f'perfect_scores_{args.eval_name}')
        with open(perfect_scores_path, 'w', encoding='utf-8') as ff:
            json.dump(perfect_score_cases, ff, indent=4, ensure_ascii=False)
        print(f"\nSaved {len(perfect_score_cases)} perfect score cases to: {perfect_scores_path}")
