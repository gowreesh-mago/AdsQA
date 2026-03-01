"""
Model Evaluation using Qwen2.5-7B-Instruct
Local replacement for GPT-4o based evaluation
"""

import argparse
import json
import os
import re
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


# Enhanced evaluation prompt with reasoning
prompt_template = """
You are an advertising expert specializing in evaluating whether a respondent's answer after watching a video matches the golden answer. We will provide the video's Meta-Information, Question, Golden Answer, and the Response to be judged below.

###The meta-information includes the advertisement video's theme, creative points, and a brief content description, which can be regarded as ground-truth information, as follows::
{meta_info}

###Question:
{question}

###Golden Answer:
{golden_answer}

###Rule:
1. If the response to be judged contains ALL key information of the golden answer or expresses the same meaning using other sentences or synonyms, it is considered a match with the golden answer, and the output is 1.
2. If the response to be judged does NOT contain the key information from the golden answer, it is considered a mismatch, and the output is 0.
3. The response to be judged should NOT contain any content that is contradictory, conflicting, or unreasonable when inferred from the meta-information. If such content exist, it is considered a mismatch, and the output is 0.
4. If the response to be judged contains the MOST of key information of the golden answer and, do NOT contain any information that is contradictory, conflicting, or unreasonable when inferred from the meta-information, it is considered a partial match, and the output is 0.5.

###Response to be judged:
{response}

###Instructions:
Provide your evaluation in the following format:

Answer: [0 or 0.5 or 1]

Missing Information:
[List specific key information from the golden answer that is MISSING or INCOMPLETE in the response. If nothing is missing, write "None"]

Hallucinations/Errors:
[List specific claims in the response that CONTRADICT the meta-information or golden answer. If no contradictions, write "None"]
"""


def read_json(jpath):
    """Load JSON file"""
    with open(jpath, 'r') as ff:
        data = json.load(ff)
    return data


def evaluate_with_qwen(model, tokenizer, meta_info, question, golden_answer, response):
    """
    Evaluate response using Qwen2.5-7B-Instruct

    Args:
        model: Qwen2.5-7B-Instruct model
        tokenizer: Corresponding tokenizer
        meta_info: Advertisement meta-information
        question: Question text
        golden_answer: Ground truth answer
        response: Model's answer to evaluate

    Returns:
        Dictionary with:
        - score: Score string in format "Answer: X" where X is 0, 0.5, or 1
        - missing_info: String describing missing information
        - hallucinations: String describing hallucinations/errors
        - full_response: Complete model response
    """
    # Format prompt
    prompt = prompt_template.format(
        meta_info=meta_info,
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

    # Parse response
    result = {
        "score": "",
        "missing_info": "",
        "hallucinations": "",
        "full_response": response_text
    }

    # Extract score
    score_match = re.search(r'Answer:\s*(0\.5|0|1)', response_text)
    if score_match:
        result["score"] = f"Answer: {score_match.group(1)}"

    # Extract missing information
    missing_match = re.search(r'Missing Information:\s*(.*?)(?=Hallucinations/Errors:|$)', response_text, re.DOTALL)
    if missing_match:
        missing_text = missing_match.group(1).strip()
        if missing_text.lower() not in ["none", "none.", ""]:
            result["missing_info"] = missing_text

    # Extract hallucinations
    halluc_match = re.search(r'Hallucinations/Errors:\s*(.*?)$', response_text, re.DOTALL)
    if halluc_match:
        halluc_text = halluc_match.group(1).strip()
        if halluc_text.lower() not in ["none", "none.", ""]:
            result["hallucinations"] = halluc_text

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

    # Tracking metrics
    strict_acc_scores = {"Type_1": 0, "Type_2": 0, "Type_3": 0, "Type_4": 0, "Type_5": 0}
    strict_acc_counts = {"Type_1": 0, "Type_2": 0, "Type_3": 0, "Type_4": 0, "Type_5": 0}
    relax_acc_scores = {"Type_1": 0, "Type_2": 0, "Type_3": 0, "Type_4": 0, "Type_5": 0}
    relax_acc_counts = {"Type_1": 0, "Type_2": 0, "Type_3": 0, "Type_4": 0, "Type_5": 0}

    pred_nums = 0
    relaxed_acc = 0.
    strict_acc = 0.

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

        # Truncate to 30 words
        if len(pred_answer.split()) > 30:
            pred_answer = ' '.join(pred_answer.split()[0:30])

        # Evaluate with Qwen2.5-7B
        try:
            eval_result = evaluate_with_qwen(
                model, tokenizer,
                meta_info, question, gt_answer, pred_answer
            )
        except Exception as e:
            print(f"\nError during evaluation of {question_id}: {e}")
            continue

        # Save score and reasoning back to file
        pred_item[0]['score'] = eval_result['score']
        pred_item[0]['missing_info'] = eval_result['missing_info']
        pred_item[0]['hallucinations'] = eval_result['hallucinations']
        pred_item[0]['evaluation_reasoning'] = eval_result['full_response']

        with open(pred_path, 'w', encoding='utf-8') as ff:
            json.dump(pred_item, ff, indent=4, ensure_ascii=False)

        # Parse and update metrics
        try:
            pred_nums += 1
            gptscore_clean = eval_result['score'].replace('Answer: ', '').strip()

            if '1' in gptscore_clean:
                strict_acc += 1
                relaxed_acc += 1
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
