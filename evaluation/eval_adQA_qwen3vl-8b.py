"""
Video QnA Inference with Qwen3-VL-8B-Instruct
Adapted from eval_adQA.py for Qwen3-VL architecture
"""

import argparse
import time
import os
import json
import random
from qwen_vl_utils import process_vision_info
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch
import numpy as np


def read_json(jpath):
    """Load JSON file"""
    with open(jpath, 'r') as ff:
        data = json.load(ff)
    return data


def get_qwen3vl_response(model, processor, video_path, question):
    """
    Get response from Qwen3-VL-8B-Instruct model

    Args:
        model: Qwen3VLForConditionalGeneration model
        processor: AutoProcessor for Qwen3-VL
        video_path: Path to video file
        question: Question text (includes ASR if available)

    Returns:
        Model's response as string
    """
    # Message format for Qwen3-VL (CRITICAL: includes video metadata parameters)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "total_pixels": 20480 * 32 * 32,  # 671,088,640
                    "min_pixels": 64 * 32 * 32,        # 65,536
                    "max_frames": 2048,
                    "sample_fps": 2
                },
                {"type": "text", "text": question},
            ],
        }
    ]

    # Apply chat template
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Process vision info with metadata (CRITICAL for Qwen3-VL)
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        [messages],
        return_video_kwargs=True,
        image_patch_size=16,
        return_video_metadata=True
    )

    # Unpack video metadata (CRITICAL for Qwen3-VL)
    if video_inputs is not None:
        video_inputs, video_metadatas = zip(*video_inputs)
        video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)
    else:
        video_metadatas = None

    # Process inputs with metadata
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        video_metadata=video_metadatas,
        **video_kwargs,
        do_resize=False,
        return_tensors="pt"
    )

    inputs = inputs.to('cuda')

    # Inference
    generated_ids = model.generate(
        **inputs,
        do_sample=True,
        temperature=0.3,
        max_new_tokens=1524
    )

    # Extract generated tokens
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    # Decode output
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    return output_text[0].strip()


def set_random_seed(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    random.seed(seed)
    np.random.seed(seed)

    print(f"Random seed set to: {seed}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Qwen3-VL-8B inference for AdsQA")
    parser.add_argument('--video_dir', type=str, default='',
                        help='Directory containing video files')
    parser.add_argument('--asr_file', type=str, default='./evaluation/asr_set.json',
                        help='Path to ASR file (asr_set.json)')
    parser.add_argument('--question_file', type=str, default='./testset_question.json',
                        help='Path to question file (testset_question.json)')
    parser.add_argument('--model_dir', type=str, default="Qwen/Qwen3-VL-8B-Instruct",
                        help='Model path or HuggingFace model ID')
    parser.add_argument('--model_name', type=str, default='qwen3vl-8b',
                        help='Prediction file name (saved as {question_id}/{model_name}.json)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Process only first N samples for testing')

    args = parser.parse_args()

    # Set random seed
    set_random_seed(42)

    # Load data
    print("Loading ASR and question data...")
    asr_results_set = read_json(args.asr_file)
    raw_test_data = read_json(args.question_file)

    # Limit samples for testing if specified
    if args.max_samples:
        raw_test_data = raw_test_data[:args.max_samples]
        print(f"Testing mode: Processing only {args.max_samples} samples")

    # Load model and processor
    print(f"Loading model from {args.model_dir}...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_dir,
        torch_dtype="auto",
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(args.model_dir)
    print("Model loaded successfully!")

    # Process each question
    for ii, item in enumerate(raw_test_data):
        print(f"\n[{ii+1}/{len(raw_test_data)}] Processing question {item['question_id']}...")
        start = time.time()

        video_name = item['video']
        question = item['question']
        question_id = item['question_id']

        # Get ASR transcript
        asr = asr_results_set.get(video_name + '.mp4', '')
        if asr != '':
            asr = f"Voiceover: {asr}\n"

        # Skip if prediction already exists
        result_path = os.path.join('./results', question_id, f'{args.model_name}.json')
        if os.path.exists(result_path):
            print(f"  Skipping (already exists): {result_path}")
            continue

        try:
            # Construct video path
            video_path = os.path.join(args.video_dir, video_name + '.mp4')

            if not os.path.exists(video_path):
                print(f"  Warning: Video not found: {video_path}")
                continue

            # Format question with instruction
            input_question = f"{asr}\n\n Question: {question}\n\nYou are permitted to gather any relevant clues, think step by step, to answer this question. Output your thought process (no length limit) and the final answer (using approximately 30 words; longer answers will be truncated) in the following format\n <think>your_thinking</think> <answer>your_answer_within_30_words</answer>"

            # Get model prediction
            print("  Running inference...")
            prediction = get_qwen3vl_response(model, processor, video_path, input_question)

            # Save result
            os.makedirs(os.path.join('./results', question_id), exist_ok=True)

            result = [{
                "prediction": prediction,
                "score": ""
            }]

            with open(result_path, 'w', encoding='utf-8') as ff:
                json.dump(result, ff, ensure_ascii=False, indent=4)

            elapsed = time.time() - start
            print(f"  Completed in {elapsed:.2f}s")
            print(f"  Prediction saved to: {result_path}")

        except Exception as ex:
            print(f"  Error processing {video_name}: {ex}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*50)
    print("Inference completed!")
    print(f"Results saved to: ./results/")
