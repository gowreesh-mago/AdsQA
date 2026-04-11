"""
Stage 2: LavaD VLM Captioning + Qwen Instruct QnA

Uses LavaD's ImageCaptioner (BLIP-2) to generate per-frame captions,
then uses Qwen2.5-7B-Instruct for question answering.

Pipeline:
1. Load extracted frames (from Stage 1) for each video
2. Use LavaD's ImageCaptioner to generate captions per frame
3. Optionally summarize captions temporally (LavaD-style)
4. Use Qwen instruct model to answer questions given captions + ASR
5. Save results in standard AdsQA format
"""

import argparse
import json
import os
import sys
import time
import random

import numpy as np
import torch
from pathlib import Path

# --- Add LavaD to path ---
LAVAD_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "external", "lavad")
sys.path.insert(0, LAVAD_DIR)

from src.models.image_captioner import ImageCaptioner
from src.data.video_record import VideoRecord
from transformers import AutoModelForCausalLM, AutoTokenizer


def set_random_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)


def caption_video_with_lavad(captioner, video_name, frames_dir):
    """
    Use LavaD's ImageCaptioner to generate captions for extracted frames.
    Frames are expected in LavaD format: {frames_dir}/{video_name}/{:06d}.jpg
    """
    # Count frames in directory
    video_frames_dir = os.path.join(frames_dir, video_name)
    frame_files = sorted(
        f for f in os.listdir(video_frames_dir)
        if f.endswith(".jpg") and f != "shots.json"
    )
    num_frames = len(frame_files)

    if num_frames == 0:
        return {}

    # Create a VideoRecord for LavaD (row format: [name, start, end, label])
    row = [video_name, "0", str(num_frames - 1), "0"]
    video_record = VideoRecord(row, frames_dir)

    # Use LavaD's captioner - it saves to output_dir/{video_name}.json
    captioner.process_video(video_record)

    # Read back the captions
    caption_path = captioner.output_dir / f"{video_name}.json"
    if caption_path.exists():
        with open(caption_path, "r") as f:
            return json.load(f)

    return {}


def load_qwen_model(model_name):
    """Load Qwen instruct model for QnA."""
    print(f"Loading Qwen model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print("Qwen model loaded.")
    return tokenizer, model


def summarize_captions(captions_dict, shot_metadata):
    """
    Build an ordered list of captions with shot context.
    Uses shot metadata to add temporal grounding (LavaD-style).
    """
    frames = shot_metadata.get("frames", [])
    ordered_captions = []

    for frame_info in frames:
        lavad_idx = str(frame_info["lavad_frame_idx"])
        caption = captions_dict.get(lavad_idx, "")
        if caption:
            ordered_captions.append({
                "shot_idx": frame_info["shot_idx"],
                "start_frame": frame_info["start"],
                "end_frame": frame_info["end"],
                "caption": caption,
            })

    return ordered_captions


def build_qna_prompt(ordered_captions, asr, question):
    """
    Build the prompt for Qwen combining LavaD captions + ASR + question.
    Follows LavaD's approach of using scene descriptions as context.
    """
    # Format captions as numbered scene descriptions with temporal context
    caption_lines = []
    for cap in ordered_captions:
        caption_lines.append(
            f"Shot {cap['shot_idx']+1} (frames {cap['start_frame']}-{cap['end_frame']}): "
            f"{cap['caption']}"
        )
    caption_text = "\n".join(caption_lines)

    parts = []
    parts.append(
        "You are analyzing a video advertisement based on visual scene descriptions "
        "extracted from key frames and an audio transcript.\n"
    )
    parts.append(f"--- Visual Scene Descriptions (from {len(ordered_captions)} shots) ---")
    parts.append(caption_text)
    parts.append("")

    if asr:
        parts.append(f"--- Voiceover Transcript ---\n{asr}\n")

    parts.append(f"--- Question ---\n{question}\n")
    parts.append(
        "You are permitted to gather any relevant clues, think step by step, "
        "to answer this question. Output your thought process (no length limit) "
        "and the final answer (using approximately 30 words; longer answers will "
        "be truncated) in the following format\n"
        "<think>your_thinking</think> <answer>your_answer_within_30_words</answer>"
    )

    return "\n".join(parts)


def get_qwen_response(tokenizer, model, prompt, max_new_tokens=1524, temperature=0.1):
    """Get response from Qwen instruct model."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=False,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output = tokenizer.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output[0].strip()


def main():
    parser = argparse.ArgumentParser(description="Stage 2: LavaD captioning + Qwen QnA")
    parser.add_argument("--frames_dir", type=str, default="./transnet_frames/",
                        help="Directory with extracted frames from Stage 1")
    parser.add_argument("--question_file", type=str, default="./testset_question.json",
                        help="Path to question file")
    parser.add_argument("--asr_file", type=str, default="./evaluation/asr_set.json",
                        help="Path to ASR file")
    parser.add_argument("--results_dir", type=str, default="./results/",
                        help="Output directory for results")
    parser.add_argument("--captions_dir", type=str, default="./transnet_captions/",
                        help="Output directory for LavaD captions")
    parser.add_argument("--model_name", type=str, default="transnet-lavad",
                        help="Model name for result files")

    # LavaD captioner settings
    parser.add_argument("--captioner_model", type=str,
                        default="Salesforce/blip2-opt-6.7b-coco",
                        help="BLIP-2 model for LavaD captioning")
    parser.add_argument("--caption_batch_size", type=int, default=8,
                        help="Batch size for LavaD ImageCaptioner")
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "float32"])

    # Qwen QnA settings
    parser.add_argument("--qwen_model", type=str,
                        default="Qwen/Qwen2.5-7B-Instruct",
                        help="Qwen model for QnA")
    parser.add_argument("--max_new_tokens", type=int, default=1524)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Process only first N samples (for testing)")
    args = parser.parse_args()

    set_random_seed(42)

    # Load data
    print("Loading data...")
    with open(args.question_file, "r") as f:
        questions = json.load(f)
    with open(args.asr_file, "r") as f:
        asr_data = json.load(f)

    # Load shot metadata from Stage 1
    shots_meta_path = os.path.join(args.frames_dir, "all_shots.json")
    with open(shots_meta_path, "r") as f:
        all_shots = json.load(f)

    if args.max_samples:
        questions = questions[:args.max_samples]
        print(f"Testing mode: Processing only {args.max_samples} samples")

    # Initialize LavaD ImageCaptioner
    # frame_interval=1 because we only have shot middle frames (no redundancy to skip)
    print("Initializing LavaD ImageCaptioner...")
    captioner = ImageCaptioner(
        batch_size=args.caption_batch_size,
        frame_interval=1,  # process every frame (each is a unique shot)
        imagefile_template="{:06d}.jpg",
        pretrained_model_name=args.captioner_model,
        dtype_str=args.dtype,
        output_dir=args.captions_dir,
    )
    print(f"LavaD captioner loaded: {args.captioner_model}")

    # Load Qwen model
    qwen_tokenizer, qwen_model = load_qwen_model(args.qwen_model)

    # Caption all videos first (LavaD captioner writes to captions_dir)
    video_names = sorted(set(item["video"] for item in questions))
    captions_cache = {}

    for video_name in video_names:
        if video_name not in all_shots:
            continue

        caption_path = Path(args.captions_dir) / f"{video_name}.json"
        if caption_path.exists():
            print(f"  Captions exist for {video_name}, loading from cache...")
            with open(caption_path, "r") as f:
                captions_cache[video_name] = json.load(f)
        else:
            print(f"  Captioning {video_name} with LavaD...")
            captions = caption_video_with_lavad(captioner, video_name, args.frames_dir)
            captions_cache[video_name] = captions

    print(f"\nCaptioned {len(captions_cache)} videos. Starting QnA...\n")

    # Process each question
    for idx, item in enumerate(questions):
        question_id = item["question_id"]
        video_name = item["video"]
        question = item["question"]

        print(f"[{idx+1}/{len(questions)}] Processing {question_id} ({video_name})...")
        start = time.time()

        # Skip if result already exists
        result_path = os.path.join(args.results_dir, question_id, f"{args.model_name}.json")
        if os.path.exists(result_path):
            print(f"  Skipping (already exists)")
            continue

        if video_name not in all_shots or video_name not in captions_cache:
            print(f"  Warning: No data for {video_name}, skipping")
            continue

        shot_metadata = all_shots[video_name]
        captions_dict = captions_cache[video_name]

        if not captions_dict:
            print(f"  Warning: No captions for {video_name}, skipping")
            continue

        # Build ordered captions with shot context
        ordered_captions = summarize_captions(captions_dict, shot_metadata)

        # Get ASR
        asr = asr_data.get(video_name + ".mp4", "")

        # Build prompt and get answer from Qwen
        prompt = build_qna_prompt(ordered_captions, asr, question)

        print(f"  Running Qwen QnA ({len(ordered_captions)} shot captions)...")
        prediction = get_qwen_response(
            qwen_tokenizer, qwen_model, prompt,
            max_new_tokens=args.max_new_tokens,
        )

        # Save result in standard AdsQA format
        os.makedirs(os.path.join(args.results_dir, question_id), exist_ok=True)

        caption_list = [c["caption"] for c in ordered_captions]
        result = [{
            "prediction": prediction,
            "captions": caption_list,
            "num_shots": shot_metadata["num_shots"],
            "score": "",
        }]

        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)

        elapsed = time.time() - start
        print(f"  Completed in {elapsed:.2f}s -> {result_path}")

    print(f"\nStage 2 complete. Results saved to {args.results_dir}")


if __name__ == "__main__":
    main()
