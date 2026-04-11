#!/bin/bash
# ============================================================
# TransNet-LavaD Two-Stage Pipeline
# Run from the repo root: bash evaluation/transnet_lavad/run_pipeline.sh
# ============================================================

set -e

# --- Configuration (override via environment variables) ---
VIDEO_DIR="${VIDEO_DIR:-./videos/}"
QUESTION_FILE="${QUESTION_FILE:-./testset_question.json}"
ASR_FILE="${ASR_FILE:-./evaluation/asr_set.json}"
FRAMES_DIR="${FRAMES_DIR:-./transnet_frames/}"
CAPTIONS_DIR="${CAPTIONS_DIR:-./transnet_captions/}"
RESULTS_DIR="${RESULTS_DIR:-./results/}"
MODEL_NAME="${MODEL_NAME:-transnet-lavad}"

# Stage 1 settings
TRANSNET_THRESHOLD="${TRANSNET_THRESHOLD:-0.5}"
MAX_VIDEOS="${MAX_VIDEOS:-}"  # empty = all videos

# Stage 2 settings (LavaD captioner + Qwen QnA)
CAPTIONER_MODEL="${CAPTIONER_MODEL:-Salesforce/blip2-opt-6.7b-coco}"
QWEN_MODEL="${QWEN_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
CAPTION_BATCH_SIZE="${CAPTION_BATCH_SIZE:-8}"
DTYPE="${DTYPE:-float16}"
MAX_SAMPLES="${MAX_SAMPLES:-}"  # empty = all samples

echo "============================================================"
echo "TransNet-LavaD Pipeline"
echo "============================================================"
echo "Video dir:        $VIDEO_DIR"
echo "Question file:    $QUESTION_FILE"
echo "Frames output:    $FRAMES_DIR"
echo "Captions output:  $CAPTIONS_DIR"
echo "Results output:   $RESULTS_DIR"
echo "Captioner:        $CAPTIONER_MODEL"
echo "QnA model:        $QWEN_MODEL"
echo "Model name:       $MODEL_NAME"
echo "============================================================"

# --- Stage 1: TransNetV2 Shot Detection + Frame Extraction ---
echo ""
echo ">>> Stage 1: TransNetV2 Shot Detection + Middle Frame Extraction"
echo "------------------------------------------------------------"

STAGE1_CMD="python evaluation/transnet_lavad/stage1_shot_detection.py \
    --video_dir $VIDEO_DIR \
    --question_file $QUESTION_FILE \
    --output_dir $FRAMES_DIR \
    --threshold $TRANSNET_THRESHOLD"

if [ -n "$MAX_VIDEOS" ]; then
    STAGE1_CMD="$STAGE1_CMD --max_videos $MAX_VIDEOS"
fi

eval $STAGE1_CMD

# --- Stage 2: LavaD BLIP-2 Captioning + Qwen QnA ---
echo ""
echo ">>> Stage 2: LavaD VLM Captioning (BLIP-2) + Qwen QnA"
echo "------------------------------------------------------------"

STAGE2_CMD="python evaluation/transnet_lavad/stage2_caption_qna.py \
    --frames_dir $FRAMES_DIR \
    --captions_dir $CAPTIONS_DIR \
    --question_file $QUESTION_FILE \
    --asr_file $ASR_FILE \
    --results_dir $RESULTS_DIR \
    --model_name $MODEL_NAME \
    --captioner_model $CAPTIONER_MODEL \
    --qwen_model $QWEN_MODEL \
    --caption_batch_size $CAPTION_BATCH_SIZE \
    --dtype $DTYPE"

if [ -n "$MAX_SAMPLES" ]; then
    STAGE2_CMD="$STAGE2_CMD --max_samples $MAX_SAMPLES"
fi

eval $STAGE2_CMD

echo ""
echo "============================================================"
echo "Pipeline complete!"
echo "Results saved to: $RESULTS_DIR"
echo "Captions saved to: $CAPTIONS_DIR"
echo "============================================================"
