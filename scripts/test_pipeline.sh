#!/bin/bash
# ============================================================
# Quick Test Pipeline for AdsQA Evaluation
# Tests the full pipeline (inference, evaluation, error report)
# on a small number of samples
# ============================================================

set -e  # Exit on error

# Default number of samples
SAMPLES=${1:-10}

# Data paths
VIDEO_DIR="/ivi/zfs/s0/original_homes/gmago/adsqa/AdsQA/AdsQA/videos"
ASR_FILE="./evaluation/asr_set.json"
QUESTION_FILE="/ivi/zfs/s0/original_homes/gmago/adsqa/AdsQA/AdsQA/testset_question.json"
GROUNDTRUTH_FILE="/ivi/zfs/s0/original_homes/gmago/adsqa/AdsQA/AdsQA/testset_groundtruth.json"

echo "========================================================"
echo "AdsQA Evaluation Pipeline - Test Mode"
echo "Testing with $SAMPLES samples"
echo "========================================================"
echo ""

# Step 1: Video Inference
echo "Step 1/3: Running video inference with Qwen3-VL-8B..."
echo "--------------------------------------------------------"
python evaluation/eval_adQA_qwen3vl-8b.py \
  --video_dir "$VIDEO_DIR" \
  --asr_file "$ASR_FILE" \
  --question_file "$QUESTION_FILE" \
  --model_name qwen3vl-8b \
  --max_samples $SAMPLES

echo ""
echo "✓ Inference complete!"
echo ""

# Step 2: Evaluation
echo "Step 2/3: Running evaluation with Qwen2.5-7B-Instruct..."
echo "--------------------------------------------------------"
python evaluation/model_evaluation_qwen.py \
  --eval_name qwen3vl-8b.json \
  --test_file "$GROUNDTRUTH_FILE" \
  --results_dir ./results/ \
  --max_samples $SAMPLES

echo ""
echo "✓ Evaluation complete!"
echo ""

# Step 3: Error Report
echo "Step 3/3: Generating error report..."
echo "--------------------------------------------------------"
python evaluation/generate_error_report.py \
  --results_dir ./results/ \
  --eval_name qwen3vl-8b.json \
  --groundtruth_file "$GROUNDTRUTH_FILE" \
  --output_path ./error_reports/qwen3vl-8b_test_${SAMPLES}_report.json \
  --max_samples $SAMPLES

echo ""
echo "✓ Error report complete!"
echo ""

echo "========================================================"
echo "Test pipeline completed successfully!"
echo "========================================================"
echo ""
echo "Results:"
echo "  - Predictions: ./results/{question_id}/qwen3vl-8b.json"
echo "  - Error Report: ./error_reports/qwen3vl-8b_test_${SAMPLES}_report.json"
echo ""
echo "To view the error report:"
echo "  cat ./error_reports/qwen3vl-8b_test_${SAMPLES}_report.json | python -m json.tool | less"
echo ""
