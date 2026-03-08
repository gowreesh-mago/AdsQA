#!/bin/bash
# ============================================================
# Video Statistics & Correlation Analysis Pipeline
# Extracts video statistics and analyzes correlations with errors
# ============================================================

set -e  # Exit on error

# ============================================================
# Configuration
# ============================================================

# Default paths
VIDEO_DIR="/ivi/zfs/s0/original_homes/gmago/adsqa/AdsQA/AdsQA/videos"
GROUNDTRUTH_FILE="/ivi/zfs/s0/original_homes/gmago/adsqa/AdsQA/AdsQA/testset_groundtruth.json"
VIDEO_STATS_OUTPUT="./video_statistics/video_stats.json"
CACHE_FILE="./video_statistics/cache.json"
NUM_WORKERS=8

# Check if error report is provided as argument
if [ -z "$1" ]; then
    echo "Usage: $0 <error_report.json> [max_videos]"
    echo ""
    echo "Example:"
    echo "  $0 ./error_reports/qwen3vl-8b_report.json"
    echo "  $0 ./error_reports/qwen3vl-8b_report.json 10  # Test with 10 videos"
    echo ""
    exit 1
fi

ERROR_REPORT="$1"
MAX_VIDEOS="${2:-}"  # Optional: limit number of videos for testing

# Derive output paths from error report filename
ERROR_REPORT_NAME=$(basename "$ERROR_REPORT" .json)
CORRELATION_OUTPUT="./correlation_reports/${ERROR_REPORT_NAME}_correlations.json"
PLOTS_DIR="./correlation_reports/${ERROR_REPORT_NAME}_plots"

# ============================================================
# Validation
# ============================================================

echo "========================================================"
echo "Video Statistics & Correlation Analysis Pipeline"
echo "========================================================"
echo ""

# Check if error report exists
if [ ! -f "$ERROR_REPORT" ]; then
    echo "Error: Error report not found: $ERROR_REPORT"
    exit 1
fi

echo "Configuration:"
echo "  Error Report: $ERROR_REPORT"
echo "  Video Directory: $VIDEO_DIR"
echo "  Groundtruth File: $GROUNDTRUTH_FILE"
echo "  Workers: $NUM_WORKERS"
if [ -n "$MAX_VIDEOS" ]; then
    echo "  Max Videos: $MAX_VIDEOS (TEST MODE)"
fi
echo ""

# ============================================================
# Step 1: Extract Video Statistics
# ============================================================

echo "Step 1/2: Extracting video statistics..."
echo "--------------------------------------------------------"

# Check if video stats already exist
if [ -f "$VIDEO_STATS_OUTPUT" ] && [ -z "$MAX_VIDEOS" ]; then
    echo "Video statistics already exist at: $VIDEO_STATS_OUTPUT"
    read -p "Do you want to skip extraction? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping video statistics extraction."
    else
        # Build command
        CMD="python evaluation/extract_video_statistics.py \
          --video-dir \"$VIDEO_DIR\" \
          --groundtruth-file \"$GROUNDTRUTH_FILE\" \
          --output \"$VIDEO_STATS_OUTPUT\" \
          --cache-file \"$CACHE_FILE\" \
          --num-workers $NUM_WORKERS \
          --resume"

        # Add max-videos if specified
        if [ -n "$MAX_VIDEOS" ]; then
            CMD="$CMD --max-videos $MAX_VIDEOS"
        fi

        eval $CMD
    fi
else
    # Build command
    CMD="python evaluation/extract_video_statistics.py \
      --video-dir \"$VIDEO_DIR\" \
      --groundtruth-file \"$GROUNDTRUTH_FILE\" \
      --output \"$VIDEO_STATS_OUTPUT\" \
      --cache-file \"$CACHE_FILE\" \
      --num-workers $NUM_WORKERS \
      --resume"

    # Add max-videos if specified
    if [ -n "$MAX_VIDEOS" ]; then
        CMD="$CMD --max-videos $MAX_VIDEOS"
    fi

    eval $CMD
fi

echo ""
echo "✓ Video statistics extraction complete!"
echo ""

# ============================================================
# Step 2: Correlation Analysis
# ============================================================

echo "Step 2/2: Running correlation analysis..."
echo "--------------------------------------------------------"

python evaluation/correlation_analysis.py \
  --video-stats "$VIDEO_STATS_OUTPUT" \
  --error-report "$ERROR_REPORT" \
  --output-json "$CORRELATION_OUTPUT" \
  --output-plots-dir "$PLOTS_DIR" \
  --top-n 15

echo ""
echo "✓ Correlation analysis complete!"
echo ""

# ============================================================
# Summary
# ============================================================

echo "========================================================"
echo "Pipeline completed successfully!"
echo "========================================================"
echo ""
echo "Outputs:"
echo "  Video Statistics: $VIDEO_STATS_OUTPUT"
echo "  Correlation Report: $CORRELATION_OUTPUT"
echo "  Visualizations: $PLOTS_DIR/"
echo ""
echo "To view correlation report:"
echo "  cat $CORRELATION_OUTPUT | python -m json.tool | less"
echo ""
echo "To view plots:"
echo "  ls $PLOTS_DIR/"
echo "  open $PLOTS_DIR/correlation_heatmap.png  # macOS"
echo ""
