"""
Stage 1: TransNetV2 Shot Boundary Detection + Middle Frame Extraction

For each video in the test set:
1. Run TransNetV2 to detect shot boundaries
2. Extract the middle frame from each detected shot
3. Save frames in LavaD-compatible format ({:06d}.jpg sequential naming)
4. Generate LavaD annotation file for Stage 2 captioning
"""

import argparse
import json
import os
import sys
import numpy as np

# Add TransNetV2 TF inference to path
TRANSNET_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "external", "TransNetV2", "inference")
sys.path.insert(0, TRANSNET_DIR)

from transnetv2 import TransNetV2


def extract_middle_frames(video_path, scenes, output_dir):
    """
    Extract the middle frame from each detected shot using ffmpeg.
    Saves frames in LavaD's expected format: {output_dir}/{video_name}/{:06d}.jpg
    with sequential numbering (0, 1, 2, ...) mapping to shot indices.
    """
    import ffmpeg

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    frames_dir = os.path.join(output_dir, video_name)
    os.makedirs(frames_dir, exist_ok=True)

    # Get video FPS to convert frame indices to timestamps
    probe = ffmpeg.probe(video_path)
    video_stream = next(s for s in probe["streams"] if s["codec_type"] == "video")
    fps_parts = video_stream["r_frame_rate"].split("/")
    fps = float(fps_parts[0]) / float(fps_parts[1])

    frame_info = []
    for shot_idx, (start_frame, end_frame) in enumerate(scenes):
        middle_frame = (start_frame + end_frame) // 2
        timestamp = middle_frame / fps

        # LavaD-compatible sequential naming
        frame_path = os.path.join(frames_dir, f"{shot_idx:06d}.jpg")

        if not os.path.exists(frame_path):
            try:
                (
                    ffmpeg.input(video_path, ss=timestamp)
                    .output(frame_path, vframes=1, qscale=2)
                    .overwrite_output()
                    .run(capture_stdout=True, capture_stderr=True)
                )
            except ffmpeg.Error as e:
                print(f"  Warning: Failed to extract frame {middle_frame} from {video_name}: {e}")
                continue

        frame_info.append({
            "shot_idx": shot_idx,
            "start": int(start_frame),
            "end": int(end_frame),
            "middle_frame": int(middle_frame),
            "lavad_frame_idx": shot_idx,
            "path": frame_path,
        })

    return frame_info


def process_video(model, video_path, output_dir, threshold=0.5):
    """Run TransNetV2 on a single video and extract middle frames."""
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    metadata_path = os.path.join(output_dir, video_name, "shots.json")

    # Skip if already processed
    if os.path.exists(metadata_path):
        print(f"  Skipping {video_name} (already processed)")
        with open(metadata_path, "r") as f:
            return json.load(f)

    print(f"  Running TransNetV2 on {video_name}...")
    video_frames, single_frame_pred, all_frame_pred = model.predict_video(video_path)

    scenes = TransNetV2.predictions_to_scenes(single_frame_pred, threshold=threshold)
    print(f"  Detected {len(scenes)} shots")

    print(f"  Extracting middle frames...")
    frame_info = extract_middle_frames(video_path, scenes, output_dir)

    metadata = {
        "video": video_name,
        "num_original_frames": len(video_frames),
        "num_shots": len(scenes),
        "num_extracted_frames": len(frame_info),
        "threshold": threshold,
        "frames": frame_info,
    }

    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Saved {len(frame_info)} frames")
    return metadata


def write_lavad_annotations(all_metadata, output_dir):
    """
    Write LavaD-compatible annotation file.
    Format: {video_name} {start_frame} {end_frame} {label}
    Where start=0, end=num_extracted_frames-1, label=0.
    """
    annotation_path = os.path.join(output_dir, "annotations.txt")
    with open(annotation_path, "w") as f:
        for video_name, meta in sorted(all_metadata.items()):
            n = meta["num_extracted_frames"]
            if n > 0:
                f.write(f"{video_name} 0 {n - 1} 0\n")

    print(f"LavaD annotation file written to {annotation_path}")
    return annotation_path


def main():
    parser = argparse.ArgumentParser(description="Stage 1: Shot detection + frame extraction")
    parser.add_argument("--video_dir", type=str, default="./videos/",
                        help="Directory containing video files")
    parser.add_argument("--question_file", type=str, default="./testset_question.json",
                        help="Path to question file (to determine which videos to process)")
    parser.add_argument("--output_dir", type=str, default="./transnet_frames/",
                        help="Output directory for extracted frames")
    parser.add_argument("--transnet_weights", type=str, default=None,
                        help="Path to TransNetV2 weights (default: auto-detect)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Shot boundary detection threshold")
    parser.add_argument("--max_videos", type=int, default=None,
                        help="Process only first N videos (for testing)")
    args = parser.parse_args()

    # Load question file to get video list
    with open(args.question_file, "r") as f:
        questions = json.load(f)

    # Get unique video names
    video_names = sorted(set(item["video"] for item in questions))
    print(f"Found {len(video_names)} unique videos in test set")

    if args.max_videos:
        video_names = video_names[:args.max_videos]
        print(f"Processing only first {args.max_videos} videos")

    # Initialize TransNetV2
    print("Loading TransNetV2 model...")
    model = TransNetV2(args.transnet_weights)

    os.makedirs(args.output_dir, exist_ok=True)
    all_metadata = {}

    for idx, video_name in enumerate(video_names):
        video_path = os.path.join(args.video_dir, video_name + ".mp4")
        if not os.path.exists(video_path):
            print(f"[{idx+1}/{len(video_names)}] Warning: Video not found: {video_path}")
            continue

        print(f"[{idx+1}/{len(video_names)}] Processing {video_name}...")
        metadata = process_video(model, video_path, args.output_dir, args.threshold)
        all_metadata[video_name] = metadata

    # Save combined metadata
    combined_path = os.path.join(args.output_dir, "all_shots.json")
    with open(combined_path, "w") as f:
        json.dump(all_metadata, f, indent=2)

    # Write LavaD-compatible annotation file
    write_lavad_annotations(all_metadata, args.output_dir)

    print(f"\nStage 1 complete. Processed {len(all_metadata)} videos.")
    print(f"Combined metadata: {combined_path}")


if __name__ == "__main__":
    main()
