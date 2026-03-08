"""
Extract Video Statistics for AdsQA Evaluation
Extracts comprehensive video statistics (duration, variance, motion, etc.)
for correlation analysis with model error patterns.
"""

import argparse
import json
import os
import time
from datetime import datetime
from multiprocessing import Pool, Manager
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def read_json(jpath):
    """Load JSON file"""
    with open(jpath, 'r') as ff:
        data = json.load(ff)
    return data


def compute_frame_variance(frames):
    """
    Measure visual diversity across frames

    Args:
        frames: List of frame arrays (numpy arrays)

    Returns:
        float: Variance of pixel intensities across frames (normalized 0-1)
    """
    if len(frames) < 2:
        return 0.0

    # Convert to grayscale and compute variance across frames
    gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    frame_stack = np.array(gray_frames, dtype=np.float32) / 255.0

    # Variance across the temporal dimension (frame axis)
    variance = np.var(frame_stack, axis=0).mean()
    return float(variance)


def compute_brightness_stats(frames):
    """
    Compute brightness statistics across frames

    Args:
        frames: List of frame arrays

    Returns:
        dict: {'mean': float, 'std': float, 'min': float, 'max': float}
    """
    if not frames:
        return {'mean': 0, 'std': 0, 'min': 0, 'max': 0}

    # Convert to grayscale and compute brightness
    gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    all_pixels = np.concatenate([f.flatten() for f in gray_frames])

    return {
        'mean': float(np.mean(all_pixels)),
        'std': float(np.std(all_pixels)),
        'min': float(np.min(all_pixels)),
        'max': float(np.max(all_pixels))
    }


def compute_motion_metrics(frames):
    """
    Measure average inter-frame difference (motion intensity)

    Args:
        frames: List of frame arrays

    Returns:
        float: Average inter-frame difference (normalized 0-1)
    """
    if len(frames) < 2:
        return 0.0

    # Convert to grayscale
    gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
                   for f in frames]

    # Compute frame differences
    diffs = []
    for i in range(len(gray_frames) - 1):
        diff = np.abs(gray_frames[i+1] - gray_frames[i])
        diffs.append(np.mean(diff))

    return float(np.mean(diffs))


def compute_color_diversity(frames):
    """
    Measure color diversity using histogram entropy

    Args:
        frames: List of frame arrays

    Returns:
        float: Color diversity score (normalized 0-1)
    """
    if not frames:
        return 0.0

    # Compute color histogram across all frames
    histograms = []
    for frame in frames:
        # Convert to HSV for better color representation
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        hist = hist.flatten() / hist.sum()  # Normalize
        histograms.append(hist)

    # Average histogram across frames
    avg_hist = np.mean(histograms, axis=0)

    # Compute entropy
    avg_hist = avg_hist[avg_hist > 0]  # Remove zeros
    entropy = -np.sum(avg_hist * np.log2(avg_hist))

    # Normalize by max possible entropy (log2(180))
    max_entropy = np.log2(180)
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

    return float(normalized_entropy)


def compute_scene_complexity(frames):
    """
    Measure scene complexity using edge detection

    Args:
        frames: List of frame arrays

    Returns:
        float: Average edge density (normalized 0-1)
    """
    if not frames:
        return 0.0

    edge_densities = []
    for frame in frames:
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Canny edge detection
        edges = cv2.Canny(gray, 100, 200)

        # Compute edge density (ratio of edge pixels)
        edge_density = np.sum(edges > 0) / edges.size
        edge_densities.append(edge_density)

    return float(np.mean(edge_densities))


def extract_single_video(video_path, sample_fps=1):
    """
    Extract statistics from a single video

    Args:
        video_path: Path to video file
        sample_fps: Frame sampling rate (frames per second)

    Returns:
        dict: Video statistics or None if extraction failed
    """
    try:
        start_time = time.time()

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        # Get basic metadata
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if fps <= 0 or total_frames <= 0:
            cap.release()
            return None

        duration = total_frames / fps

        # Sample frames
        sample_interval = max(1, int(fps / sample_fps))
        sampled_frames = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_interval == 0:
                sampled_frames.append(frame)

            frame_idx += 1

        cap.release()

        if not sampled_frames:
            return None

        # Compute statistics on sampled frames
        frame_variance = compute_frame_variance(sampled_frames)
        brightness_stats = compute_brightness_stats(sampled_frames)
        motion_intensity = compute_motion_metrics(sampled_frames)
        color_diversity = compute_color_diversity(sampled_frames)
        scene_complexity = compute_scene_complexity(sampled_frames)

        # Get file size
        file_size_mb = os.path.getsize(video_path) / (1024 ** 2)

        extraction_time = time.time() - start_time

        # Extract video_id from filename
        video_id = os.path.basename(video_path).replace('.mp4', '')

        return {
            'video_id': video_id,
            'duration_seconds': float(duration),
            'total_frames': total_frames,
            'fps': float(fps),
            'resolution': {
                'width': width,
                'height': height
            },
            'file_size_mb': float(file_size_mb),
            'frame_variance': frame_variance,
            'brightness': brightness_stats,
            'motion_intensity': motion_intensity,
            'color_diversity': color_diversity,
            'scene_complexity': scene_complexity,
            'extraction_time_seconds': float(extraction_time),
            'sampled_frames_count': len(sampled_frames)
        }

    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return None


def process_video_wrapper(args):
    """Wrapper for multiprocessing"""
    video_path, sample_fps = args
    return extract_single_video(video_path, sample_fps)


def save_cache(cache_data, cache_file):
    """Save cache to file"""
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(cache_data, f, indent=2, ensure_ascii=False)


def load_cache(cache_file):
    """Load cache from file"""
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load cache: {e}")
    return {}


def process_all_videos(video_list, video_dir, sample_fps, num_workers, cache_file, resume):
    """
    Process all videos with multiprocessing and caching

    Args:
        video_list: List of video IDs
        video_dir: Directory containing videos
        sample_fps: Frame sampling rate
        num_workers: Number of parallel workers
        cache_file: Path to cache file
        resume: Whether to resume from cache

    Returns:
        dict: {video_id: statistics}
    """
    # Load cache if resuming
    statistics = {}
    if resume:
        cache_data = load_cache(cache_file)
        statistics = cache_data.get('statistics', {})
        print(f"Loaded {len(statistics)} videos from cache")

    # Filter out already processed videos
    remaining_videos = [v for v in video_list if v not in statistics]

    if not remaining_videos:
        print("All videos already processed!")
        return statistics

    print(f"Processing {len(remaining_videos)} videos ({len(statistics)} cached)...")

    # Prepare video paths
    video_paths = []
    valid_videos = []

    for video_id in remaining_videos:
        video_path = os.path.join(video_dir, f"{video_id}.mp4")
        if os.path.exists(video_path):
            video_paths.append((video_path, sample_fps))
            valid_videos.append(video_id)
        else:
            print(f"Warning: Video not found: {video_path}")

    if not video_paths:
        print("No valid videos to process!")
        return statistics

    # Process videos with multiprocessing
    failed_count = 0
    cache_counter = 0

    with Pool(num_workers) as pool:
        for i, result in enumerate(tqdm(
            pool.imap(process_video_wrapper, video_paths),
            total=len(video_paths),
            desc="Extracting video statistics"
        )):
            video_id = valid_videos[i]

            if result is not None:
                statistics[video_id] = result
            else:
                failed_count += 1
                print(f"\nFailed to process: {video_id}")

            # Save cache every 50 videos
            cache_counter += 1
            if cache_counter % 50 == 0:
                save_cache({
                    'statistics': statistics,
                    'last_updated': datetime.now().isoformat()
                }, cache_file)

    # Final cache save
    save_cache({
        'statistics': statistics,
        'last_updated': datetime.now().isoformat()
    }, cache_file)

    print(f"\nProcessing complete!")
    print(f"  Successful: {len(statistics)}")
    print(f"  Failed: {failed_count}")

    return statistics


def main():
    parser = argparse.ArgumentParser(
        description="Extract video statistics for AdsQA evaluation"
    )
    parser.add_argument(
        '--video-dir',
        type=str,
        default='/ivi/zfs/s0/original_homes/gmago/adsqa/AdsQA/AdsQA/videos',
        help='Directory containing video files'
    )
    parser.add_argument(
        '--groundtruth-file',
        type=str,
        default='/ivi/zfs/s0/original_homes/gmago/adsqa/AdsQA/AdsQA/testset_groundtruth.json',
        help='Path to groundtruth JSON file (to get list of videos)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./video_statistics/video_stats.json',
        help='Output path for video statistics JSON'
    )
    parser.add_argument(
        '--cache-file',
        type=str,
        default='./video_statistics/cache.json',
        help='Cache file location'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of parallel workers'
    )
    parser.add_argument(
        '--sample-fps',
        type=float,
        default=1.0,
        help='Frame sampling rate (frames per second)'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from cache'
    )
    parser.add_argument(
        '--max-videos',
        type=int,
        default=None,
        help='Process only first N videos (for testing)'
    )

    args = parser.parse_args()

    # Load groundtruth to get list of videos
    print(f"Loading groundtruth from: {args.groundtruth_file}")
    groundtruth = read_json(args.groundtruth_file)

    # Extract unique video IDs
    video_ids = list(set(item['video'] for item in groundtruth))
    print(f"Found {len(video_ids)} unique videos in groundtruth")

    # Limit for testing
    if args.max_videos:
        video_ids = video_ids[:args.max_videos]
        print(f"Testing mode: Processing only {args.max_videos} videos")

    # Process videos
    statistics = process_all_videos(
        video_list=video_ids,
        video_dir=args.video_dir,
        sample_fps=args.sample_fps,
        num_workers=args.num_workers,
        cache_file=args.cache_file,
        resume=args.resume
    )

    # Create output
    output_data = {
        'metadata': {
            'extraction_date': datetime.now().isoformat(),
            'total_videos': len(video_ids),
            'processed_videos': len(statistics),
            'failed_videos': len(video_ids) - len(statistics),
            'sample_fps': args.sample_fps,
            'video_dir': args.video_dir
        },
        'statistics': statistics
    }

    # Save output
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print("VIDEO STATISTICS EXTRACTION COMPLETE")
    print(f"{'='*70}")
    print(f"Total videos in dataset: {output_data['metadata']['total_videos']}")
    print(f"Successfully processed: {output_data['metadata']['processed_videos']}")
    print(f"Failed: {output_data['metadata']['failed_videos']}")
    print(f"Output saved to: {args.output}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
