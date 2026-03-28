"""
Error Analysis for Video Statistics vs. Model Errors
Analyzes how video characteristics relate to different error types.
Shows clear, interpretable comparisons (e.g., "1-minute videos have more hallucinations than 30-second videos").
"""

import argparse
import json
import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


def convert_to_native_types(obj):
    """
    Recursively convert numpy types to native Python types for JSON serialization

    Args:
        obj: Object to convert (can be dict, list, numpy type, etc.)

    Returns:
        Object with all numpy types converted to Python native types
    """
    if isinstance(obj, dict):
        return {k: convert_to_native_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def read_json(jpath):
    """Load JSON file"""
    with open(jpath, 'r') as ff:
        data = json.load(ff)
    return data


def merge_data(video_stats_path, error_report_path, groundtruth_path=None):
    """
    Merge video statistics with error report data

    Args:
        video_stats_path: Path to video statistics JSON
        error_report_path: Path to error report JSON
        groundtruth_path: Optional path to groundtruth JSON (to include perfect score questions)

    Returns:
        tuple: (question_df, video_df, merged_df)
    """
    print("Loading data...")

    # Load video statistics
    video_stats_data = read_json(video_stats_path)
    video_stats = video_stats_data['statistics']

    # Convert to DataFrame
    video_records = []
    for video_id, stats in video_stats.items():
        record = {
            'video_id': video_id,
            'duration_seconds': stats['duration_seconds'],
            'total_frames': stats['total_frames'],
            'fps': stats['fps'],
            'resolution_width': stats['resolution']['width'],
            'resolution_height': stats['resolution']['height'],
            'resolution_area': stats['resolution']['width'] * stats['resolution']['height'],
            'file_size_mb': stats['file_size_mb'],
            'frame_variance': stats['frame_variance'],
            'brightness_mean': stats['brightness']['mean'],
            'brightness_std': stats['brightness']['std'],
            'brightness_min': stats['brightness']['min'],
            'brightness_max': stats['brightness']['max'],
            'motion_intensity': stats['motion_intensity'],
            'color_diversity': stats['color_diversity'],
            'scene_complexity': stats['scene_complexity']
        }
        video_records.append(record)

    video_df = pd.DataFrame(video_records)

    # Load error report
    error_report = read_json(error_report_path)

    # Extract question data from detailed_errors
    question_records = []
    for error_entry in error_report.get('detailed_errors', []):
        question_id = error_entry['question_id']
        # Extract video_id from question_id (format: {video_hash}_{question_num})
        video_id = question_id.rsplit('_', 1)[0]

        # Parse error types
        errors = error_entry.get('errors', [])

        # Count error types (verification already done in evaluation step)
        has_hallucination = any(e['category'] == 'Hallucination' for e in errors)
        has_missing_info = any(e['category'] == 'Missing Information' for e in errors)
        is_partial_match = any(e['category'] == 'Partial Match' for e in errors)

        # Get question types
        question_types = error_entry.get('question_type', [])
        primary_question_type = question_types[0] if question_types else 'Unknown'

        question_records.append({
            'question_id': question_id,
            'video_id': video_id,
            'score': error_entry['score'],
            'has_hallucination': has_hallucination,
            'has_missing_info': has_missing_info,
            'is_partial_match': is_partial_match,
            'question_type': primary_question_type
        })

    question_df = pd.DataFrame(question_records)

    print(f"  Loaded {len(video_df)} video statistics")
    print(f"  Loaded {len(question_df)} question error records from detailed_errors")

    # Add perfect score questions if groundtruth is provided
    if groundtruth_path:
        print(f"  Loading groundtruth to identify perfect score questions...")
        groundtruth = read_json(groundtruth_path)

        # Get all question IDs from groundtruth
        all_question_ids = set(item['question_id'] for item in groundtruth)
        error_question_ids = set(q['question_id'] for q in question_records)

        # Perfect score questions = all questions - error questions
        perfect_question_ids = all_question_ids - error_question_ids

        print(f"  Found {len(perfect_question_ids)} perfect score questions")

        # Add perfect score questions
        perfect_records = []
        for item in groundtruth:
            if item['question_id'] in perfect_question_ids:
                video_id = item['question_id'].rsplit('_', 1)[0]

                question_types = item.get('question_type', [])
                primary_question_type = question_types[0] if question_types else 'Unknown'

                perfect_records.append({
                    'question_id': item['question_id'],
                    'video_id': video_id,
                    'score': 1.0,
                    'has_hallucination': False,
                    'has_missing_info': False,
                    'is_partial_match': False,
                    'question_type': primary_question_type
                })

        # Combine with error questions
        question_df = pd.DataFrame(question_records + perfect_records)
        print(f"  Total questions after adding perfect scores: {len(question_df)}")

    # Merge on video_id
    merged_df = pd.merge(question_df, video_df, on='video_id', how='left')

    print(f"  Merged dataset: {len(merged_df)} questions with video stats")
    print(f"  Questions without video stats: {merged_df['duration_seconds'].isna().sum()}")

    # Drop rows without video stats
    merged_df = merged_df.dropna(subset=['duration_seconds'])

    return question_df, video_df, merged_df


def analyze_error_rates_by_video_features(merged_df):
    """
    Analyze error rates binned by video characteristics
    Shows clear comparisons like "1-minute videos have X% hallucinations vs Y% for 30-second videos"

    Args:
        merged_df: Merged DataFrame

    Returns:
        dict: Analysis results with error rates per feature bin
    """
    print("\nAnalyzing error rates by video features...")

    results = {}

    # Define features to analyze and their bin labels
    features_config = {
        'duration_seconds': {
            'bins': [0, 15, 30, 45, 60, float('inf')],
            'labels': ['0-15s', '15-30s', '30-45s', '45-60s', '>60s']
        },
        'frame_variance': {
            'bins': 4,  # Quartiles
            'labels': ['Low', 'Medium-Low', 'Medium-High', 'High']
        },
        'motion_intensity': {
            'bins': 4,
            'labels': ['Low', 'Medium-Low', 'Medium-High', 'High']
        },
        'scene_complexity': {
            'bins': 4,
            'labels': ['Low', 'Medium-Low', 'Medium-High', 'High']
        },
        'brightness_mean': {
            'bins': 4,
            'labels': ['Low', 'Medium-Low', 'Medium-High', 'High']
        }
    }

    error_types = {
        'hallucination': 'has_hallucination',
        'missing_info': 'has_missing_info',
        'partial_match': 'is_partial_match'
    }

    for feature, config in features_config.items():
        if feature not in merged_df.columns:
            continue

        # Create bins
        if isinstance(config['bins'], list):
            merged_df[f'{feature}_bin'] = pd.cut(
                merged_df[feature],
                bins=config['bins'],
                labels=config['labels'],
                include_lowest=True
            )
        else:
            merged_df[f'{feature}_bin'] = pd.qcut(
                merged_df[feature],
                q=config['bins'],
                labels=config['labels'],
                duplicates='drop'
            )

        feature_results = {}

        # For each bin, calculate error rates
        for bin_label in merged_df[f'{feature}_bin'].dropna().unique():
            bin_data = merged_df[merged_df[f'{feature}_bin'] == bin_label]

            bin_stats = {
                'count': len(bin_data),
                'mean_score': float(bin_data['score'].mean()),
                'error_rates': {}
            }

            # Calculate error rates for each error type
            for error_name, error_col in error_types.items():
                if error_col in bin_data.columns:
                    error_rate = (bin_data[error_col].sum() / len(bin_data) * 100) if len(bin_data) > 0 else 0
                    bin_stats['error_rates'][error_name] = float(error_rate)

            # Add feature value range for this bin
            bin_stats['feature_range'] = {
                'min': float(bin_data[feature].min()),
                'max': float(bin_data[feature].max()),
                'mean': float(bin_data[feature].mean())
            }

            feature_results[str(bin_label)] = bin_stats

        results[feature] = feature_results

    return results


def plot_error_rates_by_features(merged_df, output_dir):
    """
    Plot error rates by video feature bins
    Shows how different video characteristics affect error rates

    Args:
        merged_df: Merged DataFrame
        output_dir: Output directory
    """
    print("\nGenerating error rate comparison plots...")

    # Check if we have any perfect score questions
    total_questions = len(merged_df)
    perfect_questions = (merged_df['score'] == 1.0).sum()
    error_questions = (merged_df['score'] < 1.0).sum()

    print(f"  Dataset composition: {total_questions} total, {perfect_questions} perfect (score=1.0), {error_questions} with errors")

    if perfect_questions == 0:
        print(f"  WARNING: No perfect score questions in dataset!")
        print(f"  This means error rates will be 100% for common error types.")
        print(f"  The 'scores by features' plot will be more informative.")

    error_types = {
        'Hallucination': 'has_hallucination',
        'Missing Info': 'has_missing_info',
        'Partial Match': 'is_partial_match'
    }

    features_config = {
        'duration_seconds': {
            'bins': [0, 15, 30, 45, 60, float('inf')],
            'labels': ['0-15s', '15-30s', '30-45s', '45-60s', '>60s'],
            'title': 'Video Duration'
        },
        'frame_variance': {
            'bins': 4,
            'labels': ['Low', 'Med-Low', 'Med-High', 'High'],
            'title': 'Frame Variance (Visual Diversity)'
        },
        'motion_intensity': {
            'bins': 4,
            'labels': ['Low', 'Med-Low', 'Med-High', 'High'],
            'title': 'Motion Intensity'
        },
        'scene_complexity': {
            'bins': 4,
            'labels': ['Low', 'Med-Low', 'Med-High', 'High'],
            'title': 'Scene Complexity'
        }
    }

    for feature, config in features_config.items():
        if feature not in merged_df.columns:
            continue

        # Create bins
        if isinstance(config['bins'], list):
            merged_df[f'{feature}_bin'] = pd.cut(
                merged_df[feature],
                bins=config['bins'],
                labels=config['labels'],
                include_lowest=True
            )
        else:
            merged_df[f'{feature}_bin'] = pd.qcut(
                merged_df[feature],
                q=config['bins'],
                labels=config['labels'],
                duplicates='drop'
            )

        # Calculate error rates for each bin
        fig, ax = plt.subplots(figsize=(14, 7))

        # Sort bin labels in logical order
        if feature == 'duration_seconds':
            bin_order = ['0-15s', '15-30s', '30-45s', '45-60s', '>60s']
            bin_labels = [b for b in bin_order if b in merged_df[f'{feature}_bin'].dropna().unique()]
        else:
            bin_labels = ['Low', 'Med-Low', 'Med-High', 'High']
            bin_labels = [b for b in bin_labels if b in merged_df[f'{feature}_bin'].dropna().unique()]

        x = np.arange(len(bin_labels))
        width = 0.15

        for i, (error_name, error_col) in enumerate(error_types.items()):
            if error_col not in merged_df.columns:
                continue

            error_rates = []
            counts = []
            for bin_label in bin_labels:
                bin_data = merged_df[merged_df[f'{feature}_bin'] == bin_label]
                if len(bin_data) > 0:
                    # Calculate as percentage of ALL questions in this bin
                    error_count = bin_data[error_col].sum()
                    error_rate = (error_count / len(bin_data) * 100)
                    error_rates.append(error_rate)
                    counts.append(f"n={len(bin_data)}")
                else:
                    error_rates.append(0)
                    counts.append("n=0")

            offset = width * (i - len(error_types)/2)
            bars = ax.bar(x + offset, error_rates, width, label=error_name, alpha=0.8)

            # Add value labels on bars (only if > 5%)
            for j, bar in enumerate(bars):
                height = bar.get_height()
                if height > 5:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}%',
                           ha='center', va='bottom', fontsize=8)

        # Add sample sizes below x-axis
        for i, bin_label in enumerate(bin_labels):
            bin_data = merged_df[merged_df[f'{feature}_bin'] == bin_label]
            ax.text(i, -5, f'n={len(bin_data)}', ha='center', fontsize=9, color='gray')

        ax.set_xlabel(config['title'], fontsize=13)
        ax.set_ylabel('Error Rate (% of all questions in bin)', fontsize=13)
        ax.set_title(f'Error Rates by {config["title"]}\n(% of questions with each error type)', fontsize=15)
        ax.set_xticks(x)
        ax.set_xticklabels(bin_labels)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 105)  # Give space for labels
        plt.tight_layout()

        output_path = os.path.join(output_dir, f'error_rates_by_{feature}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved error rates by {feature} plot")


def plot_score_by_features(merged_df, output_dir):
    """
    Plot average score by video feature bins

    Args:
        merged_df: Merged DataFrame
        output_dir: Output directory
    """
    print("\nGenerating score comparison plots...")

    features_config = {
        'duration_seconds': {
            'bins': [0, 15, 30, 45, 60, float('inf')],
            'labels': ['0-15s', '15-30s', '30-45s', '45-60s', '>60s'],
            'title': 'Video Duration'
        },
        'frame_variance': {
            'bins': 4,
            'labels': ['Low', 'Med-Low', 'Med-High', 'High'],
            'title': 'Frame Variance'
        },
        'motion_intensity': {
            'bins': 4,
            'labels': ['Low', 'Med-Low', 'Med-High', 'High'],
            'title': 'Motion Intensity'
        },
        'scene_complexity': {
            'bins': 4,
            'labels': ['Low', 'Med-Low', 'Med-High', 'High'],
            'title': 'Scene Complexity'
        }
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, (feature, config) in enumerate(features_config.items()):
        if feature not in merged_df.columns or idx >= len(axes):
            continue

        ax = axes[idx]

        # Create bins
        if isinstance(config['bins'], list):
            merged_df[f'{feature}_bin'] = pd.cut(
                merged_df[feature],
                bins=config['bins'],
                labels=config['labels'],
                include_lowest=True
            )
        else:
            merged_df[f'{feature}_bin'] = pd.qcut(
                merged_df[feature],
                q=config['bins'],
                labels=config['labels'],
                duplicates='drop'
            )

        # Calculate mean scores and counts
        bin_labels = merged_df[f'{feature}_bin'].dropna().unique()
        mean_scores = []
        counts = []

        for bin_label in bin_labels:
            bin_data = merged_df[merged_df[f'{feature}_bin'] == bin_label]
            mean_scores.append(bin_data['score'].mean())
            counts.append(len(bin_data))

        # Bar plot
        bars = ax.bar(range(len(bin_labels)), mean_scores, alpha=0.7, edgecolor='black')

        # Color bars by score
        for bar, score in zip(bars, mean_scores):
            if score >= 0.8:
                bar.set_color('#2ca02c')  # Green
            elif score >= 0.5:
                bar.set_color('#ff7f0e')  # Orange
            else:
                bar.set_color('#d62728')  # Red

        # Add labels
        for i, (bar, count) in enumerate(zip(bars, counts)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}\n(n={count})',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_xlabel(config['title'], fontsize=12)
        ax.set_ylabel('Average Score', fontsize=12)
        ax.set_title(f'Average Score by {config["title"]}', fontsize=13)
        ax.set_xticks(range(len(bin_labels)))
        ax.set_xticklabels(bin_labels, rotation=15, ha='right')
        ax.set_ylim(0, 1.1)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Threshold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend()

    plt.suptitle('Average Scores by Video Characteristics', fontsize=16)
    plt.tight_layout()

    output_path = os.path.join(output_dir, 'scores_by_video_features.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved scores by features plot")


def plot_error_report_basics(merged_df, error_report_path, output_dir):
    """
    Plot basic error report statistics

    Args:
        merged_df: Merged DataFrame
        error_report_path: Path to error report JSON
        output_dir: Output directory
    """
    print("\nGenerating basic error report plots...")

    # Load error report for summary stats
    error_report = read_json(error_report_path)

    # 1. Score Distribution
    plt.figure(figsize=(10, 6))
    score_counts = merged_df['score'].value_counts().sort_index()
    colors = ['#d62728', '#ff7f0e', '#2ca02c']  # red, orange, green
    bars = plt.bar(score_counts.index, score_counts.values, color=colors, alpha=0.7, edgecolor='black')

    # Add count labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.xlabel('Score', fontsize=14)
    plt.ylabel('Number of Questions', fontsize=14)
    plt.title('Score Distribution', fontsize=16)
    plt.xticks([0, 0.5, 1.0], ['0\n(Complete\nMismatch)', '0.5\n(Partial\nMatch)', '1.0\n(Perfect)'])
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'score_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved score distribution plot")

    # 2. Error Category Frequency
    plt.figure(figsize=(12, 6))
    error_categories = {
        'Hallucination': merged_df['has_hallucination'].sum(),
        'Missing Information': merged_df['has_missing_info'].sum(),
        'Partial Match': merged_df['is_partial_match'].sum()
    }

    categories = list(error_categories.keys())
    counts = list(error_categories.values())
    colors_palette = ['#e74c3c', '#3498db', '#1abc9c']

    bars = plt.bar(categories, counts, color=colors_palette, alpha=0.7, edgecolor='black')

    # Add count labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.xlabel('Error Category', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.title('Error Category Frequency', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_category_frequency.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved error category frequency plot")

    # 3. Error Categories by Question Type
    if 'question_type' in merged_df.columns:
        question_types = merged_df['question_type'].unique()
        question_types = [qt for qt in question_types if qt != 'Unknown']

        if len(question_types) > 0:
            fig, ax = plt.subplots(figsize=(14, 6))

            error_cols = {
                'Hallucination': 'has_hallucination',
                'Missing Info': 'has_missing_info',
                'Partial Match': 'is_partial_match'
            }

            x = np.arange(len(question_types))
            width = 0.15

            for i, (label, col) in enumerate(error_cols.items()):
                counts = [merged_df[merged_df['question_type'] == qt][col].sum()
                         for qt in question_types]
                offset = width * (i - len(error_cols)/2)
                ax.bar(x + offset, counts, width, label=label, alpha=0.8)

            ax.set_xlabel('Question Type', fontsize=13)
            ax.set_ylabel('Error Count', fontsize=13)
            ax.set_title('Error Categories by Question Type', fontsize=15)
            ax.set_xticks(x)
            ax.set_xticklabels(question_types, rotation=45, ha='right')
            ax.legend(loc='upper right', fontsize=10)
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'errors_by_question_type.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  Saved errors by question type plot")

    # 4. Score vs Error Type Heatmap
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create matrix: rows = scores, cols = error types
    scores = [0, 0.5, 1.0]
    error_types = ['Hallucination', 'Missing Info', 'Partial Match']
    error_col_map = {
        'Hallucination': 'has_hallucination',
        'Missing Info': 'has_missing_info',
        'Partial Match': 'is_partial_match'
    }

    matrix = []
    for score in scores:
        row = []
        score_df = merged_df[merged_df['score'] == score]
        total_count = len(score_df)
        for error_type in error_types:
            col = error_col_map[error_type]
            error_count = score_df[col].sum()
            percentage = (error_count / total_count * 100) if total_count > 0 else 0
            row.append(percentage)
        matrix.append(row)

    sns.heatmap(matrix, annot=True, fmt='.1f', cmap='YlOrRd',
                xticklabels=error_types, yticklabels=['Score 0', 'Score 0.5', 'Score 1.0'],
                cbar_kws={'label': 'Percentage (%)'})
    plt.title('Error Type Prevalence by Score', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_prevalence_by_score.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved error prevalence by score heatmap")


def generate_visualizations(merged_df, output_dir, error_report_path):
    """
    Generate all visualizations (simplified - no correlation plots)

    Args:
        merged_df: Merged DataFrame
        output_dir: Output directory
        error_report_path: Path to error report JSON
    """
    print("\nGenerating visualizations...")

    os.makedirs(output_dir, exist_ok=True)

    # 1. Basic error report visualizations
    plot_error_report_basics(merged_df, error_report_path, output_dir)

    # 2. Error rates by video features (KEY PLOTS!)
    plot_error_rates_by_features(merged_df, output_dir)

    # 3. Scores by video features
    plot_score_by_features(merged_df, output_dir)

    print(f"\nAll visualizations saved to: {output_dir}")


def create_report(
    video_stats_path,
    error_report_path,
    merged_df,
    error_rates_analysis,
    output_path
):
    """
    Create JSON report with error rate analysis by video features

    Args:
        video_stats_path: Path to video statistics
        error_report_path: Path to error report
        merged_df: Merged DataFrame (question level)
        error_rates_analysis: Results from analyze_error_rates_by_video_features
        output_path: Output JSON path
    """
    print("\nGenerating JSON report...")

    # Compute summary statistics at question level
    total_questions = len(merged_df)
    questions_with_errors = (merged_df['score'] < 1).sum()
    questions_perfect = (merged_df['score'] == 1).sum()

    # Unique videos
    unique_videos = merged_df['video_id'].nunique()

    # Error questions vs perfect questions
    error_questions = merged_df[merged_df['score'] < 1]
    perfect_questions = merged_df[merged_df['score'] == 1]

    # Summary stats
    error_summary = {
        'total_questions': int(total_questions),
        'questions_with_errors': int(questions_with_errors),
        'questions_perfect': int(questions_perfect),
        'unique_videos': int(unique_videos),
        'score_distribution': {
            'score_0': int((merged_df['score'] == 0).sum()),
            'score_0.5': int((merged_df['score'] == 0.5).sum()),
            'score_1.0': int((merged_df['score'] == 1.0).sum())
        },
        'error_type_counts': {
            'hallucinations': int(merged_df['has_hallucination'].sum()),
            'missing_information': int(merged_df['has_missing_info'].sum()),
            'partial_match': int(merged_df['is_partial_match'].sum())
        }
    }

    # Feature comparison: error vs perfect
    feature_comparison = {}
    video_features = ['duration_seconds', 'frame_variance', 'motion_intensity',
                     'brightness_mean', 'scene_complexity', 'color_diversity']

    for feature in video_features:
        if feature in merged_df.columns:
            feature_comparison[feature] = {
                'mean_error_questions': float(error_questions[feature].mean()) if len(error_questions) > 0 else 0,
                'mean_perfect_questions': float(perfect_questions[feature].mean()) if len(perfect_questions) > 0 else 0,
                'std_error_questions': float(error_questions[feature].std()) if len(error_questions) > 0 else 0,
                'std_perfect_questions': float(perfect_questions[feature].std()) if len(perfect_questions) > 0 else 0
            }

    # Generate key insights
    key_insights = []

    # Duration insights
    if 'duration_seconds' in error_rates_analysis:
        duration_data = error_rates_analysis['duration_seconds']
        for bin_label, stats in duration_data.items():
            hallucination_rate = stats['error_rates'].get('hallucination', 0)
            if hallucination_rate > 50:
                key_insights.append(
                    f"Videos {bin_label} have high hallucination rate: {hallucination_rate:.1f}%"
                )

    # Add feature comparison insights
    for feature, comparison in feature_comparison.items():
        diff_pct = abs(comparison['mean_error_questions'] - comparison['mean_perfect_questions'])
        if diff_pct > 0:
            direction = "higher" if comparison['mean_error_questions'] > comparison['mean_perfect_questions'] else "lower"
            key_insights.append(
                f"Error questions have {direction} {feature.replace('_', ' ')}: "
                f"{comparison['mean_error_questions']:.2f} vs {comparison['mean_perfect_questions']:.2f}"
            )

    report = {
        'metadata': {
            'analysis_date': datetime.now().isoformat(),
            'video_stats_file': video_stats_path,
            'error_report_file': error_report_path,
            'analysis_type': 'error_rates_by_video_features'
        },
        'summary': error_summary,
        'error_rates_by_video_features': error_rates_analysis,
        'feature_comparison': feature_comparison,
        'key_insights': key_insights
    }

    # Convert all numpy types to native Python types for JSON serialization
    report = convert_to_native_types(report)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"  Report saved to: {output_path}")

    # Print summary
    print(f"\n{'='*70}")
    print("ERROR ANALYSIS SUMMARY")
    print(f"{'='*70}")
    print(f"Total questions analyzed: {error_summary['total_questions']}")
    print(f"Questions with errors: {error_summary['questions_with_errors']} ({error_summary['questions_with_errors']/error_summary['total_questions']*100:.1f}%)")
    print(f"Questions with perfect score: {error_summary['questions_perfect']} ({error_summary['questions_perfect']/error_summary['total_questions']*100:.1f}%)")
    print(f"Unique videos: {error_summary['unique_videos']}")
    print(f"\nScore Distribution:")
    for score_type, count in error_summary['score_distribution'].items():
        print(f"  {score_type}: {count}")
    print(f"\nError Type Counts:")
    for error_type, count in error_summary['error_type_counts'].items():
        print(f"  {error_type}: {count}")
    print(f"\nKey Insights:")
    for i, insight in enumerate(key_insights[:10], 1):
        print(f"  {i}. {insight}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze how video characteristics relate to model errors"
    )
    parser.add_argument(
        '--video-stats',
        type=str,
        default='./video_statistics/video_stats.json',
        help='Path to video statistics JSON'
    )
    parser.add_argument(
        '--error-report',
        type=str,
        default='./error_reports/qwen3vl-8b_report.json',
        help='Path to error report JSON'
    )
    parser.add_argument(
        '--groundtruth-file',
        type=str,
        default='/ivi/zfs/s0/original_homes/gmago/adsqa/AdsQA/AdsQA/testset_groundtruth.json',
        help='Path to groundtruth JSON (to include perfect score questions)'
    )
    parser.add_argument(
        '--output-json',
        type=str,
        default='./correlation_reports/qwen3vl-8b_analysis.json',
        help='Output path for analysis report JSON'
    )
    parser.add_argument(
        '--output-plots-dir',
        type=str,
        default='./correlation_reports/qwen3vl-8b_plots/',
        help='Output directory for visualization plots'
    )

    args = parser.parse_args()

    # Load and merge data
    _, _, merged_df = merge_data(
        args.video_stats,
        args.error_report,
        args.groundtruth_file if os.path.exists(args.groundtruth_file) else None
    )

    print(f"\nAnalyzing {len(merged_df)} questions from {merged_df['video_id'].nunique()} unique videos")

    # Analyze error rates by video features
    error_rates_analysis = analyze_error_rates_by_video_features(merged_df)

    # Generate visualizations
    generate_visualizations(
        merged_df,
        args.output_plots_dir,
        args.error_report
    )

    # Create JSON report
    create_report(
        args.video_stats,
        args.error_report,
        merged_df,
        error_rates_analysis,
        args.output_json
    )

    print("\nError analysis complete!")


if __name__ == '__main__':
    main()
