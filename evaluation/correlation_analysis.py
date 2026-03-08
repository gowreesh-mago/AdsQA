"""
Correlation Analysis for Video Statistics vs. Model Errors
Analyzes correlations between video characteristics and model error patterns,
generating both JSON reports and visualizations.
"""

import argparse
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def read_json(jpath):
    """Load JSON file"""
    with open(jpath, 'r') as ff:
        data = json.load(ff)
    return data


def merge_data(video_stats_path, error_report_path):
    """
    Merge video statistics with error report data

    Args:
        video_stats_path: Path to video statistics JSON
        error_report_path: Path to error report JSON

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
        has_hallucination = any(e['category'] == 'Hallucination' for e in errors)
        has_missing_info = any(e['category'] == 'Missing Information' for e in errors)
        has_format_error = any(e['category'] == 'Format Error' for e in errors)
        is_complete_mismatch = any(e['category'] == 'Complete Mismatch' for e in errors)
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
            'has_format_error': has_format_error,
            'is_complete_mismatch': is_complete_mismatch,
            'is_partial_match': is_partial_match,
            'question_type': primary_question_type
        })

    # Also add perfect scores from the main report
    total_questions = error_report['total_questions']
    evaluated_questions = error_report['evaluated_questions']
    perfect_scores = error_report['perfect_scores']

    question_df = pd.DataFrame(question_records)

    print(f"  Loaded {len(video_df)} video statistics")
    print(f"  Loaded {len(question_df)} question error records")

    # Merge on video_id
    merged_df = pd.merge(question_df, video_df, on='video_id', how='left')

    print(f"  Merged dataset: {len(merged_df)} questions with video stats")
    print(f"  Questions without video stats: {merged_df['duration_seconds'].isna().sum()}")

    # Drop rows without video stats
    merged_df = merged_df.dropna(subset=['duration_seconds'])

    return question_df, video_df, merged_df


def aggregate_to_video_level(merged_df):
    """
    Aggregate question-level data to video level

    Args:
        merged_df: Merged DataFrame with questions and video stats

    Returns:
        DataFrame: Video-level aggregated data
    """
    # Video features (take first value since they're the same for all questions from same video)
    video_feature_cols = [
        'duration_seconds', 'total_frames', 'fps', 'resolution_width',
        'resolution_height', 'resolution_area', 'file_size_mb',
        'frame_variance', 'brightness_mean', 'brightness_std',
        'brightness_min', 'brightness_max', 'motion_intensity',
        'color_diversity', 'scene_complexity'
    ]

    # Aggregate error metrics
    video_agg = merged_df.groupby('video_id').agg({
        'score': ['mean', 'std', 'count'],
        'has_hallucination': 'mean',  # Hallucination rate
        'has_missing_info': 'mean',   # Missing info rate
        'has_format_error': 'mean',   # Format error rate
        'is_complete_mismatch': 'mean',  # Complete mismatch rate
        'is_partial_match': 'mean',   # Partial match rate
        **{col: 'first' for col in video_feature_cols}  # Take first value
    })

    # Flatten column names
    video_agg.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col
                         for col in video_agg.columns]

    video_agg = video_agg.reset_index()

    return video_agg


def compute_correlations(video_agg_df, min_p_value=0.05):
    """
    Compute correlations between video features and error metrics

    Args:
        video_agg_df: Video-level aggregated DataFrame
        min_p_value: Significance threshold

    Returns:
        dict: Correlation results
    """
    print("\nComputing correlations...")

    # Define feature and target columns
    video_features = [
        'duration_seconds', 'total_frames', 'fps', 'resolution_area',
        'file_size_mb', 'frame_variance', 'brightness_mean',
        'brightness_std', 'motion_intensity', 'color_diversity',
        'scene_complexity'
    ]

    error_metrics = [
        'score_mean', 'has_hallucination', 'has_missing_info',
        'has_format_error', 'is_complete_mismatch', 'is_partial_match'
    ]

    correlation_results = {}

    for feature in video_features:
        for metric in error_metrics:
            # Skip if either column has all NaN
            if video_agg_df[feature].isna().all() or video_agg_df[metric].isna().all():
                continue

            # Drop NaN values for this pair
            valid_data = video_agg_df[[feature, metric]].dropna()

            if len(valid_data) < 3:  # Need at least 3 points
                continue

            x = valid_data[feature].values
            y = valid_data[metric].values

            # Pearson correlation (linear)
            pearson_r, pearson_p = stats.pearsonr(x, y)

            # Spearman correlation (monotonic, robust)
            spearman_r, spearman_p = stats.spearmanr(x, y)

            correlation_results[f"{feature}_vs_{metric}"] = {
                'feature': feature,
                'target': metric,
                'pearson_r': float(pearson_r),
                'pearson_p': float(pearson_p),
                'spearman_r': float(spearman_r),
                'spearman_p': float(spearman_p),
                'significant': pearson_p < min_p_value,
                'sample_size': len(valid_data)
            }

    print(f"  Computed {len(correlation_results)} correlation pairs")

    # Identify significant correlations
    significant = {k: v for k, v in correlation_results.items() if v['significant']}
    print(f"  Found {len(significant)} significant correlations (p < {min_p_value})")

    return correlation_results


def compute_per_question_type_correlations(merged_df, min_p_value=0.05):
    """
    Compute correlations per question type

    Args:
        merged_df: Merged DataFrame
        min_p_value: Significance threshold

    Returns:
        dict: Per-question-type correlation results
    """
    print("\nComputing per-question-type correlations...")

    question_types = merged_df['question_type'].unique()
    per_type_results = {}

    for qtype in question_types:
        if qtype == 'Unknown':
            continue

        type_df = merged_df[merged_df['question_type'] == qtype]

        if len(type_df) < 10:  # Skip if too few samples
            continue

        # Aggregate to video level for this type
        type_agg = aggregate_to_video_level(type_df)

        # Compute correlations
        type_corr = compute_correlations(type_agg, min_p_value)

        per_type_results[qtype] = type_corr

        significant_count = sum(1 for v in type_corr.values() if v['significant'])
        print(f"  {qtype}: {significant_count} significant correlations ({len(type_df)} questions)")

    return per_type_results


def identify_top_correlations(correlation_results, top_n=10):
    """
    Identify top N correlations by absolute correlation value

    Args:
        correlation_results: Dictionary of correlation results
        top_n: Number of top correlations to return

    Returns:
        list: Top N correlations sorted by |r|
    """
    # Sort by absolute Pearson correlation
    sorted_corr = sorted(
        correlation_results.items(),
        key=lambda x: abs(x[1]['pearson_r']),
        reverse=True
    )

    top_correlations = []
    for key, corr in sorted_corr[:top_n]:
        # Generate interpretation
        direction = "positively" if corr['pearson_r'] > 0 else "negatively"
        strength = "strongly" if abs(corr['pearson_r']) > 0.5 else "moderately" if abs(corr['pearson_r']) > 0.3 else "weakly"

        interpretation = f"{corr['feature']} correlates {strength} {direction} with {corr['target']} (r={corr['pearson_r']:.3f}, p={corr['pearson_p']:.4f})"

        top_correlations.append({
            'feature': corr['feature'],
            'target': corr['target'],
            'pearson_r': corr['pearson_r'],
            'spearman_r': corr['spearman_r'],
            'p_value': corr['pearson_p'],
            'significant': corr['significant'],
            'interpretation': interpretation
        })

    return top_correlations


def plot_correlation_heatmap(correlation_results, output_path):
    """
    Generate correlation heatmap

    Args:
        correlation_results: Dictionary of correlation results
        output_path: Output file path
    """
    # Extract unique features and metrics
    features = sorted(set(v['feature'] for v in correlation_results.values()))
    metrics = sorted(set(v['target'] for v in correlation_results.values()))

    # Create correlation matrix
    corr_matrix = np.zeros((len(features), len(metrics)))
    p_matrix = np.ones((len(features), len(metrics)))

    for i, feature in enumerate(features):
        for j, metric in enumerate(metrics):
            key = f"{feature}_vs_{metric}"
            if key in correlation_results:
                corr_matrix[i, j] = correlation_results[key]['pearson_r']
                p_matrix[i, j] = correlation_results[key]['pearson_p']

    # Create figure
    plt.figure(figsize=(12, 10))

    # Plot heatmap
    sns.heatmap(
        corr_matrix,
        xticklabels=metrics,
        yticklabels=features,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        cbar_kws={'label': 'Pearson Correlation'}
    )

    plt.title('Correlation Heatmap: Video Features vs. Error Metrics', fontsize=14, pad=20)
    plt.xlabel('Error Metrics', fontsize=12)
    plt.ylabel('Video Features', fontsize=12)
    plt.tight_layout()

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved correlation heatmap to: {output_path}")


def plot_top_correlations(merged_df, top_correlations, output_path, top_n=10):
    """
    Create scatter plot grid for top N correlations

    Args:
        merged_df: Merged DataFrame
        top_correlations: List of top correlations
        output_path: Output file path
        top_n: Number of correlations to plot
    """
    # Aggregate to video level
    video_agg = aggregate_to_video_level(merged_df)

    # Determine grid size
    n_plots = min(len(top_correlations), top_n)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = axes.flatten() if n_plots > 1 else [axes]

    for idx, corr in enumerate(top_correlations[:top_n]):
        ax = axes[idx]

        feature = corr['feature']
        target = corr['target']

        # Get data
        valid_data = video_agg[[feature, target]].dropna()
        x = valid_data[feature].values
        y = valid_data[target].values

        # Scatter plot
        ax.scatter(x, y, alpha=0.5, s=30)

        # Add regression line
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)

        # Labels
        ax.set_xlabel(feature.replace('_', ' ').title(), fontsize=10)
        ax.set_ylabel(target.replace('_', ' ').title(), fontsize=10)
        ax.set_title(f"r={corr['pearson_r']:.3f}, p={corr['p_value']:.4f}", fontsize=11)
        ax.grid(True, alpha=0.3)

    # Remove extra subplots
    for idx in range(n_plots, len(axes)):
        fig.delaxes(axes[idx])

    plt.suptitle(f'Top {n_plots} Correlations', fontsize=16, y=1.00)
    plt.tight_layout()

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved top correlations plot to: {output_path}")


def plot_score_distributions(merged_df, output_dir):
    """
    Plot score distributions across feature bins

    Args:
        merged_df: Merged DataFrame
        output_dir: Output directory for plots
    """
    features_to_plot = ['duration_seconds', 'frame_variance', 'motion_intensity', 'scene_complexity']

    for feature in features_to_plot:
        # Create quartile bins
        merged_df[f'{feature}_bin'] = pd.qcut(
            merged_df[feature],
            q=4,
            labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'],
            duplicates='drop'
        )

        # Box plot
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=merged_df, x=f'{feature}_bin', y='score', palette='Set2')
        plt.title(f'Score Distribution by {feature.replace("_", " ").title()}', fontsize=14)
        plt.xlabel(f'{feature.replace("_", " ").title()} Quartile', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()

        # Save
        output_path = os.path.join(output_dir, f'score_distribution_by_{feature}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  Saved {feature} distribution plot to: {output_path}")


def plot_feature_distributions(merged_df, output_path):
    """
    Compare feature distributions for error vs. non-error cases

    Args:
        merged_df: Merged DataFrame
        output_path: Output file path
    """
    features = ['duration_seconds', 'frame_variance', 'motion_intensity', 'brightness_mean']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, feature in enumerate(features):
        ax = axes[idx]

        # Split by score
        error_data = merged_df[merged_df['score'] < 1][feature]
        perfect_data = merged_df[merged_df['score'] == 1][feature]

        # Histogram
        ax.hist(error_data, bins=30, alpha=0.6, label='Error (score < 1)', color='red', density=True)
        ax.hist(perfect_data, bins=30, alpha=0.6, label='Perfect (score = 1)', color='green', density=True)

        ax.set_xlabel(feature.replace('_', ' ').title(), fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'{feature.replace("_", " ").title()} Distribution', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('Feature Distributions: Error vs. Perfect Score', fontsize=14)
    plt.tight_layout()

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved feature distributions plot to: {output_path}")


def generate_visualizations(merged_df, correlation_results, top_correlations, output_dir, top_n=10):
    """
    Generate all visualizations

    Args:
        merged_df: Merged DataFrame
        correlation_results: Dictionary of correlation results
        top_correlations: List of top correlations
        output_dir: Output directory
        top_n: Number of top correlations to visualize
    """
    print("\nGenerating visualizations...")

    os.makedirs(output_dir, exist_ok=True)

    # 1. Correlation heatmap
    plot_correlation_heatmap(
        correlation_results,
        os.path.join(output_dir, 'correlation_heatmap.png')
    )

    # 2. Top correlations scatter plots
    plot_top_correlations(
        merged_df,
        top_correlations,
        os.path.join(output_dir, f'top_{top_n}_correlations.png'),
        top_n=top_n
    )

    # 3. Score distribution plots
    plot_score_distributions(merged_df, output_dir)

    # 4. Feature distribution comparisons
    plot_feature_distributions(
        merged_df,
        os.path.join(output_dir, 'feature_distributions_error_comparison.png')
    )

    print(f"\nAll visualizations saved to: {output_dir}")


def create_report(
    video_stats_path,
    error_report_path,
    merged_df,
    video_agg_df,
    correlation_results,
    per_type_correlations,
    top_correlations,
    output_path
):
    """
    Create JSON correlation report

    Args:
        video_stats_path: Path to video statistics
        error_report_path: Path to error report
        merged_df: Merged DataFrame
        video_agg_df: Video-level aggregated DataFrame
        correlation_results: Overall correlation results
        per_type_correlations: Per-question-type correlations
        top_correlations: Top correlations list
        output_path: Output JSON path
    """
    print("\nGenerating JSON report...")

    # Compute summary statistics
    videos_with_errors = (video_agg_df['score_mean'] < 1).sum()
    videos_perfect_score = (video_agg_df['score_mean'] == 1).sum()

    # Error rate by video features
    error_videos = video_agg_df[video_agg_df['score_mean'] < 1]
    perfect_videos = video_agg_df[video_agg_df['score_mean'] == 1]

    report = {
        'metadata': {
            'analysis_date': datetime.now().isoformat(),
            'video_stats_file': video_stats_path,
            'error_report_file': error_report_path,
            'total_videos': len(video_agg_df),
            'total_questions': len(merged_df),
            'questions_with_video_stats': len(merged_df)
        },
        'overall_correlations': {
            'pearson': {k: {'r': v['pearson_r'], 'p': v['pearson_p'], 'significant': v['significant']}
                        for k, v in correlation_results.items()},
            'spearman': {k: {'r': v['spearman_r'], 'p': v['spearman_p']}
                         for k, v in correlation_results.items()}
        },
        'per_question_type_correlations': per_type_correlations,
        'top_significant_correlations': top_correlations,
        'aggregated_statistics': {
            'videos_with_errors': int(videos_with_errors),
            'videos_perfect_score': int(videos_perfect_score),
            'mean_video_duration_errors': float(error_videos['duration_seconds'].mean()) if len(error_videos) > 0 else 0,
            'mean_video_duration_perfect': float(perfect_videos['duration_seconds'].mean()) if len(perfect_videos) > 0 else 0,
            'mean_frame_variance_errors': float(error_videos['frame_variance'].mean()) if len(error_videos) > 0 else 0,
            'mean_frame_variance_perfect': float(perfect_videos['frame_variance'].mean()) if len(perfect_videos) > 0 else 0
        }
    }

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"  Report saved to: {output_path}")

    # Print summary
    print(f"\n{'='*70}")
    print("CORRELATION ANALYSIS SUMMARY")
    print(f"{'='*70}")
    print(f"Total videos analyzed: {report['metadata']['total_videos']}")
    print(f"Total questions: {report['metadata']['total_questions']}")
    print(f"Videos with errors: {report['aggregated_statistics']['videos_with_errors']}")
    print(f"Videos with perfect score: {report['aggregated_statistics']['videos_perfect_score']}")
    print(f"\nTop {len(top_correlations)} Significant Correlations:")
    for i, corr in enumerate(top_correlations[:5], 1):
        print(f"  {i}. {corr['interpretation']}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze correlations between video statistics and model errors"
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
        '--output-json',
        type=str,
        default='./correlation_reports/qwen3vl-8b_correlations.json',
        help='Output path for correlation report JSON'
    )
    parser.add_argument(
        '--output-plots-dir',
        type=str,
        default='./correlation_reports/qwen3vl-8b_plots/',
        help='Output directory for visualization plots'
    )
    parser.add_argument(
        '--min-p-value',
        type=float,
        default=0.05,
        help='Significance threshold for correlations'
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=10,
        help='Number of top correlations to visualize'
    )

    args = parser.parse_args()

    # Load and merge data
    question_df, video_df, merged_df = merge_data(
        args.video_stats,
        args.error_report
    )

    # Aggregate to video level
    video_agg_df = aggregate_to_video_level(merged_df)
    print(f"\nVideo-level aggregation: {len(video_agg_df)} unique videos")

    # Compute overall correlations
    correlation_results = compute_correlations(video_agg_df, args.min_p_value)

    # Compute per-question-type correlations
    per_type_correlations = compute_per_question_type_correlations(
        merged_df,
        args.min_p_value
    )

    # Identify top correlations
    top_correlations = identify_top_correlations(correlation_results, args.top_n)

    # Generate visualizations
    generate_visualizations(
        merged_df,
        correlation_results,
        top_correlations,
        args.output_plots_dir,
        args.top_n
    )

    # Create JSON report
    create_report(
        args.video_stats,
        args.error_report,
        merged_df,
        video_agg_df,
        correlation_results,
        per_type_correlations,
        top_correlations,
        args.output_json
    )

    print("\nCorrelation analysis complete!")


if __name__ == '__main__':
    main()
