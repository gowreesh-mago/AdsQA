"""
Clear Previous Evaluation Results

This script removes evaluation fields from prediction files while preserving
the original model predictions. Use this before re-running evaluation with
the new LLM-based verification system.

Usage:
    python clear_evaluations.py --eval_name qwen3vl-8b.json --results_dir ../results/

Options:
    --eval_name: Name of the prediction file (e.g., qwen3vl-8b.json)
    --results_dir: Directory containing prediction files (default: ./results/)
    --dry_run: Preview what would be cleared without actually doing it
"""

import argparse
import json
import os
from pathlib import Path
from tqdm import tqdm


def clear_evaluation_fields(pred_file_path, dry_run=False):
    """
    Remove evaluation fields from a prediction file.

    Args:
        pred_file_path: Path to prediction JSON file
        dry_run: If True, don't actually modify files

    Returns:
        True if fields were cleared, False if no changes needed
    """
    try:
        with open(pred_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not data or not isinstance(data, list) or len(data) == 0:
            return False

        # Check which evaluation fields exist
        eval_fields = ['score', 'missing_info', 'hallucinations', 'evaluation_reasoning',
                      'hallucination_verification', 'missing_info_verification']

        fields_to_remove = [field for field in eval_fields if field in data[0]]

        if not fields_to_remove:
            return False  # No evaluation fields present

        if dry_run:
            return True  # Would be cleared

        # Remove evaluation fields
        for field in fields_to_remove:
            if field in data[0]:
                del data[0][field]

        # Save back to file
        with open(pred_file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        return True

    except Exception as e:
        print(f"\nError processing {pred_file_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Clear evaluation fields from prediction files"
    )
    parser.add_argument(
        '--eval_name',
        type=str,
        default='qwen3vl-8b.json',
        help='Prediction file name (e.g., qwen3vl-8b.json)'
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        default='./results/',
        help='Directory containing prediction files'
    )
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Preview what would be cleared without actually doing it'
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return

    print("=" * 70)
    print("CLEAR EVALUATION RESULTS")
    print("=" * 70)
    print(f"Results directory: {results_dir}")
    print(f"Prediction file name: {args.eval_name}")
    print(f"Mode: {'DRY RUN (preview only)' if args.dry_run else 'LIVE (will modify files)'}")
    print()

    if args.dry_run:
        print("⚠️  DRY RUN MODE: Files will NOT be modified")
        print()

    # Find all prediction files
    pred_files = []
    for question_dir in results_dir.iterdir():
        if question_dir.is_dir():
            pred_file = question_dir / args.eval_name
            if pred_file.exists():
                pred_files.append(pred_file)

    if not pred_files:
        print(f"No prediction files found matching: {args.eval_name}")
        return

    print(f"Found {len(pred_files)} prediction files")
    print()

    # Process files
    cleared_count = 0
    skipped_count = 0

    for pred_file in tqdm(pred_files, desc="Processing"):
        was_cleared = clear_evaluation_fields(pred_file, dry_run=args.dry_run)
        if was_cleared:
            cleared_count += 1
        else:
            skipped_count += 1

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total files scanned: {len(pred_files)}")
    print(f"Files {'that would be' if args.dry_run else ''} cleared: {cleared_count}")
    print(f"Files skipped (no evaluation fields): {skipped_count}")

    if args.dry_run:
        print()
        print("This was a DRY RUN. No files were modified.")
        print("Run without --dry_run to actually clear the evaluation fields.")
    else:
        print()
        print("✓ Evaluation fields cleared successfully!")
        print("You can now re-run evaluation with the new LLM-based verification:")
        print()
        print("  python model_evaluation_qwen.py \\")
        print(f"      --eval_name {args.eval_name} \\")
        print("      --test_file ../testset_groundtruth.json \\")
        print(f"      --results_dir {args.results_dir} \\")
        print("      --model_path Qwen/Qwen2.5-7B-Instruct")

    print("=" * 70)


if __name__ == '__main__':
    main()
