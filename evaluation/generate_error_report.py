"""
Generate Comprehensive Error Report for AdsQA Evaluation
Combines error categorization and explanation analysis
"""

import argparse
import json
import os
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from error_analysis import ErrorAnalyzer
from explanation_analyzer import ExplanationAnalyzer


def read_json(jpath):
    """Load JSON file"""
    with open(jpath, 'r') as ff:
        data = json.load(ff)
    return data


def generate_comprehensive_report(
    results_dir,
    eval_name,
    groundtruth_file,
    output_path,
    model,
    tokenizer,
    max_samples=None
):
    """
    Generate comprehensive error report with categorization and explanation analysis

    Args:
        results_dir: Directory containing prediction files (./results/)
        eval_name: Model prediction file name (e.g., qwen3vl-8b.json)
        groundtruth_file: Path to ground truth JSON file
        output_path: Where to save the error report
        model: Loaded Qwen2.5-7B-Instruct model
        tokenizer: Loaded tokenizer
        max_samples: Limit to first N samples for testing (None = all)

    Returns:
        Dictionary containing the complete error report
    """
    # Initialize analyzers
    print("Initializing error and explanation analyzers...")
    error_analyzer = ErrorAnalyzer(model, tokenizer)
    explanation_analyzer = ExplanationAnalyzer(model, tokenizer)

    # Load ground truth
    print(f"Loading ground truth from: {groundtruth_file}")
    with open(groundtruth_file) as f:
        groundtruth = json.load(f)

    # Limit samples for testing
    if max_samples:
        groundtruth = groundtruth[:max_samples]
        print(f"Testing mode: Analyzing only {max_samples} samples")

    # Initialize report structure
    report = {
        "model": eval_name,
        "total_questions": len(groundtruth),
        "evaluated_questions": 0,
        "perfect_scores": 0,
        "error_summary": {
            "Format Error": 0,
            "Complete Mismatch": 0,
            "Partial Match": 0,
            "Hallucination": 0,
            "Missing Information": 0
        },
        "score_distribution": {
            "score_0": 0,
            "score_0.5": 0,
            "score_1": 0
        },
        "question_type_analysis": {},
        "detailed_errors": []
    }

    # Analyze each sample
    print("\nAnalyzing errors...")
    for item in tqdm(groundtruth, desc="Processing"):
        question_id = item["question_id"]
        pred_path = os.path.join(results_dir, question_id, eval_name)

        # Skip if prediction doesn't exist
        if not os.path.exists(pred_path):
            continue

        try:
            with open(pred_path) as f:
                pred_data = json.load(f)
        except Exception as e:
            print(f"\nWarning: Failed to load {pred_path}: {e}")
            continue

        # Extract prediction and score
        prediction = pred_data[0].get("prediction", "")
        score_str = pred_data[0].get("score", "")

        if not prediction:
            continue

        # Parse score
        score_match = re.search(r"(\d+(?:\.\d+)?)", score_str)
        score = float(score_match.group(1)) if score_match else None

        report["evaluated_questions"] += 1

        # Track score distribution
        if score == 0:
            report["score_distribution"]["score_0"] += 1
        elif score == 0.5:
            report["score_distribution"]["score_0.5"] += 1
        elif score == 1.0:
            report["score_distribution"]["score_1"] += 1
            report["perfect_scores"] += 1

        # Initialize question type tracking
        question_types = item.get("question_type", [])
        for qtype in question_types:
            if qtype not in report["question_type_analysis"]:
                report["question_type_analysis"][qtype] = {
                    "total": 0,
                    "perfect_scores": 0,
                    "format_errors": 0,
                    "hallucinations": 0,
                    "missing_information": 0,
                    "avg_score": 0,
                    "score_sum": 0
                }
            report["question_type_analysis"][qtype]["total"] += 1
            report["question_type_analysis"][qtype]["score_sum"] += score if score is not None else 0

        # Skip perfect scores (no errors to analyze)
        if score == 1.0:
            for qtype in question_types:
                report["question_type_analysis"][qtype]["perfect_scores"] += 1
            continue

        # Get ground truth info
        ground_truth = item.get("answer", item.get("gt_answer", ""))
        meta_info = item.get("meta_info", "")

        # Categorize errors
        errors = error_analyzer.categorize_error(
            prediction,
            ground_truth,
            meta_info,
            score
        )

        # Analyze explanation differences
        exp_analysis = explanation_analyzer.compare_explanations(prediction, item)

        # Update error summary
        for error in errors:
            category = error["category"]
            if category in report["error_summary"]:
                report["error_summary"][category] += 1

            # Update question type analysis
            for qtype in question_types:
                if category == "Format Error":
                    report["question_type_analysis"][qtype]["format_errors"] += 1
                elif category == "Hallucination":
                    report["question_type_analysis"][qtype]["hallucinations"] += 1
                elif category == "Missing Information":
                    report["question_type_analysis"][qtype]["missing_information"] += 1

        # Add to detailed errors
        report["detailed_errors"].append({
            "question_id": question_id,
            "question": item.get("question", ""),
            "question_type": question_types,
            "score": score,
            "errors": errors,
            "prediction": prediction,
            "ground_truth": ground_truth,
            "meta_info": meta_info,
            "explanation_analysis": exp_analysis
        })

    # Calculate averages for question types
    for qtype, stats in report["question_type_analysis"].items():
        if stats["total"] > 0:
            stats["avg_score"] = stats["score_sum"] / stats["total"]

    # Add overall insights
    total_errors = len(report["detailed_errors"])
    hallucination_count = report["error_summary"]["Hallucination"]
    missing_info_count = report["error_summary"]["Missing Information"]

    report["overall_insights"] = {
        "error_rate": (total_errors / report["evaluated_questions"] * 100) if report["evaluated_questions"] > 0 else 0,
        "perfect_score_rate": (report["perfect_scores"] / report["evaluated_questions"] * 100) if report["evaluated_questions"] > 0 else 0,
        "hallucination_prevalence": (hallucination_count / total_errors * 100) if total_errors > 0 else 0,
        "missing_information_prevalence": (missing_info_count / total_errors * 100) if total_errors > 0 else 0,
        "primary_issue": "Hallucination" if hallucination_count > missing_info_count else "Missing Information" if missing_info_count > hallucination_count else "Both equally prevalent",
        "format_error_rate": (report["error_summary"]["Format Error"] / report["evaluated_questions"] * 100) if report["evaluated_questions"] > 0 else 0
    }

    # Save report
    print(f"\nSaving error report to: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n" + "="*70)
    print("ERROR REPORT SUMMARY")
    print("="*70)
    print(f"Model: {report['model']}")
    print(f"Total Questions: {report['total_questions']}")
    print(f"Evaluated: {report['evaluated_questions']}")
    print(f"Perfect Scores: {report['perfect_scores']} ({report['overall_insights']['perfect_score_rate']:.1f}%)")
    print(f"Errors Found: {total_errors} ({report['overall_insights']['error_rate']:.1f}%)")

    print("\n" + "-"*70)
    print("SCORE DISTRIBUTION")
    print("-"*70)
    for score_type, count in report["score_distribution"].items():
        percentage = (count / report["evaluated_questions"] * 100) if report["evaluated_questions"] > 0 else 0
        print(f"  {score_type}: {count} ({percentage:.1f}%)")

    print("\n" + "-"*70)
    print("ERROR CATEGORIES")
    print("-"*70)
    for category, count in report["error_summary"].items():
        percentage = (count / total_errors * 100) if total_errors > 0 else 0
        print(f"  {category}: {count} ({percentage:.1f}% of errors)")

    print("\n" + "-"*70)
    print("KEY INSIGHTS")
    print("-"*70)
    print(f"  Primary Issue: {report['overall_insights']['primary_issue']}")
    print(f"  Hallucination Rate: {report['overall_insights']['hallucination_prevalence']:.1f}% of errors")
    print(f"  Missing Information Rate: {report['overall_insights']['missing_information_prevalence']:.1f}% of errors")
    print(f"  Format Error Rate: {report['overall_insights']['format_error_rate']:.1f}% of evaluated questions")

    print("\n" + "-"*70)
    print("PER-QUESTION-TYPE ANALYSIS")
    print("-"*70)
    for qtype, stats in sorted(report["question_type_analysis"].items()):
        print(f"\n  {qtype}:")
        print(f"    Total: {stats['total']}")
        print(f"    Avg Score: {stats['avg_score']:.2f}")
        print(f"    Perfect: {stats['perfect_scores']} ({stats['perfect_scores']/stats['total']*100:.1f}%)")
        print(f"    Format Errors: {stats['format_errors']}")
        print(f"    Hallucinations: {stats['hallucinations']}")
        print(f"    Missing Info: {stats['missing_information']}")

    print("\n" + "="*70)
    print(f"Report saved to: {output_path}")
    print("="*70)

    return report


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Generate comprehensive error report")
    parser.add_argument('--results_dir', type=str, default='./results/',
                        help='Directory containing prediction files')
    parser.add_argument('--eval_name', type=str, default='qwen3vl-8b.json',
                        help='Prediction file name (e.g., qwen3vl-8b.json)')
    parser.add_argument('--groundtruth_file', type=str, default='./testset_groundtruth.json',
                        help='Ground truth file path')
    parser.add_argument('--output_path', type=str, default='./error_reports/qwen3vl-8b_report.json',
                        help='Output path for error report')
    parser.add_argument('--model_path', type=str, default='Qwen/Qwen2.5-7B-Instruct',
                        help='Qwen2.5-7B-Instruct model path for analysis')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Analyze only first N samples for testing')

    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model.eval()
    print("Model loaded successfully!")

    # Generate report
    generate_comprehensive_report(
        results_dir=args.results_dir,
        eval_name=args.eval_name,
        groundtruth_file=args.groundtruth_file,
        output_path=args.output_path,
        model=model,
        tokenizer=tokenizer,
        max_samples=args.max_samples
    )
