"""
Error Analysis Module for AdsQA Evaluation
Categorizes errors into fixed categories and detects hallucinations
"""

import torch


class ErrorAnalyzer:
    """Analyze and categorize errors in model predictions"""

    def __init__(self, model, tokenizer):
        """
        Initialize ErrorAnalyzer

        Args:
            model: Qwen2.5-7B-Instruct model for hallucination detection
            tokenizer: Corresponding tokenizer
        """
        self.model = model
        self.tokenizer = tokenizer

    def categorize_error(self, prediction, ground_truth, meta_info, score):
        """
        Categorize errors in prediction

        Args:
            prediction: Model's prediction string
            ground_truth: Ground truth answer
            meta_info: Meta-information about the ad
            score: Evaluation score (0, 0.5, or 1)

        Returns:
            List of error dictionaries with category and detail
        """
        errors = []

        # 1. Format check
        if not self._has_valid_format(prediction):
            errors.append({
                "category": "Format Error",
                "detail": self._get_format_issues(prediction)
            })

        # 2. Score-based categorization
        if score == 0:
            errors.append({
                "category": "Complete Mismatch",
                "detail": "Prediction has no overlap with ground truth"
            })
        elif score == 0.5:
            errors.append({
                "category": "Partial Match",
                "detail": "Missing key information from ground truth"
            })

        # 3. Hallucination detection (use model to check for contradictions)
        if self._check_hallucination(prediction, meta_info):
            errors.append({
                "category": "Hallucination",
                "detail": "Prediction contradicts meta-information"
            })

        return errors

    def _has_valid_format(self, prediction):
        """
        Check if prediction has valid format with <think> and <answer> tags

        Args:
            prediction: Model's prediction string

        Returns:
            True if format is valid, False otherwise
        """
        return ("<think>" in prediction and "</think>" in prediction and
                "<answer>" in prediction and "</answer>" in prediction)

    def _get_format_issues(self, prediction):
        """
        Get specific format issues

        Args:
            prediction: Model's prediction string

        Returns:
            String describing format issues
        """
        issues = []
        if "<think>" not in prediction or "</think>" not in prediction:
            issues.append("Missing <think> tags")
        if "<answer>" not in prediction or "</answer>" not in prediction:
            issues.append("Missing <answer> tags")
        return ", ".join(issues) if issues else "Unknown format issue"

    def _check_hallucination(self, prediction, meta_info):
        """
        Use Qwen2.5-7B to detect if prediction contradicts meta-information

        Args:
            prediction: Model's prediction string
            meta_info: Meta-information about the ad

        Returns:
            True if hallucination detected, False otherwise
        """
        prompt = f"""Given this advertisement meta-information:
{meta_info}

Does this prediction contradict or make claims inconsistent with the meta-info?
Prediction: {prediction}

Answer YES or NO and explain briefly."""

        try:
            response = self._query_model(prompt)
            return "YES" in response.upper()
        except Exception as e:
            print(f"Warning: Hallucination check failed ({e}). Skipping.")
            return False

    def _query_model(self, prompt):
        """
        Helper to query Qwen2.5-7B model

        Args:
            prompt: Prompt string

        Returns:
            Model's response
        """
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        generated = outputs[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()
