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

        # 2. Identify missing information (for partial matches and mismatches)
        if score < 1.0:
            missing_info = self._identify_missing_information(prediction, ground_truth, meta_info)
            if missing_info:
                errors.append({
                    "category": "Missing Information",
                    "detail": missing_info
                })

        # 3. Hallucination detection (use model to check for contradictions)
        hallucination_detail = self._check_hallucination(prediction, meta_info, ground_truth)
        if hallucination_detail:
            errors.append({
                "category": "Hallucination",
                "detail": hallucination_detail
            })

        # 4. Score-based categorization (for high-level summary)
        if score == 0:
            errors.append({
                "category": "Complete Mismatch",
                "detail": "Prediction has no overlap with ground truth"
            })
        elif score == 0.5:
            errors.append({
                "category": "Partial Match",
                "detail": "Contains some correct information but incomplete"
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

    def _identify_missing_information(self, prediction, ground_truth, meta_info):
        """
        Use Qwen2.5-7B to identify what specific information is missing from prediction

        Args:
            prediction: Model's prediction string
            ground_truth: Ground truth answer
            meta_info: Meta-information about the ad

        Returns:
            String describing missing information, or empty string if none
        """
        # Extract answer content from prediction
        import re
        answer_match = re.search(r'<answer>(.*?)</answer>', prediction, re.DOTALL)
        pred_answer = answer_match.group(1).strip() if answer_match else prediction

        prompt = f"""Compare the prediction with the ground truth answer and meta-information to identify what key information is MISSING from the prediction.

Ground Truth Answer:
{ground_truth}

Meta-Information:
{meta_info}

Prediction:
{pred_answer}

Instructions:
1. List the key information from ground truth that is MISSING or INCOMPLETE in the prediction
2. Be specific - identify exact concepts, entities, or details that are absent
3. Ignore stylistic differences - focus only on substantive missing content
4. If nothing is missing, respond "None"

Missing Information:"""

        try:
            response = self._query_model(prompt, max_tokens=512)
            response = response.strip()
            if response.lower() in ["none", "none.", "nothing", "nothing."]:
                return ""
            return response
        except Exception as e:
            print(f"Warning: Missing information analysis failed ({e}). Skipping.")
            return ""

    def _check_hallucination(self, prediction, meta_info, ground_truth):
        """
        Use Qwen2.5-7B to detect if prediction contains incorrect/contradictory information

        Args:
            prediction: Model's prediction string
            meta_info: Meta-information about the ad
            ground_truth: Ground truth answer

        Returns:
            String describing hallucination, or empty string if none detected
        """
        # Extract answer content from prediction
        import re
        answer_match = re.search(r'<answer>(.*?)</answer>', prediction, re.DOTALL)
        pred_answer = answer_match.group(1).strip() if answer_match else prediction

        prompt = f"""Identify if the prediction contains any INCORRECT or CONTRADICTORY information compared to the meta-information and ground truth.

Meta-Information:
{meta_info}

Ground Truth Answer:
{ground_truth}

Prediction:
{pred_answer}

Instructions:
1. Identify specific claims in the prediction that contradict the meta-information or ground truth
2. Focus on factual errors, not missing information
3. Distinguish between wrong information (hallucination) vs incomplete information (not hallucination)
4. If no contradictions found, respond "None"

Contradictions/Hallucinations:"""

        try:
            response = self._query_model(prompt, max_tokens=512)
            response = response.strip()
            if response.lower() in ["none", "none.", "nothing", "nothing."]:
                return ""
            return response
        except Exception as e:
            print(f"Warning: Hallucination check failed ({e}). Skipping.")
            return ""

    def _query_model(self, prompt, max_tokens=256):
        """
        Helper to query Qwen2.5-7B model

        Args:
            prompt: Prompt string
            max_tokens: Maximum tokens to generate

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
                max_new_tokens=max_tokens,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        generated = outputs[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()
