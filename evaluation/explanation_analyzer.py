"""
Explanation Analysis Module for AdsQA Evaluation
Compares model reasoning with ground truth and identifies reasoning gaps
"""

import re
import torch


class ExplanationAnalyzer:
    """Analyze and compare explanations between model and ground truth"""

    def __init__(self, model, tokenizer):
        """
        Initialize ExplanationAnalyzer

        Args:
            model: Qwen2.5-7B-Instruct model for analysis
            tokenizer: Corresponding tokenizer
        """
        self.model = model
        self.tokenizer = tokenizer

    def compare_explanations(self, prediction, ground_truth_item):
        """
        Extract and compare reasoning from prediction with ground truth

        Args:
            prediction: Model's prediction string with <think> and <answer> tags
            ground_truth_item: Ground truth item dict with answer and optional explanation

        Returns:
            Dictionary with:
            - model_reasoning: Extracted <think> content
            - ground_truth_reasoning: Ground truth explanation (if available)
            - difference_explanation: Analysis of differences
            - potential_causes: List of potential causes for errors
        """
        pred_think = self._extract_think(prediction)
        gt_explanation = ground_truth_item.get("explanation", "")
        pred_answer = self._extract_answer(prediction)
        gt_answer = ground_truth_item.get("answer", ground_truth_item.get("gt_answer", ""))

        if not pred_think:
            return {
                "model_reasoning": "",
                "ground_truth_reasoning": gt_explanation,
                "difference_explanation": "No reasoning provided in prediction",
                "potential_causes": ["Format Error"]
            }

        # Use Qwen2.5-7B to analyze differences
        prompt = f"""Compare these two explanations for an advertisement question:

Ground Truth Answer:
{gt_answer}

Model's Answer:
{pred_answer}

Model's Reasoning:
{pred_think}

Identify what is missing or wrong in the model's reasoning. Consider:
1. Missing visual elements or frames
2. Incorrect interpretation of scenes
3. ASR/audio misunderstanding
4. Wrong connections between visual and conceptual elements

Provide a concise analysis (2-3 sentences) explaining the key differences."""

        try:
            difference_analysis = self._query_model(prompt)
        except Exception as e:
            print(f"Warning: Explanation analysis failed ({e}). Using fallback.")
            difference_analysis = "Unable to analyze differences due to model error."

        return {
            "model_reasoning": pred_think,
            "ground_truth_reasoning": gt_explanation,
            "difference_explanation": difference_analysis,
            "potential_causes": self._identify_causes(difference_analysis)
        }

    def _extract_think(self, prediction):
        """
        Extract content from <think> tags

        Args:
            prediction: Model's prediction string

        Returns:
            Content within <think> tags, or empty string if not found
        """
        match = re.search(r'<think>(.*?)</think>', prediction, re.DOTALL)
        return match.group(1).strip() if match else ""

    def _extract_answer(self, prediction):
        """
        Extract content from <answer> tags

        Args:
            prediction: Model's prediction string

        Returns:
            Content within <answer> tags, or empty string if not found
        """
        match = re.search(r'<answer>(.*?)</answer>', prediction, re.DOTALL)
        return match.group(1).strip() if match else ""

    def _identify_causes(self, analysis):
        """
        Extract potential causes from the analysis based on keywords

        Args:
            analysis: Difference analysis text

        Returns:
            List of potential causes
        """
        causes = []
        keywords = {
            "frame": "Missing frames",
            "visual": "Visual misunderstanding",
            "audio": "ASR issues",
            "transcript": "ASR issues",
            "connection": "Wrong interpretation",
            "concept": "Conceptual error",
            "missing": "Missing information"
        }

        analysis_lower = analysis.lower()
        for keyword, cause in keywords.items():
            if keyword in analysis_lower:
                if cause not in causes:
                    causes.append(cause)

        return causes if causes else ["Reasoning gap"]

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
                max_new_tokens=512,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        generated = outputs[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()
