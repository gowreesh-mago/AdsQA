"""
Hallucination Verification Module
Verifies that claimed hallucinations are genuinely absent from ground truth
and validates partial match scores.
"""

import re
from difflib import SequenceMatcher
from typing import List, Tuple, Dict, Optional


class HallucinationVerifier:
    """Verifies hallucination claims and partial match scores against ground truth"""

    def __init__(self, fuzzy_threshold: float = 0.85):
        """
        Initialize the verifier.

        Args:
            fuzzy_threshold: Similarity threshold for fuzzy matching (0-1)
                           0.85 means 85% similarity is considered a match
        """
        self.fuzzy_threshold = fuzzy_threshold

    def verify_hallucinations(
        self,
        hallucination_text: str,
        ground_truth: str
    ) -> Dict[str, any]:
        """
        Verify hallucination claims against ground truth.
        Only returns claims that are genuinely NOT in the ground truth.

        Args:
            hallucination_text: Pipe-separated hallucination claims from evaluator
            ground_truth: Ground truth answer text

        Returns:
            {
                'verified_hallucinations': List of verified hallucinations,
                'false_positives': List of claims that ARE in ground truth,
                'verification_details': Details for each claim
            }
        """
        if not hallucination_text or hallucination_text.strip().lower() in ["none", "none."]:
            return {
                'verified_hallucinations': [],
                'false_positives': [],
                'verification_details': []
            }

        # Split by pipe separator
        claims = [c.strip() for c in hallucination_text.split('|') if c.strip()]

        verified = []
        false_positives = []
        details = []

        # Normalize ground truth for comparison
        gt_normalized = self._normalize_text(ground_truth)

        for claim in claims:
            claim_normalized = self._normalize_text(claim)

            # Check if claim exists in ground truth
            is_in_gt = self._is_present_in_text(
                claim_normalized,
                gt_normalized
            )

            similarity_score = self._compute_max_similarity(
                claim_normalized,
                gt_normalized
            )

            detail = {
                'claim': claim,
                'is_in_ground_truth': is_in_gt,
                'similarity_score': similarity_score
            }
            details.append(detail)

            if is_in_gt:
                # Claim is found in ground truth - false positive
                false_positives.append(claim)
            else:
                # Claim is NOT in ground truth - verified hallucination
                verified.append(claim)

        return {
            'verified_hallucinations': verified,
            'false_positives': false_positives,
            'verification_details': details
        }

    def verify_missing_info(
        self,
        missing_info_text: str,
        ground_truth: str
    ) -> Dict[str, any]:
        """
        Verify missing information claims against ground truth.
        Ensures that claimed missing items are actually IN the ground truth.

        Args:
            missing_info_text: Pipe-separated missing info claims
            ground_truth: Ground truth answer text

        Returns:
            {
                'verified_missing': List of verified missing items,
                'false_positives': List of items that are NOT in ground truth,
                'verification_details': Details for each claim
            }
        """
        if not missing_info_text or missing_info_text.strip().lower() in ["none", "none."]:
            return {
                'verified_missing': [],
                'false_positives': [],
                'verification_details': []
            }

        # Split by pipe separator
        items = [i.strip() for i in missing_info_text.split('|') if i.strip()]

        verified = []
        false_positives = []
        details = []

        gt_normalized = self._normalize_text(ground_truth)

        for item in items:
            item_normalized = self._normalize_text(item)

            # Check if item SHOULD be in ground truth
            is_in_gt = self._is_present_in_text(
                item_normalized,
                gt_normalized
            )

            similarity_score = self._compute_max_similarity(
                item_normalized,
                gt_normalized
            )

            detail = {
                'missing_item': item,
                'is_in_ground_truth': is_in_gt,
                'similarity_score': similarity_score
            }
            details.append(detail)

            if is_in_gt:
                # Good: item IS in ground truth and was marked as missing from response
                verified.append(item)
            else:
                # Bad: item is NOT in ground truth, so it can't be "missing"
                false_positives.append(item)

        return {
            'verified_missing': verified,
            'false_positives': false_positives,
            'verification_details': details
        }

    def verify_partial_match(
        self,
        response: str,
        ground_truth: str,
        min_overlap: float = 0.2
    ) -> bool:
        """
        Verify that a response has partial overlap with ground truth.
        Used to validate 0.5 scores.

        Args:
            response: The model's response
            ground_truth: Ground truth answer
            min_overlap: Minimum overlap percentage (0-1) to consider partial match

        Returns:
            True if response has at least min_overlap with ground truth
        """
        if not response or not ground_truth:
            return False

        response_normalized = self._normalize_text(response)
        gt_normalized = self._normalize_text(ground_truth)

        # Calculate overlap percentage
        overlap_pct = self._compute_overlap_percentage(
            response_normalized,
            gt_normalized
        )

        return overlap_pct >= min_overlap

    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for comparison.

        Args:
            text: Input text

        Returns:
            Normalized text (lowercase, no extra whitespace, no punctuation)
        """
        if not text:
            return ""

        # Lowercase
        text = text.lower()

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Remove punctuation for fuzzy matching
        text = re.sub(r'[^\w\s]', '', text)

        return text

    def _is_present_in_text(
        self,
        substring: str,
        full_text: str
    ) -> bool:
        """
        Check if substring is present in full_text.
        Uses both exact and fuzzy matching.

        Args:
            substring: Text to search for
            full_text: Text to search in

        Returns:
            True if substring found (exactly or fuzzily)
        """
        if not substring or not full_text:
            return False

        # Exact substring match
        if substring in full_text:
            return True

        # Fuzzy match for paraphrases
        # Check if any sliding window in full_text matches substring
        words_substring = substring.split()
        words_fulltext = full_text.split()

        if len(words_substring) == 0 or len(words_fulltext) == 0:
            return False

        # Sliding window approach
        for i in range(len(words_fulltext) - len(words_substring) + 1):
            window = ' '.join(words_fulltext[i:i+len(words_substring)])
            similarity = SequenceMatcher(None, substring, window).ratio()

            if similarity >= self.fuzzy_threshold:
                return True

        # Also check with shorter windows for partial matches
        # This helps catch cases where the claim is longer than any phrase in ground truth
        if len(words_substring) > 3:
            # Try with windows of varying sizes
            for window_size in range(3, min(len(words_substring), len(words_fulltext)) + 1):
                for i in range(len(words_fulltext) - window_size + 1):
                    window = ' '.join(words_fulltext[i:i+window_size])
                    similarity = SequenceMatcher(None, substring, window).ratio()

                    if similarity >= self.fuzzy_threshold:
                        return True

        return False

    def _compute_max_similarity(
        self,
        substring: str,
        full_text: str
    ) -> float:
        """
        Compute maximum similarity between substring and any part of full_text.

        Args:
            substring: Text to compare
            full_text: Text to compare against

        Returns:
            Maximum similarity score (0-1)
        """
        if not substring or not full_text:
            return 0.0

        words_substring = substring.split()
        words_fulltext = full_text.split()

        if len(words_substring) == 0 or len(words_fulltext) == 0:
            return 0.0

        max_sim = 0.0

        # Check similarity with sliding windows
        for i in range(len(words_fulltext) - len(words_substring) + 1):
            window = ' '.join(words_fulltext[i:i+len(words_substring)])
            similarity = SequenceMatcher(None, substring, window).ratio()
            max_sim = max(max_sim, similarity)

        # Also check with varying window sizes
        for window_size in range(1, min(len(words_substring) + 1, len(words_fulltext) + 1)):
            for i in range(len(words_fulltext) - window_size + 1):
                window = ' '.join(words_fulltext[i:i+window_size])
                similarity = SequenceMatcher(None, substring, window).ratio()
                max_sim = max(max_sim, similarity)

        return max_sim

    def _compute_overlap_percentage(
        self,
        response: str,
        ground_truth: str
    ) -> float:
        """
        Compute percentage of ground truth that appears in response.

        Args:
            response: Model's response (normalized)
            ground_truth: Ground truth answer (normalized)

        Returns:
            Overlap percentage (0-1)
        """
        if not ground_truth:
            return 0.0

        if not response:
            return 0.0

        # Split into words
        gt_words = ground_truth.split()
        resp_words = set(response.split())

        if len(gt_words) == 0:
            return 0.0

        # Count how many ground truth words appear in response
        matched_words = sum(1 for word in gt_words if word in resp_words)

        return matched_words / len(gt_words)


class LLMSemanticVerifier:
    """
    LLM-based semantic verification for hallucinations and missing information.
    Uses Qwen2.5-7B-Instruct to understand semantic equivalence and paraphrases.
    """

    def __init__(self):
        """Initialize the LLM semantic verifier"""
        pass

    def _clean_claims(self, text: str) -> List[str]:
        """
        Clean and parse pipe-separated claims, filtering out "None" variants.
        Handles malformed outputs like "claim OR None" or "claim or none".

        Args:
            text: Pipe-separated claims or "None"

        Returns:
            List of cleaned claims
        """
        if not text or text.strip().lower() in ["none", "none.", ""]:
            return []

        claims = [c.strip() for c in text.split('|') if c.strip()]

        # Filter explicit "None" variants
        claims = [c for c in claims if c.lower() not in ["none", "none.", ""]]

        # Handle malformed "X OR None" or "X or none" patterns
        cleaned_claims = []
        for claim in claims:
            # Remove " OR None" suffix (case-insensitive)
            claim_cleaned = re.sub(r'\s+(or|OR)\s+(none|None|NONE)\.?$', '', claim).strip()
            if claim_cleaned and claim_cleaned.lower() not in ["none", "none.", ""]:
                cleaned_claims.append(claim_cleaned)

        return cleaned_claims

    def _ask_llm_binary(self, prompt: str, model, tokenizer) -> bool:
        """
        Ask LLM a binary Yes/No question.

        Args:
            prompt: The question to ask
            model: Loaded Qwen2.5-7B-Instruct model
            tokenizer: Corresponding tokenizer

        Returns:
            True if answer is Yes, False if No or parsing fails
        """
        import torch

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = tokenizer([text], return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        generated = outputs[0][inputs["input_ids"].shape[-1]:]
        response = tokenizer.decode(generated, skip_special_tokens=True).strip()

        # Robust parsing: find first Yes or No (case-insensitive)
        match = re.search(r'\b(yes|no)\b', response.lower())
        if match:
            answer = match.group(1).lower() == 'yes'
            return answer
        else:
            print(f"WARNING: Could not parse Yes/No from LLM response: {response}")
            return False  # Conservative default

    def verify_hallucinations_llm(
        self,
        hallucination_text: str,
        ground_truth: str,
        model,
        tokenizer
    ) -> Dict[str, any]:
        """
        Verify hallucination claims using LLM semantic understanding.

        Args:
            hallucination_text: Pipe-separated hallucination claims
            ground_truth: Ground truth answer text
            model: Loaded Qwen2.5-7B-Instruct model
            tokenizer: Corresponding tokenizer

        Returns:
            {
                'verified_hallucinations': List of claims verified as NOT in GT,
                'false_positives': List of claims that ARE in GT,
                'verification_details': Details for each claim
            }
        """
        claims = self._clean_claims(hallucination_text)

        if not claims:
            return {
                'verified_hallucinations': [],
                'false_positives': [],
                'verification_details': []
            }

        verified = []
        false_positives = []
        details = []

        for claim in claims:
            # Prompt LLM to check if claim is in ground truth
            prompt = f"""You are verifying if a claim is present or implied in a ground truth answer.

Ground Truth: {ground_truth}
Claim to verify: {claim}

Question: Is the claim present or semantically implied in the ground truth?
Consider paraphrases and semantic equivalence.
IMPORTANT: Only check if the claim matches the ground truth. Ignore whether the claim might be factually correct from other sources.

You must respond with EXACTLY one word: Yes or No
Do not add any explanation, punctuation, or additional text.

Examples:
- GT: "A plush toy puppy", Claim: "IFAW logo" → No
- GT: "PSA format", Claim: "Public service announcement" → Yes
- GT: "live in the home temporarily", Claim: "overnight stays" → Yes

Your answer (Yes or No):"""

            is_in_gt = self._ask_llm_binary(prompt, model, tokenizer)

            detail = {
                'claim': claim,
                'is_in_ground_truth': is_in_gt,
                'llm_verified': True
            }
            details.append(detail)

            if is_in_gt:
                # Claim is found in ground truth - false positive
                false_positives.append(claim)
            else:
                # Claim is NOT in ground truth - verified hallucination
                verified.append(claim)

        return {
            'verified_hallucinations': verified,
            'false_positives': false_positives,
            'verification_details': details
        }

    def verify_missing_info_llm(
        self,
        missing_info_text: str,
        ground_truth: str,
        model,
        tokenizer
    ) -> Dict[str, any]:
        """
        Verify missing information claims using LLM semantic understanding.

        Args:
            missing_info_text: Pipe-separated missing info claims
            ground_truth: Ground truth answer text
            model: Loaded Qwen2.5-7B-Instruct model
            tokenizer: Corresponding tokenizer

        Returns:
            {
                'verified_missing': List of verified missing items,
                'false_positives': List of items that are NOT in ground truth,
                'verification_details': Details for each claim
            }
        """
        items = self._clean_claims(missing_info_text)

        if not items:
            return {
                'verified_missing': [],
                'false_positives': [],
                'verification_details': []
            }

        verified = []
        false_positives = []
        details = []

        for item in items:
            # Prompt LLM to check if item is in ground truth
            prompt = f"""You are verifying if information is present or implied in a ground truth answer.

Ground Truth: {ground_truth}
Information to verify: {item}

Question: Is this information present or semantically implied in the ground truth?
Consider paraphrases and semantic equivalence.

You must respond with EXACTLY one word: Yes or No
Do not add any explanation, punctuation, or additional text.

Your answer (Yes or No):"""

            is_in_gt = self._ask_llm_binary(prompt, model, tokenizer)

            detail = {
                'missing_item': item,
                'is_in_ground_truth': is_in_gt,
                'llm_verified': True
            }
            details.append(detail)

            if is_in_gt:
                # Good: item IS in ground truth and was marked as missing from response
                verified.append(item)
            else:
                # Bad: item is NOT in ground truth, so it can't be "missing"
                false_positives.append(item)

        return {
            'verified_missing': verified,
            'false_positives': false_positives,
            'verification_details': details
        }

    def verify_partial_match_llm(
        self,
        response: str,
        ground_truth: str,
        model,
        tokenizer
    ) -> bool:
        """
        Verify that a response has partial semantic overlap with ground truth using LLM.

        Args:
            response: The model's response
            ground_truth: Ground truth answer
            model: Loaded Qwen2.5-7B-Instruct model
            tokenizer: Corresponding tokenizer

        Returns:
            True if response contains at least SOME key information from ground truth
        """
        if not response or not ground_truth:
            return False

        prompt = f"""You are verifying if a response contains at least some key information from the ground truth.

Ground Truth: {ground_truth}
Response: {response}

Question: Does the response contain at least SOME of the key information from the ground truth?
Consider paraphrases and semantic equivalence. Ignore additional details not in ground truth.

You must respond with EXACTLY one word: Yes or No
Do not add any explanation, punctuation, or additional text.

Your answer (Yes or No):"""

        has_overlap = self._ask_llm_binary(prompt, model, tokenizer)

        return has_overlap
