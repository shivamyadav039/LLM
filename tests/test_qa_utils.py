"""
Test suite for REALISTA — QA Utilities
=======================================
Deterministic tests for prompt construction and probability formatting.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch


class TestTokenIDMapping:
    """Verify ABCD token ID mappings exist for all models."""

    def test_all_models_have_mappings(self):
        from src.qa_utils import CHOICE_TOKEN_IDS
        from src.config import MODEL_REGISTRY
        for model_key in MODEL_REGISTRY:
            assert model_key in CHOICE_TOKEN_IDS

    def test_each_mapping_has_four_choices(self):
        from src.qa_utils import CHOICE_TOKEN_IDS
        for model, mapping in CHOICE_TOKEN_IDS.items():
            assert set(mapping.keys()) == {"A", "B", "C", "D"}

    def test_token_ids_are_positive_integers(self):
        from src.qa_utils import CHOICE_TOKEN_IDS
        for model, mapping in CHOICE_TOKEN_IDS.items():
            for letter, tid in mapping.items():
                assert isinstance(tid, int) and tid > 0


class TestPromptConstruction:
    """Verify prompt templates for both MCQA and reasoning settings."""

    def _make_task(self):
        return {
            "subject": "machine_learning",
            "choices": ["True, True", "True, False", "False, True", "False, False"],
        }

    def test_mcqa_prefix_contains_subject(self):
        from src.qa_utils import get_prompt
        prefix, _ = get_prompt(self._make_task())
        assert "machine learning" in prefix

    def test_mcqa_suffix_contains_answer_marker(self):
        from src.qa_utils import get_prompt
        _, suffix = get_prompt(self._make_task())
        assert "Answer:" in suffix

    def test_mcqa_suffix_contains_all_choices(self):
        from src.qa_utils import get_prompt
        _, suffix = get_prompt(self._make_task())
        assert "A." in suffix and "B." in suffix and "C." in suffix and "D." in suffix

    def test_reasoning_suffix_contains_step_by_step(self):
        from src.qa_utils import get_prompt
        _, suffix = get_prompt(self._make_task(), is_reasoning=True)
        assert "step by step" in suffix

    def test_reasoning_suffix_no_forced_answer(self):
        from src.qa_utils import get_prompt
        _, suffix = get_prompt(self._make_task(), is_reasoning=True)
        assert "Answer:" not in suffix

    def test_subject_underscore_replacement(self):
        from src.qa_utils import get_prompt
        task = {"subject": "computer_science", "choices": ["a", "b", "c", "d"]}
        prefix, _ = get_prompt(task)
        assert "computer science" in prefix
        assert "computer_science" not in prefix


class TestFormatProbs:
    """Verify probability formatting output."""

    def test_format_contains_all_letters(self):
        from src.qa_utils import format_probs
        result = format_probs([0.25, 0.25, 0.25, 0.25])
        assert "A:" in result and "B:" in result and "C:" in result and "D:" in result

    def test_format_percentages(self):
        from src.qa_utils import format_probs
        result = format_probs([0.1, 0.5, 0.3, 0.1])
        assert "50.00%" in result
        assert "10.00%" in result

    def test_format_zero_prob(self):
        from src.qa_utils import format_probs
        result = format_probs([0.0, 1.0, 0.0, 0.0])
        assert "100.00%" in result
        assert "0.00%" in result
