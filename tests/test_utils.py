"""
Test suite for REALISTA — Dictionary & Utilities
==================================================
Deterministic tests for dictionary utilities, seed fixing, and packaging.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import inspect


class TestDictionaryUtilsSignatures:
    """Verify dictionary loading functions have expected signatures."""

    def test_load_rephrasing_prompts_signature(self):
        from src.dictionary_utils import load_rephrasing_prompts
        sig = inspect.signature(load_rephrasing_prompts)
        assert "subject" in sig.parameters

    def test_load_latent_dict_signature(self):
        from src.dictionary_utils import load_latent_dict
        sig = inspect.signature(load_latent_dict)
        params = list(sig.parameters.keys())
        assert "model_type" in params
        assert "mmlu_subject" in params

    def test_get_original_latent_signature(self):
        from src.dictionary_utils import get_original_latent
        sig = inspect.signature(get_original_latent)
        params = list(sig.parameters.keys())
        assert "model" in params
        assert "tokenizer" in params
        assert "question_text" in params
        assert "model_type" in params


class TestDictConstructionSignatures:
    """Verify dictionary construction functions."""

    def test_wordnet_concepts_returns_list(self):
        from src.optional_dict_construction.dict_construction_utils import (
            get_wordnet_adjective_concepts,
        )
        concepts = get_wordnet_adjective_concepts(pool_size=10, seed=0)
        assert isinstance(concepts, list)
        assert len(concepts) == 10
        assert all(isinstance(c, str) for c in concepts)

    def test_wordnet_concepts_deterministic(self):
        from src.optional_dict_construction.dict_construction_utils import (
            get_wordnet_adjective_concepts,
        )
        c1 = get_wordnet_adjective_concepts(pool_size=20, seed=42)
        c2 = get_wordnet_adjective_concepts(pool_size=20, seed=42)
        assert c1 == c2

    def test_select_concepts_signature(self):
        from src.optional_dict_construction.dict_construction_utils import select_concepts
        sig = inspect.signature(select_concepts)
        params = list(sig.parameters.keys())
        assert "question_embedding" in params
        assert "concept_embeddings" in params
        assert "editability_scores" in params
        assert "n_select" in params

    def test_build_latent_directions_signature(self):
        from src.optional_dict_construction.dict_construction_utils import build_latent_directions
        sig = inspect.signature(build_latent_directions)
        params = list(sig.parameters.keys())
        assert "model" in params
        assert "rephrasings" in params

    def test_editability_instruction_exists(self):
        from src.optional_dict_construction.dict_construction_utils import EDITABILITY_INSTRUCTION
        assert isinstance(EDITABILITY_INSTRUCTION, str)
        assert len(EDITABILITY_INSTRUCTION) > 100
        assert "editability" in EDITABILITY_INSTRUCTION.lower()


class TestSeedReproducibility:
    """Verify seed fixing produces deterministic results."""

    def test_seed_produces_same_torch_random(self):
        import torch
        from src.utils import set_seed
        set_seed(42)
        a = torch.randn(10)
        set_seed(42)
        b = torch.randn(10)
        assert torch.allclose(a, b)

    def test_seed_produces_same_numpy_random(self):
        import numpy as np
        from src.utils import set_seed
        set_seed(99)
        a = np.random.randn(10)
        set_seed(99)
        b = np.random.randn(10)
        assert (a == b).all()

    def test_different_seeds_different_output(self):
        import torch
        from src.utils import set_seed
        set_seed(1)
        a = torch.randn(10)
        set_seed(2)
        b = torch.randn(10)
        assert not torch.allclose(a, b)


class TestModelUtilsStructure:
    """Verify model utils module structure."""

    def test_gpt_class_exists(self):
        from src.model_utils import GPT
        assert hasattr(GPT, "generate")
        assert hasattr(GPT, "API_MAX_RETRY")

    def test_gpt_retry_count(self):
        from src.model_utils import GPT
        assert GPT.API_MAX_RETRY >= 3

    def test_load_function_exists(self):
        from src.model_utils import load_model_and_tokenizer
        sig = inspect.signature(load_model_and_tokenizer)
        assert "model_type" in sig.parameters
