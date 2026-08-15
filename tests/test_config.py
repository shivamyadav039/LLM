"""
Test suite for REALISTA — Configuration & Arguments
====================================================
Deterministic unit tests for config.py and arguments.py
"""
import os
import sys
sys.path.insert(0, "/workspace/submission")
sys.path.insert(1, os.path.join(os.path.dirname(__file__), ".."))




import pytest


class TestModelRegistry:
    """Verify the model registry contains all expected models."""

    def test_registry_has_four_models(self):
        from src.config import MODEL_REGISTRY
        assert len(MODEL_REGISTRY) == 4

    def test_registry_contains_llama3_3b(self):
        from src.config import MODEL_REGISTRY
        assert "llama3_3b" in MODEL_REGISTRY

    def test_registry_contains_llama3_8b(self):
        from src.config import MODEL_REGISTRY
        assert "llama3_8b" in MODEL_REGISTRY

    def test_registry_contains_qwen2_5_7b(self):
        from src.config import MODEL_REGISTRY
        assert "qwen2_5_7b" in MODEL_REGISTRY

    def test_registry_contains_qwen2_5_14b(self):
        from src.config import MODEL_REGISTRY
        assert "qwen2_5_14b" in MODEL_REGISTRY

    def test_registry_values_are_huggingface_paths(self):
        from src.config import MODEL_REGISTRY
        for key, path in MODEL_REGISTRY.items():
            assert "/" in path, f"{key} path '{path}' doesn't look like a HF path"


class TestLayerRegistry:
    """Verify the layer number registry matches the paper (layer 3)."""

    def test_layer_registry_has_all_models(self):
        from src.config import LAYER_NUM_REGISTRY, MODEL_REGISTRY
        for model_key in MODEL_REGISTRY:
            assert model_key in LAYER_NUM_REGISTRY

    def test_all_layers_are_3(self):
        from src.config import LAYER_NUM_REGISTRY
        for model, layer in LAYER_NUM_REGISTRY.items():
            assert layer == 3, f"{model} layer should be 3, got {layer}"


class TestReasoningTargetMap:
    """Verify reasoning target model mapping."""

    def test_gpt_5_nano_mapped(self):
        from src.config import REASONING_TARGET_MODEL_MAP
        assert "gpt_5_nano" in REASONING_TARGET_MODEL_MAP

    def test_gpt_5_mini_mapped(self):
        from src.config import REASONING_TARGET_MODEL_MAP
        assert "gpt_5_mini" in REASONING_TARGET_MODEL_MAP


class TestRealistaArgs:
    """Verify the hyperparameter dataclass defaults match the paper."""

    def test_default_model_type(self):
        from src.arguments import RealistaArgs
        args = RealistaArgs()
        assert args.model_type == "llama3_8b"

    def test_default_seed(self):
        from src.arguments import RealistaArgs
        assert RealistaArgs().seed == 18

    def test_default_trial_num(self):
        from src.arguments import RealistaArgs
        assert RealistaArgs().trial_num == 10

    def test_default_pld_iterations(self):
        from src.arguments import RealistaArgs
        assert RealistaArgs().pld_iterations == 50

    def test_default_attack_budget(self):
        from src.arguments import RealistaArgs
        assert RealistaArgs().attack_budget == 1.0

    def test_default_step_size(self):
        from src.arguments import RealistaArgs
        assert RealistaArgs().pld_step_size == 0.05

    def test_default_temperature_decay(self):
        from src.arguments import RealistaArgs
        assert RealistaArgs().pld_temperature_decay == 0.95

    def test_override_works(self):
        from src.arguments import RealistaArgs
        args = RealistaArgs(trial_num=3, attack_budget=0.5)
        assert args.trial_num == 3
        assert args.attack_budget == 0.5

    def test_stage1_top_n_default(self):
        from src.arguments import RealistaArgs
        assert RealistaArgs().stage1_top_n == 5
