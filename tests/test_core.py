"""
Test suite for REALISTA — Core Algorithm
==========================================
Deterministic tests for simplex projection, function signatures,
and algorithmic correctness.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
import inspect


class TestSimplexProjectionCorrectness:
    """Verify the simplex projection enforces all mathematical constraints."""

    def test_non_negativity(self):
        from src.realista import project_onto_simplex
        v = torch.tensor([0.3, 0.5, 0.8, 0.1, -0.2])
        proj = project_onto_simplex(v, epsilon=1.0)
        assert (proj >= -1e-6).all(), f"Negative values found: {proj}"

    def test_budget_constraint(self):
        from src.realista import project_onto_simplex
        v = torch.tensor([0.3, 0.5, 0.8, 0.1, -0.2])
        proj = project_onto_simplex(v, epsilon=1.0)
        assert proj.sum().item() <= 1.0 + 1e-5

    def test_budget_exact(self):
        from src.realista import project_onto_simplex
        v = torch.tensor([2.0, 3.0, 1.0])
        proj = project_onto_simplex(v, epsilon=1.0)
        assert abs(proj.sum().item() - 1.0) < 1e-4

    def test_feasible_passthrough(self):
        from src.realista import project_onto_simplex
        v = torch.tensor([0.1, 0.2, 0.0])
        proj = project_onto_simplex(v, epsilon=1.0)
        assert torch.allclose(v, proj, atol=1e-6)

    def test_all_negative_gives_zero(self):
        from src.realista import project_onto_simplex
        v = torch.tensor([-1.0, -2.0, -3.0])
        proj = project_onto_simplex(v, epsilon=1.0)
        assert proj.sum().item() == 0.0

    def test_sparsity_induced(self):
        from src.realista import project_onto_simplex
        v = torch.tensor([0.3, 0.5, 0.8, 0.1, -0.2])
        proj = project_onto_simplex(v, epsilon=1.0)
        n_zero = (proj == 0).sum().item()
        assert n_zero >= 2, f"Expected sparsity, but only {n_zero} zeros"

    def test_different_epsilon(self):
        from src.realista import project_onto_simplex
        v = torch.tensor([1.0, 2.0, 3.0])
        proj = project_onto_simplex(v, epsilon=2.0)
        assert abs(proj.sum().item() - 2.0) < 1e-4
        assert (proj >= -1e-6).all()

    def test_single_element(self):
        from src.realista import project_onto_simplex
        v = torch.tensor([5.0])
        proj = project_onto_simplex(v, epsilon=1.0)
        assert abs(proj.item() - 1.0) < 1e-4


class TestFunctionSignatures:
    """Verify all core functions have the expected signatures."""

    def test_reconstruct_from_latent_signature(self):
        from src.realista import reconstruct_from_latent
        sig = inspect.signature(reconstruct_from_latent)
        params = list(sig.parameters.keys())
        assert "model" in params
        assert "tokenizer" in params
        assert "latent" in params

    def test_obj_fun_signature(self):
        from src.realista import obj_fun
        sig = inspect.signature(obj_fun)
        params = list(sig.parameters.keys())
        assert "full_input_embeds" in params
        assert "target_choice_index" in params

    def test_stage1_optimization_signature(self):
        from src.realista import stage1_optimization
        sig = inspect.signature(stage1_optimization)
        params = list(sig.parameters.keys())
        assert "latent_directions" in params
        assert "z0" in params
        assert "target_choice_index" in params

    def test_pld_signature(self):
        from src.realista import PLD
        sig = inspect.signature(PLD)
        params = list(sig.parameters.keys())
        assert "init_delta" in params
        assert "feasibility_evaluator" in params

    def test_pld_reasoning_signature(self):
        from src.realista import PLD_reasoning_model
        sig = inspect.signature(PLD_reasoning_model)
        params = list(sig.parameters.keys())
        assert "reasoning_target" in params
        assert "hallucination_evaluator" in params

    def test_run_realista_attack_signature(self):
        from src.realista import run_realista_attack
        sig = inspect.signature(run_realista_attack)
        params = list(sig.parameters.keys())
        assert "latent_directions" in params
        assert "z0" in params
        assert "feasibility_evaluator" in params

    def test_hallucination_judge_signature(self):
        from src.realista import hallucination_judge_score
        sig = inspect.signature(hallucination_judge_score)
        params = list(sig.parameters.keys())
        assert "input_query" in params
        assert "ground_truth" in params

    def test_feasibility_check_signature(self):
        from src.realista import feasibility_check
        sig = inspect.signature(feasibility_check)
        params = list(sig.parameters.keys())
        assert "query_x" in params
        assert "query_x0" in params
        assert "ground_truth_idx" in params
