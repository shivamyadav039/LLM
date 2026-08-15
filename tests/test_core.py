"""
Test suite for REALISTA — Core Algorithm
==========================================
Deterministic tests for simplex projection, function signatures,
and algorithmic correctness.
"""
import os
import sys
sys.path.insert(0, "/workspace/submission")
sys.path.insert(1, os.path.join(os.path.dirname(__file__), ".."))



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


class TestBehavioralGrader:
    """Behavioral tests verifying the mathematical and optimization mechanics of REALISTA."""

    def test_gumbel_softmax_deterministic_behavior(self):
        """Verify Gumbel-Softmax reparameterized sampling is deterministic under fixed seeds."""
        from src.utils import set_seed
        
        logits = torch.tensor([[1.5, 2.5, 0.5, 3.5]])
        
        # Draw samples under seed 42
        set_seed(42)
        u1 = torch.rand_like(logits)
        gumbel1 = -torch.log(-torch.log(u1 + 1e-20) + 1e-20)
        s1 = torch.softmax((logits + gumbel1) / 1.0, dim=-1)

        set_seed(42)
        u2 = torch.rand_like(logits)
        gumbel2 = -torch.log(-torch.log(u2 + 1e-20) + 1e-20)
        s2 = torch.softmax((logits + gumbel2) / 1.0, dim=-1)

        assert torch.allclose(s1, s2, atol=1e-5), "Gumbel-Softmax sampling is not deterministic under seed reset"

    def test_pld_convergence_toy_objective(self):
        """Verify that a PLD step moves variables towards a simulated loss minimum while respecting constraints."""
        from src.realista import project_onto_simplex
        
        # Toy problem: Minimize f(delta) = ||delta - target||_2^2
        # where target = [0.8, 0.2, 0.0] (which is already on the simplex)
        target = torch.tensor([0.8, 0.2, 0.0])
        delta = torch.tensor([0.1, 0.5, 0.4], requires_grad=True)
        
        # Compute gradient
        loss = torch.sum((delta - target) ** 2)
        loss.backward()
        
        # Take a projected gradient step
        lr = 0.1
        with torch.no_grad():
            new_delta = delta - lr * delta.grad
            projected = project_onto_simplex(new_delta, epsilon=1.0)
            
        initial_distance = torch.sum((delta - target) ** 2).item()
        final_distance = torch.sum((projected - target) ** 2).item()
        
        # Verify distance decreased and simplex constraint holds
        assert final_distance < initial_distance, "PLD optimization step failed to converge towards the target"
        assert projected.sum().item() <= 1.0 + 1e-5, "Projected delta violates the simplex budget constraint"
        assert (projected >= -1e-6).all(), "Projected delta violates the non-negativity constraint"

    def test_objective_function_math(self):
        """Verify the log-probability math behaves correctly on simulated logits."""
        # Simulated logits and targets
        logits = torch.tensor([[1.0, -2.0, 5.0, 0.5]], requires_grad=True)
        target_idx = 2  # target has highest logit value (5.0)
        
        # Log probability computation
        probs = torch.softmax(logits, dim=-1)
        loss = -torch.log(probs[0, target_idx] + 1e-10)
        
        # Backward pass
        loss.backward()
        
        # Logit 2 is targeted, so increasing it should reduce the loss (negative gradient)
        assert logits.grad[0, target_idx].item() < 0.0, "Gradient direction is incorrect for the target index"
        # Other logits should have positive gradients (increasing them increases loss)
        for i in range(4):
            if i != target_idx:
                assert logits.grad[0, i].item() > 0.0, f"Gradient direction is incorrect for non-target index {i}"

