"""
🎛️ Attack Hyperparameters
=========================

Every knob you can turn to control how REALISTA runs.
Instantiate with defaults for a quick test, or override
individual fields for fine-grained control.

Example:
    args = RealistaArgs()                          # sensible defaults
    args = RealistaArgs(trial_num=3, attack_budget=0.5)  # quick test
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class RealistaArgs:
    """All hyperparameters for a single REALISTA attack run.

    Organized into logical groups so you can quickly find what to tune.
    """

    # ── What to attack ────────────────────────────────────────────────
    # Pick a subject and question from the MMLU benchmark.
    mmlu_subject: str = "machine_learning"
    mmlu_question_idx: int = 0

    # ── Reproducibility ───────────────────────────────────────────────
    # Same seed = same results (random, numpy, torch all get seeded).
    seed: int = 18

    # ── Which LLM to fool ────────────────────────────────────────────
    # Must be a key in config.MODEL_REGISTRY:
    #   "llama3_3b"   → smallest, ~7 GB,  good for testing
    #   "llama3_8b"   → paper's main target, ~16 GB
    #   "qwen2_5_7b"  → alternative architecture, ~14 GB
    #   "qwen2_5_14b" → largest, ~28 GB
    model_type: str = "llama3_8b"

    # ── How hard to try ───────────────────────────────────────────────
    # Each "trial" is an independent PLD run from a different Stage-1
    # initialization. More trials = better chance of success, but slower.
    # Paper uses 10 trials; ASR@K reports "at least 1 of K succeeded."
    trial_num: int = 10

    # ── Reasoning model target ────────────────────────────────────────
    # Set to "none" to attack the open-source model directly.
    # Set to "gpt_5_nano" or "gpt_5_mini" to attack a reasoning model
    # via the free-form response setting (requires OpenAI API).
    reasoning_target: str = "none"

    # ── Stage-1 speed ─────────────────────────────────────────────────
    # How many concept rephrasings to score in parallel.
    # Higher = faster Stage 1, but uses more VRAM.
    stage1_batch_size: int = 16

    # ── Dictionary size control ───────────────────────────────────────
    # Set to an integer to randomly subsample the edit dictionary.
    # Useful during development (fewer concepts = much faster runs).
    # None = use the full dictionary (recommended for final results).
    stage2_concept_subsample: Optional[int] = None

    # ── PLD Optimization Knobs ────────────────────────────────────────
    #
    # These control the Projected Langevin Dynamics loop (Stage 2).
    # Think of PLD as "gradient descent + random exploration" — the noise
    # helps escape flat regions in the loss landscape.

    # Step size η: how far to move each iteration.
    # Too small → stuck on flat plateaus. Too large → divergence.
    pld_step_size: float = 0.05

    # Initial temperature T₀: controls noise magnitude at the start.
    # Higher = more exploration early on.
    pld_temperature_init: float = 1.0

    # Temperature decay γ ∈ [0, 1]: how fast noise decreases.
    # T_k = T₀ · γ^k, so γ=0.95 means noise halves every ~14 steps.
    pld_temperature_decay: float = 0.95

    # Number of PLD iterations per trial.
    # More iterations = better convergence, but diminishing returns past ~50.
    pld_iterations: int = 50

    # Attack budget ε: maximum ℓ₁ norm of δ (the editing strengths).
    # Bigger ε = more aggressive edits (higher ASR but more semantic drift).
    # The simplex constraint forces δ ≥ 0 and ‖δ‖₁ ≤ ε.
    attack_budget: float = 1.0

    # ── Decoder settings ──────────────────────────────────────────────
    # Max tokens the decoder can generate when inverting latent → text.
    # Most questions decode in 20-40 tokens; 50 is a safe upper bound.
    decode_prompt_len: int = 50

    # ── Stage-1 → Stage-2 handoff ────────────────────────────────────
    # How many of Stage 1's best candidates to carry into Stage 2.
    # Trials cycle through these initializations round-robin.
    stage1_top_n: int = 5
