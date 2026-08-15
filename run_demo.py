#!/usr/bin/env python3
"""
🚀 REALISTA — End-to-End Attack Runner
========================================

This is the main entry point. It wires everything together:

  1. Parse CLI arguments → RealistaArgs
  2. Load the target LLM (open-source, local)
  3. Load an MMLU question from the benchmark
  4. Build or load the edit dictionary (latent directions)
  5. Set up LLM judges (if OpenAI key available)
  6. Run the full two-stage REALISTA attack
  7. Report results and save to JSON

Quick start:
  python run_demo.py                                    # defaults (Llama-3.1-8B)
  python run_demo.py --model_type llama3_3b --trial_num 1 --pld_iterations 10  # fast test
  python run_demo.py --reasoning_target gpt_5_nano      # attack a reasoning model

What happens when you run this:

  ┌────────────────────────────────────────────────────────────────┐
  │  Load Llama-3.1-8B (frozen, fp16)                            │
  │         ↓                                                     │
  │  Load MMLU question: "What is supervised learning?"           │
  │         ↓                                                     │
  │  Build edit dictionary: 20 concept directions                 │
  │         ↓                                                     │
  │  Stage 1: Try each concept → pick top 5                      │
  │         ↓                                                     │
  │  Stage 2: 10 PLD trials × 50 iterations each                 │
  │         ↓                                                     │
  │  Result: "In the domain of computational learning..."        │
  │  → LLM now picks the WRONG answer with 68% probability!     │
  └────────────────────────────────────────────────────────────────┘
"""
import argparse
import json
import os
import sys

import torch
from datasets import load_dataset

# Ensure src/ is importable when running from the project root
sys.path.insert(0, os.path.dirname(__file__))

from src.arguments import RealistaArgs
from src.config import (
    MODEL_REGISTRY, MMLU_DATASET, REASONING_TARGET_MODEL_MAP,
    FEASIBILITY_CHECKER_MODEL, HALLUCINATION_EVALUATOR_MODEL,
)
from src.model_utils import load_model_and_tokenizer, GPT
from src.dictionary_utils import (
    get_original_latent, load_rephrasing_prompts,
)
from src.optional_dict_construction.dict_construction_utils import (
    build_latent_directions,
)
from src.realista import run_realista_attack
from src.utils import set_seed


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CLI Argument Parsing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def parse_args() -> RealistaArgs:
    """Parse command-line arguments into a RealistaArgs dataclass.

    Every field in RealistaArgs is exposed as a CLI flag, so you can
    override any hyperparameter from the command line without editing code.
    """
    parser = argparse.ArgumentParser(
        description="🚀 REALISTA: Realistic Latent Adversarial Attacks "
                    "that Elicit LLM Hallucinations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # What to attack
    parser.add_argument("--mmlu_subject", type=str, default="machine_learning",
                        help="MMLU subject (e.g. machine_learning, anatomy)")
    parser.add_argument("--mmlu_question_idx", type=int, default=0,
                        help="Question index within the subject")

    # Reproducibility
    parser.add_argument("--seed", type=int, default=18,
                        help="Random seed for reproducibility")

    # Model selection
    parser.add_argument("--model_type", type=str, default="llama3_8b",
                        choices=list(MODEL_REGISTRY.keys()),
                        help="Target LLM to attack")

    # Attack intensity
    parser.add_argument("--trial_num", type=int, default=10,
                        help="Number of independent PLD trials")
    parser.add_argument("--reasoning_target", type=str, default="none",
                        help="Reasoning model target (none, gpt_5_nano, gpt_5_mini)")

    # Stage 1
    parser.add_argument("--stage1_batch_size", type=int, default=16,
                        help="Batch size for Stage 1 concept scoring")
    parser.add_argument("--stage2_concept_subsample", type=int, default=None,
                        help="Subsample edit dictionary for faster runs")

    # PLD hyperparameters
    parser.add_argument("--pld_step_size", type=float, default=0.05,
                        help="PLD gradient step size η")
    parser.add_argument("--pld_temperature_init", type=float, default=1.0,
                        help="Initial Langevin noise temperature T₀")
    parser.add_argument("--pld_temperature_decay", type=float, default=0.95,
                        help="Temperature decay rate γ (T_k = T₀·γ^k)")
    parser.add_argument("--pld_iterations", type=int, default=50,
                        help="PLD optimization steps per trial")
    parser.add_argument("--attack_budget", type=float, default=1.0,
                        help="Attack budget ε (ℓ₁ bound on δ)")
    parser.add_argument("--decode_prompt_len", type=int, default=50,
                        help="Max tokens for latent decoder")
    parser.add_argument("--stage1_top_n", type=int, default=5,
                        help="How many Stage 1 winners to carry forward")

    cli_args = parser.parse_args()
    return RealistaArgs(**vars(cli_args))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MMLU Question Loader
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_mmlu_question(subject: str, question_idx: int) -> dict:
    """Load a single multiple-choice question from the MMLU benchmark.

    Tries to download from HuggingFace first. If that fails (e.g. no
    internet), falls back to a hard-coded example question so you can
    still test the pipeline.

    Returns
    -------
    dict with keys: "question", "choices", "answer" (0-3), "subject"
    """
    print(f"📝 Loading MMLU question: {subject} #{question_idx}")

    try:
        dataset = load_dataset(MMLU_DATASET, subject, split="test")
        row = dataset[question_idx]
        return {
            "question": row["question"],
            "choices": row["choices"],
            "answer": row["answer"],
            "subject": subject,
        }
    except Exception as e:
        print(f"  ⚠️ Could not load from HuggingFace: {e}")
        print("  📦 Using built-in fallback question for demo...")

        # A machine_learning question that works well for testing
        return {
            "question": (
                "Statement 1| Linear regression is a supervised learning "
                "algorithm. Statement 2| Gradient descent is guaranteed to "
                "find the global minimum for any cost function."
            ),
            "choices": [
                "True, True",
                "True, False",
                "False, True",
                "False, False",
            ],
            "answer": 1,  # B: True, False
            "subject": subject,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Dictionary Builder (from pre-computed rephrasings)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_directions_from_rephrasings(
    model, tokenizer, question_text: str, rephrasings_data: dict,
    question_idx: int, model_type: str,
):
    """Convert pre-computed rephrasings into latent editing directions.

    This bridges the gap between the JSON rephrasing files and the
    tensor-based latent directions that the attack algorithm needs.

    For each rephrasing text:
      z^(i) = φ(rephrasing_i) − φ(original_question)

    Parameters
    ----------
    rephrasings_data : dict — output of load_rephrasing_prompts()
    question_idx     : int  — which question in the subject

    Returns
    -------
    directions : Tensor [n_concepts, L, d_model]
    z0         : Tensor [1, L, d_model]
    """
    # Look up rephrasings for this question
    key = str(question_idx)
    if key not in rephrasings_data:
        # Try finding ANY key that works
        for k, v in rephrasings_data.items():
            if isinstance(v, dict):
                key = k
                break
        else:
            raise KeyError(
                f"Question index {question_idx} not found in rephrasings. "
                f"Available keys: {list(rephrasings_data.keys())[:5]}..."
            )

    question_rephrasings = rephrasings_data[key]

    # Collect one rephrasing per concept
    all_rephrasings = []
    if isinstance(question_rephrasings, dict):
        # Format: {concept_name: [rephrasing1, rephrasing2, ...]}
        for concept_name, rephrasings_list in question_rephrasings.items():
            if isinstance(rephrasings_list, list) and rephrasings_list:
                all_rephrasings.append(rephrasings_list[0])  # first rephrasing per concept
            elif isinstance(rephrasings_list, str):
                all_rephrasings.append(rephrasings_list)
    elif isinstance(question_rephrasings, list):
        all_rephrasings = question_rephrasings

    if not all_rephrasings:
        raise ValueError("No rephrasings found for this question!")

    print(f"  🔧 Building latent directions from {len(all_rephrasings)} rephrasings...")

    directions, z0 = build_latent_directions(
        model, tokenizer, question_text, all_rephrasings, model_type,
    )

    print(f"  ✅ Latent directions shape: {directions.shape}")
    return directions, z0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Fallback: Simple Text Generation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def model_generate_simple(model, tokenizer, prompt: str, max_tokens: int) -> str:
    """Generate text using the target model directly.

    This is a fallback for when no OpenAI API is available for
    rephrasing generation — we use the target model itself.
    Not ideal (the attacker shouldn't use the target for prep),
    but works well enough for demos.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    new_tokens = output_ids[0, inputs.input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  🏁 Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    args = parse_args()
    set_seed(args.seed)

    # ── Banner ────────────────────────────────────────────────────────
    print()
    print("  ╔═══════════════════════════════════════════════════════╗")
    print("  ║   ⚔️  REALISTA — Realistic Latent Adversarial        ║")
    print("  ║       Attacks that Elicit LLM Hallucinations         ║")
    print("  ║                                                      ║")
    print("  ║   Paper: arxiv.org/abs/2605.12813 (ICML 2026)       ║")
    print("  ╚═══════════════════════════════════════════════════════╝")
    print(f"\n  Config: {args}\n")

    # ── Step 1: Load the target LLM ───────────────────────────────────
    print("━" * 60)
    print("  Step 1/6: Loading target model...")
    print("━" * 60)
    model, tokenizer = load_model_and_tokenizer(args.model_type)

    # ── Step 2: Load the MMLU question ────────────────────────────────
    print("\n" + "━" * 60)
    print("  Step 2/6: Loading benchmark question...")
    print("━" * 60)
    cur_task_dict = load_mmlu_question(args.mmlu_subject, args.mmlu_question_idx)

    print(f"\n  Question: {cur_task_dict['question'][:80]}...")
    print(f"  Choices:  {cur_task_dict['choices']}")
    print(f"  Answer:   {chr(65 + cur_task_dict['answer'])}")

    # ── Step 3: Build/load the edit dictionary ────────────────────────
    print("\n" + "━" * 60)
    print("  Step 3/6: Building edit dictionary...")
    print("━" * 60)

    try:
        # Try loading pre-computed rephrasings first
        rephrasings = load_rephrasing_prompts(args.mmlu_subject)
        directions, z0 = build_directions_from_rephrasings(
            model, tokenizer, cur_task_dict["question"],
            rephrasings, args.mmlu_question_idx, args.model_type,
        )
    except FileNotFoundError:
        # No pre-computed data? Build from scratch using the target model
        print("\n  📦 No pre-computed rephrasings found. Building from scratch...")
        print("  (Using the target model for rephrasing generation)")

        from src.optional_dict_construction.dict_construction_utils import (
            generate_concept_rephrasings,
        )

        # Use a handful of manually chosen concepts for a quick demo
        demo_concepts = ["concise", "formal", "indirect", "passive", "elaborate"]
        all_rephrasings = []

        for concept in demo_concepts:
            print(f"    Generating rephrasings for concept: '{concept}'...")

            # Create a simple generator that uses the target model
            class SimpleGenerator:
                """Minimal wrapper to match the .generate() interface."""
                def generate(self, messages, **kwargs):
                    return model_generate_simple(
                        model, tokenizer, messages[-1]["content"], 512,
                    )

            rephrasings_for_concept = generate_concept_rephrasings(
                cur_task_dict["question"], concept,
                cur_task_dict["choices"], cur_task_dict["subject"],
                SimpleGenerator(),
                n_rephrasings=2,
            )
            all_rephrasings.extend(rephrasings_for_concept)

        if not all_rephrasings:
            print("  ❌ Could not generate any rephrasings. Exiting.")
            return

        directions, z0 = build_latent_directions(
            model, tokenizer, cur_task_dict["question"],
            all_rephrasings, args.model_type,
        )

    # Optional: subsample for faster iteration
    if args.stage2_concept_subsample is not None:
        n = min(args.stage2_concept_subsample, directions.shape[0])
        print(f"  🎲 Subsampling: {directions.shape[0]} → {n} concepts")
        indices = torch.randperm(directions.shape[0])[:n]
        directions = directions[indices]

    # ── Step 4: Set up LLM judges ─────────────────────────────────────
    print("\n" + "━" * 60)
    print("  Step 4/6: Setting up LLM judges...")
    print("━" * 60)

    reasoning_target = None
    hallucination_evaluator = None
    feasibility_evaluator = None

    # Reasoning model target (if attacking GPT-5 etc.)
    if args.reasoning_target != "none":
        target_model_name = REASONING_TARGET_MODEL_MAP.get(args.reasoning_target)
        if target_model_name:
            reasoning_target = GPT(target_model_name)
            print(f"  🎯 Reasoning target: {target_model_name}")

    # LLM judges (need OpenAI API key)
    from src.config import OPENAI_API_KEY
    if OPENAI_API_KEY:
        hallucination_evaluator = GPT(HALLUCINATION_EVALUATOR_MODEL)
        feasibility_evaluator = GPT(FEASIBILITY_CHECKER_MODEL)
        print("  ✅ Judges loaded: hallucination evaluator + feasibility checker")
    else:
        print("  ⚠️  No OPENAI_API_KEY found — LLM judges will be skipped.")
        print("     Set OPENAI_API_KEY in .env for the full attack pipeline.")

    # ── Step 5: Run the attack! ───────────────────────────────────────
    print("\n" + "━" * 60)
    print("  Step 5/6: Running REALISTA attack...")
    print("━" * 60)

    results = run_realista_attack(
        args, model, tokenizer, cur_task_dict,
        directions, z0,
        reasoning_target=reasoning_target,
        hallucination_evaluator=hallucination_evaluator,
        feasibility_evaluator=feasibility_evaluator,
    )

    # ── Step 6: Report and save ───────────────────────────────────────
    print("\n" + "━" * 60)
    print("  Step 6/6: Final Results")
    print("━" * 60)

    best = results["best_trial"]
    success_emoji = "🎉" if results["success"] else "😞"

    print(f"\n  {success_emoji} Attack {'SUCCEEDED' if results['success'] else 'FAILED'}")
    print(f"  ┌──────────────────────────────────────────────────────┐")
    print(f"  │ Best score:       {best['best_score']:.4f}")
    print(f"  │ Feasible:         {best['best_is_feasible']}")
    print(f"  │ Target choice:    {chr(65 + results['target_choice_index'])}")
    print(f"  │ Ground truth:     {chr(65 + results['ground_truth_idx'])}")
    print(f"  ├──────────────────────────────────────────────────────┤")
    print(f"  │ Original:         \"{cur_task_dict['question'][:45]}...\"")
    print(f"  │ Adversarial:      \"{best['best_decoded_text'][:45]}...\"")
    print(f"  └──────────────────────────────────────────────────────┘")

    # Save results to JSON
    output_path = os.path.join(
        os.path.dirname(__file__),
        f"results_{args.model_type}_{args.mmlu_subject}_{args.mmlu_question_idx}.json",
    )
    save_data = {
        "args": vars(args) if hasattr(args, "__dict__") else str(args),
        "success": results["success"],
        "best_score": best["best_score"],
        "best_decoded_text": best["best_decoded_text"],
        "best_is_feasible": best["best_is_feasible"],
        "original_question": cur_task_dict["question"],
        "ground_truth": chr(65 + results["ground_truth_idx"]),
        "target_choice": chr(65 + results["target_choice_index"]),
    }
    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)

    print(f"\n  💾 Results saved to: {output_path}")


if __name__ == "__main__":
    main()
