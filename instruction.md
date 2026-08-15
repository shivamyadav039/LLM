# Task Description: Replicating REALISTA

You are tasked with reproducing the end-to-end implementation of the REALISTA framework as described in the accompanying paper (located at `/workspace/environment/paper/paper.pdf` or `/workspace/environment/paper/paper.md`).

REALISTA (Realistic Latent Adversarial Attacks that Elicit LLM Hallucinations) is a framework designed to elicit hallucinations from LLMs via realistic latent-space attacks. It bridges continuous optimization and discrete realism through three key components:
1. **Input-Dependent Edit Dictionary**: Constructing latent editing directions representing semantically equivalent transformations.
2. **Scaled Latent Simplex Optimization**: Projected Langevin Dynamics (PLD) on a scaled simplex constraint to find sparse latent perturbations.
3. **LLM-Based Decoding**: Gumbel-Softmax reparameterization for differentiable autoregressive decoding back to natural language.

---

## Task Contract & Deliverables

All deliverables **must** be placed under `/workspace/submission/` directory inside the container.

### Clear Deliverables:
1. **Core Attack Module** (`/workspace/submission/src/realista.py`):
   - Must implement Gumbel-Softmax autoregressive decoding from a latent hiding layer (`reconstruct_from_latent`).
   - Must implement simplex projection (`project_onto_simplex`) enforcing non-negativity and $\ell_1$ norm constraint.
   - Must implement Stage 1 initialization (`stage1_optimization`).
   - Must implement Stage 2 Projected Langevin Dynamics (`PLD`) with temperature annealing and semantic equivalence safeguards.
2. **Configuration & Argument Handling** (`/workspace/submission/src/config.py`, `/workspace/submission/src/arguments.py`):
   - Model registry, early residual-stream layer lookup (e.g. layer 3), and hyperparameters map.
3. **QA Utilities** (`/workspace/submission/src/qa_utils.py`):
   - Answer choice token ID mapping and probability extraction over answer choices (A, B, C, D) with gradients preserved.
4. **Runnable Entrypoint Script** (`/workspace/submission/run_demo.py`):
   - Command-line runner that parses arguments, loads an MMLU question, builds the latent edit dictionary, runs the two-stage optimization attack, and outputs the adversarial prompt and success rate.

Your implementation will be verified by running unit tests and evaluating attack results. Ensure your code is modular, robust, and correctly implements the mathematical details described in the paper.
