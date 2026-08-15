# Task Description: Replicating REALISTA

You are tasked with reproducing the end-to-end implementation of the REALISTA framework as described in the accompanying paper (located at `/workspace/environment/paper/paper.pdf` or `/workspace/environment/paper/paper.md`).

## Agent-Facing Task Contract

Please note that all deliverables belong under /workspace/submission/. You must implement and place your code, models, and scripts in this directory.

Here is the list of clear deliverables you need to place under /workspace/submission/:

1. **Core Attack Module** (`/workspace/submission/src/realista.py`):
   - Must implement Gumbel-Softmax autoregressive decoding from a latent hiding layer (`reconstruct_from_latent`).
   - Must implement simplex projection (`project_onto_simplex`) enforcing non-negativity and l1 norm constraint.
   - Must implement Stage 1 & 2 Projected Langevin Dynamics (`PLD`) with temperature annealing and semantic equivalence safeguards.
2. **Configuration & Argument Handling** (`/workspace/submission/src/config.py`, `/workspace/submission/src/arguments.py`):
   - Model registry, early residual-stream layer lookup, and hyperparameters.
3. **QA Utilities** (`/workspace/submission/src/qa_utils.py`):
   - Answer choice token ID mapping and probability extraction over answer choices (A, B, C, D).
4. **Runnable Entrypoint Script** (`/workspace/submission/run_demo.py`):
   - Command-line runner that parses arguments, loads an MMLU question, builds the latent edit dictionary, runs the two-stage optimization attack, and outputs the adversarial prompt.

Your implementation will be verified by running unit tests and evaluating attack results. Ensure your code is modular, robust, and correctly implements the mathematical details described in the paper.
