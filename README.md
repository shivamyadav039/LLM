# ⚔️ REALISTA: Realistic Latent Adversarial Attacks that Elicit LLM Hallucinations

> **PaperBench-style end-to-end replication** of the ICML 2026 paper by Buyun Liang et al.
>
> 📄 Paper: [arXiv:2605.12813](https://arxiv.org/abs/2605.12813) · 💻 Original code: [Buyun-Liang/REALISTA](https://github.com/Buyun-Liang/REALISTA)

---

## 🧠 What is REALISTA?

Existing methods for tricking LLMs into hallucinating face a dilemma:
- **Discrete prompt attacks** (rephrasing the question) are readable but weak — they only explore a small set of variations.
- **Continuous latent attacks** (optimizing in embedding space) are powerful but produce gibberish prompts that change the question's meaning.

**REALISTA bridges both worlds** — it optimizes in continuous latent space but always decodes back to fluent, human-readable questions that mean the same thing as the original.

### How It Works (3 Key Ideas)

```
  📚 Concept Dictionary        📐 Simplex Optimization      🔓 LLM Decoder
  ─────────────────────        ────────────────────────      ──────────────
  WordNet adjectives           δ = [0.6, 0.4, 0, 0, ...]    Gumbel-Softmax
  → "formal", "concise"       ‖δ‖₁ ≤ ε (stay close)        autoregressive
  → latent directions          δᵢ ≥ 0 (no inversions)       z → fluent text
```

1. **Edit Dictionary**: For each concept (e.g. "formal"), compute a latent direction that makes the question more formal.
2. **Simplex Optimization**: Find the best weighted combination of concept directions, constrained to be sparse and bounded.
3. **LLM Decoder**: Decode the perturbed latent back to readable text using the same LLM as encoder/decoder.

---

## 📊 Before vs. After Attack

```
  ┌─────────────────────────────────────────────────────────────────────┐
  │  BEFORE (Original Question)                                       │
  │  "Statement 1: Linear regression is supervised learning.          │
  │   Statement 2: Gradient descent guarantees global minimum."       │
  │                                                                    │
  │  LLM Answer: B ✅  (True, False)                                  │
  │  P(A)=5%  P(B)=82%  P(C)=3%  P(D)=10%                           │
  ├─────────────────────────────────────────────────────────────────────┤
  │  AFTER (Adversarial Question — same meaning!)                     │
  │  "Evaluate whether linear regression falls under supervised       │
  │   methodologies and if gradient-based optimization ensures        │
  │   convergence to the global optimum in all scenarios."            │
  │                                                                    │
  │  LLM Answer: A ❌  (True, True)                                   │
  │  P(A)=68%  P(B)=22%  P(C)=5%  P(D)=5%                           │
  └─────────────────────────────────────────────────────────────────────┘
```

The question means the **same thing**, but the LLM now picks the **wrong answer**!

---

## 🗂️ Project Structure

```
crossingLLM/
├── 🚀 run_demo.py                           # End-to-end runner (start here!)
├── 📦 pyproject.toml                        # Project metadata + dependencies
├── 📋 requirements.txt                      # Dependencies (pip fallback)
├── 🐳 Dockerfile                            # GPU-enabled container
├── 🐳 docker-compose.yml                    # One-command GPU orchestration
├── 🔒 .env.example                          # API key template
├── src/
│   ├── ⚔️ realista.py                       # Core: Stage 1 + Stage 2 PLD + judges
│   ├── 📝 qa_utils.py                       # Prompt templates + probability extraction
│   ├── 📖 dictionary_utils.py               # Load pre-computed dictionaries
│   ├── 🤖 model_utils.py                    # HuggingFace loader + OpenAI wrapper
│   ├── 🔧 config.py                         # Model registry, paths, constants
│   ├── 🎛️ arguments.py                      # All hyperparameters in one place
│   ├── 🎲 utils.py                          # Seed fixing
│   └── optional_dict_construction/
│       └── 🏗️ dict_construction_utils.py    # Build dictionaries from scratch
└── data/
    └── rephrasing_prompts/                   # Pre-computed rephrasings (download)
```

---

## ⚡ Quick Start

### Option A: pip install

```bash
# 1. Install via pyproject.toml (recommended)
pip install -e .

# 2. Or install via requirements.txt
pip install -r requirements.txt

# 3. Set up OpenAI API key (needed for LLM judges)
cp .env.example .env   # then edit .env with your key

# 4. Run a quick test (smallest model, fast settings)
python run_demo.py --model_type llama3_3b --trial_num 1 --pld_iterations 10

# 5. Full attack (default settings from the paper)
python run_demo.py --model_type llama3_8b --mmlu_subject machine_learning --trial_num 10

# 6. Attack a reasoning model (requires OpenAI API)
python run_demo.py --model_type llama3_8b --reasoning_target gpt_5_nano
```

### Option B: Docker (recommended for GPU machines)

```bash
# 1. Build the image
docker build -t realista .

# 2. Quick smoke test (smallest model)
docker run --gpus all -it realista

# 3. Full attack with persistent model cache + OpenAI judges
docker run --gpus all \
  --env-file .env \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -it realista \
  --model_type llama3_8b --trial_num 10

# 4. Or use Docker Compose (handles GPU + cache + .env automatically)
docker compose up                    # default quick test
docker compose run attack \          # custom args
  --model_type llama3_8b --trial_num 10

# 5. Interactive shell inside the container
docker compose run attack bash
```

> **Note:** Docker GPU access requires [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed on the host.

---

## 🎛️ Hyperparameter Guide

| Parameter | Default | What it does |
|-----------|---------|-------------|
| `--model_type` | `llama3_8b` | Which LLM to attack |
| `--mmlu_subject` | `machine_learning` | MMLU topic |
| `--trial_num` | `10` | PLD trials (more = better chance, slower) |
| `--pld_iterations` | `50` | Steps per trial |
| `--attack_budget` | `1.0` | ε: how far we can perturb (larger = more aggressive) |
| `--pld_step_size` | `0.05` | η: gradient step size |
| `--pld_temperature_init` | `1.0` | T₀: initial noise level |
| `--pld_temperature_decay` | `0.95` | γ: how fast noise decreases |
| `--reasoning_target` | `none` | Set to `gpt_5_nano` or `gpt_5_mini` for API attacks |

---

## 🔬 Algorithm Deep Dive

### Stage 1: Single-Concept Initialization
```
For each concept i in the edit dictionary:
  1. Set δ = ε · eᵢ (activate ONLY concept i at full budget)
  2. Perturb: z = z₀ + ε · z^(i)
  3. Decode z → adversarial prompt text
  4. Score: log P(wrong_answer | adversarial_prompt)
  5. Keep the top-N highest-scoring concepts
```

### Stage 2: Projected Langevin Dynamics
```
Starting from a Stage 1 winner:
  For k = 1, 2, ..., K:
    1. z = z₀ + Σ δᵢ · z^(i)          ← apply edit directions
    2. x = ψ(z)                        ← decode (Gumbel-Softmax)
    3. L = log P(wrong | x)            ← attack objective
    4. Check: is x ≈ x₀ semantically? ← LLM judge
    5. If yes: δ += η·∇L + noise       ← gradient ascent + exploration
       If no:  δ += noise only          ← escape infeasible region
    6. δ = Proj_{Δε}(δ)                ← project onto simplex
    7. T *= γ                           ← cool down (less noise over time)
```

### Key Mathematical Properties
- **Simplex constraint** Δ_ε = {δ ≥ 0 : ‖δ‖₁ ≤ ε}
  - Non-negativity prevents "inverting" concept directions (→ gibberish)
  - ℓ₁ bound limits total edit strength (→ semantic equivalence)
  - Sparsity: most δᵢ are zero (only a few concepts active)
- **Gumbel-Softmax**: differentiable token sampling for end-to-end backprop
- **KV-cache reuse**: O(prompt_len) decoding instead of O(prompt_len²)

---

## 💻 GPU Requirements

| Model | VRAM (fp16) | Recommended GPU |
|-------|-------------|-----------------|
| Llama-3.2-3B | ~7 GB | RTX 3080 / A10 |
| Llama-3.1-8B | ~16 GB | RTX 4090 / A100 |
| Qwen-2.5-7B | ~14 GB | RTX 4090 / A100 |
| Qwen-2.5-14B | ~28 GB | A100 40GB |

---

## 📚 References

```bibtex
@inproceedings{liang2026realista,
  title     = {REALISTA: Realistic Latent Adversarial Attacks
               that Elicit LLM Hallucinations},
  author    = {Liang, Buyun and Luo, Jinqi and Peng, Liangzu
               and Chan, Kwan Ho Ryan and Thaker, Darshan
               and Kinfu, Kaleab A. and Tian, Fengrui
               and Hassani, Hamed and Vidal, Ren{\'e}},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```
