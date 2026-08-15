"""
🤖 Model Loading Utilities
===========================

Two kinds of models in REALISTA:

1. **Target LLMs** (open-source, run locally)
   - These are the models we're trying to trick into hallucinating.
   - Loaded via HuggingFace in fp16 with all weights frozen.
   - We only need forward passes + gradient through embeddings.

2. **API Models** (OpenAI, accessed remotely)
   - Used as reasoning targets (GPT-5-Nano/Mini)
   - Used as judges (feasibility checker, hallucination evaluator)
   - Wrapped in the `GPT` class with automatic retry on failures.

               ┌─────────────────────────────────┐
               │   load_model_and_tokenizer()     │
               │   Downloads from HuggingFace     │
               │   Loads in fp16 → freezes all    │
               │   parameters → ready for attack  │
               └─────────────────────────────────┘

               ┌─────────────────────────────────┐
               │   GPT (wrapper class)            │
               │   Sends chat messages → gets     │
               │   responses. Retries on errors.  │
               └─────────────────────────────────┘
"""
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import openai

from src.config import OPENAI_API_KEY, MODEL_REGISTRY


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Local LLM Loader
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_model_and_tokenizer(model_type: str):
    """Download (if needed) and load a target LLM, ready for attack.

    What this does step by step:
      1. Looks up the HuggingFace model path from our registry.
      2. Downloads the tokenizer (the "dictionary" that converts text → numbers).
      3. Downloads the model weights in half-precision (fp16) to save VRAM.
      4. Puts the model in eval mode and freezes ALL parameters.

    Why freeze everything?
      We never update the model's weights during an attack.
      We only need gradients flowing *through* the model (to compute
      how changing δ affects the output probabilities), not gradients
      *of* the model parameters. Freezing saves memory and prevents
      accidental weight corruption.

    Parameters
    ----------
    model_type : str
        One of: "llama3_3b", "llama3_8b", "qwen2_5_7b", "qwen2_5_14b"

    Returns
    -------
    model : AutoModelForCausalLM  — the frozen LLM
    tokenizer : AutoTokenizer     — its tokenizer
    """
    # Step 1: Look up the HuggingFace path
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model type: {model_type!r}. "
            f"Choose from: {list(MODEL_REGISTRY)}"
        )
    model_path = MODEL_REGISTRY[model_type]
    print(f"🔄 Loading target LLM: {model_type} ({model_path})")

    # Step 2: Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, low_cpu_mem_usage=True,
    )

    # Step 3: Load the model in half-precision
    #   - `torch_dtype=float16` halves VRAM usage vs. float32
    #   - `device_map="auto"` intelligently splits across available GPUs
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )

    # Step 4: Freeze everything — no gradient updates to model weights
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    print(f"✅ Model loaded and frozen. Ready for attack.")
    return model, tokenizer


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  OpenAI API Wrapper
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class GPT:
    """A simple, retry-safe wrapper around OpenAI's chat API.

    Used in two roles:
      1. **As a reasoning target** — we send adversarial prompts to
         GPT-5-Nano/Mini and check if they hallucinate.
      2. **As a judge** — we ask GPT-4.1 to evaluate whether:
         - The adversarial prompt is semantically equivalent (feasibility)
         - The LLM's response contains hallucinations (scoring)

    Why wrap the API?
      Transient errors (rate limits, timeouts, server hiccups) are common
      when making hundreds of API calls during an attack. This wrapper
      automatically retries up to 5 times with a 10-second pause between
      attempts, so a single glitch doesn't crash the whole experiment.

    Usage:
        judge = GPT("gpt-4.1-2025-04-14")
        response = judge.generate([
            {"role": "user", "content": "Is this a hallucination?"}
        ])
    """

    # How long to wait between retries (seconds)
    API_RETRY_SLEEP = 10

    # What to return if all retries fail
    API_ERROR_OUTPUT = "$ERROR$"

    # Maximum number of retry attempts
    API_MAX_RETRY = 5

    def __init__(self, model_name: str, api_key: str | None = None):
        self.model_name = model_name
        self.client = openai.OpenAI(api_key=api_key or OPENAI_API_KEY)

    def generate(
        self,
        messages: list[dict],
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str:
        """Send a chat message and get the assistant's response.

        Parameters
        ----------
        messages : list[dict]
            Standard OpenAI chat format:
            [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
        max_new_tokens : int
            Cap on response length.
        temperature : float
            0.0 = deterministic, higher = more creative/random.

        Returns
        -------
        str — the assistant's response text, or "$ERROR$" if all retries fail.
        """
        for attempt in range(1, self.API_MAX_RETRY + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                )
                return response.choices[0].message.content.strip()

            except openai.APIError as exc:
                print(f"  ⚠️ API error (attempt {attempt}/{self.API_MAX_RETRY}): {exc}")
                time.sleep(self.API_RETRY_SLEEP)

            except Exception as exc:
                print(f"  ⚠️ Unexpected error (attempt {attempt}): {exc}")
                time.sleep(self.API_RETRY_SLEEP)

        # All retries exhausted
        print(f"  ❌ All {self.API_MAX_RETRY} attempts failed.")
        return self.API_ERROR_OUTPUT
