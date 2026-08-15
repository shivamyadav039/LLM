"""
📝 QA Utilities — Prompt Building & Probability Extraction
===========================================================

This module handles the "interface" between REALISTA and the target LLM:

  1. **Token ID Mapping**
     When the LLM outputs "A", "B", "C", or "D", each letter has a specific
     numeric ID in the tokenizer's vocabulary. These IDs differ between Llama
     and Qwen tokenizers, so we hard-code them per model family.

  2. **Probability Extraction**
     After a forward pass, we grab the logits for just those 4 token IDs,
     apply softmax, and get P(A), P(B), P(C), P(D). This is the attack
     objective: we want to MAXIMIZE P(wrong_answer).

  3. **Prompt Templates**
     We wrap the question in a standard MMLU-style template. The key insight
     is that the question itself exists as a latent embedding — the prefix
     and suffix are plain text embeddings that we concatenate around it.

     ┌──────────────┬─────────────────────┬──────────────────────┐
     │  PREFIX      │  QUESTION LATENT    │  SUFFIX              │
     │  "The fol-   │  z₀ (or z₀ + Dδ   │  "A. ... B. ...     │
     │   lowing..." │   if perturbed)     │   Answer:"           │
     └──────────────┴─────────────────────┴──────────────────────┘
"""
import torch

from src.config import LAYER_NUM_REGISTRY


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Token IDs for answer choices A/B/C/D
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Each LLM's tokenizer assigns a different integer ID to the letters
# A, B, C, D when they appear as the FIRST generated token.
#
# How to find these: tokenizer.encode("A") → [362] (for Llama)
#
# We hard-code them because looking them up at runtime adds complexity
# and they never change for a given model.

CHOICE_TOKEN_IDS = {
    "llama3_8b":   {"A": 362, "B": 426, "C": 356, "D": 423},
    "llama3_3b":   {"A": 362, "B": 426, "C": 356, "D": 423},   # same tokenizer as 8B
    "qwen2_5_7b":  {"A": 362, "B": 425, "C": 356, "D": 422},   # slightly different IDs
    "qwen2_5_14b": {"A": 362, "B": 425, "C": 356, "D": 422},   # same tokenizer as 7B
}


def _choice_token_ids(model_type: str) -> dict:
    """Look up the A/B/C/D token IDs for a given model."""
    if model_type not in CHOICE_TOKEN_IDS:
        raise ValueError(
            f"No token-ID mapping for model: {model_type!r}. "
            f"Supported: {list(CHOICE_TOKEN_IDS)}"
        )
    return CHOICE_TOKEN_IDS[model_type]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Probability Extraction
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_probs(args, outputs):
    """Extract P(A), P(B), P(C), P(D) from a model's forward pass.

    This is the core of the attack objective. After the model processes
    the full prompt (prefix + question + suffix ending in "Answer:"),
    its last-position logits tell us how likely each answer letter is.

    We pick just the 4 logits for A/B/C/D, apply softmax, and return
    a 4-element probability vector — WITH gradients attached so we can
    backpropagate through this to update δ.

    Returns
    -------
    probs : Tensor [4]  — [P(A), P(B), P(C), P(D)], sums to 1
    """
    token_ids_map = _choice_token_ids(args.model_type)

    # Grab logits at the last position (the "next token" prediction)
    last_position_logits = outputs.logits[0, -1, :]  # shape: [vocab_size]

    # Pick out just the 4 logits for A, B, C, D
    choice_ids = torch.tensor(
        list(token_ids_map.values()), device=last_position_logits.device
    )
    abcd_logits = last_position_logits[choice_ids]  # shape: [4]

    # Softmax → proper probabilities that sum to 1
    return torch.softmax(abcd_logits, dim=0)


def get_probs_batch(args, outputs):
    """Same as get_probs, but for a batch of inputs.

    Returns
    -------
    probs : Tensor [batch_size, 4]
    """
    token_ids_map = _choice_token_ids(args.model_type)
    last_position_logits = outputs.logits[:, -1, :]  # [B, vocab_size]
    choice_ids = torch.tensor(
        list(token_ids_map.values()), device=last_position_logits.device
    )
    abcd_logits = last_position_logits[:, choice_ids]  # [B, 4]
    return torch.softmax(abcd_logits, dim=-1)


def format_probs(probs) -> str:
    """Pretty-print a probability distribution over A/B/C/D.

    Example output: "A: 12.34%  B: 56.78%  C:  3.45%  D: 27.43%"
    """
    return "  ".join(
        f"{letter}: {p * 100:5.2f}%"
        for letter, p in zip(["A", "B", "C", "D"], probs)
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Prompt Template Construction
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_prompt(cur_task_dict: dict, is_reasoning: bool = False):
    """Build the text that goes BEFORE and AFTER the question's latent.

    The MMLU prompt template looks like:

      PREFIX: "The following is a multiple choice question about [subject]."
      [QUESTION LATENT GOES HERE]
      SUFFIX: "A. ... B. ... C. ... D. ... Answer:"

    For reasoning models (free-form response), the suffix changes to
    ask for step-by-step reasoning instead of just a letter.

    Parameters
    ----------
    cur_task_dict : dict  — must have "subject" and "choices"
    is_reasoning  : bool  — True for free-form, False for MCQA

    Returns
    -------
    prefix_text : str  — goes BEFORE the question embedding
    suffix_text : str  — goes AFTER the question embedding
    """
    subject = cur_task_dict["subject"].replace("_", " ")
    choices = cur_task_dict["choices"]

    # Format the answer options as "A. ...\nB. ...\nC. ...\nD. ..."
    choice_block = "\n".join(
        f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices)
    )

    if is_reasoning:
        # Free-form: let the model think step-by-step (no forced letter)
        prefix_text = (
            f"The following is a multiple choice question about {subject}.\n\n"
        )
        suffix_text = (
            f"\n{choice_block}\n\n"
            "Please think step by step and provide your answer with explanation."
        )
    else:
        # Standard MCQA: force the model to start with a letter
        prefix_text = (
            f"The following is a multiple choice question about {subject}.\n\n"
        )
        suffix_text = (
            f"\n{choice_block}\n\n"
            "Answer:"
        )

    return prefix_text, suffix_text


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Full Input Embedding Assembly
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_full_input_embeds(
    model, tokenizer, prefix_text: str, suffix_text: str,
    question_latent, device=None,
):
    """Stitch together: [prefix_embedding | question_latent | suffix_embedding]

    This is how we "inject" the (possibly perturbed) question latent into
    the model's input. Instead of feeding text tokens, we directly feed
    embedding vectors — the prefix and suffix are normal text embeddings,
    but the middle part is our manipulated latent representation.

    This is what makes the whole attack differentiable: we can compute
    gradients of the output w.r.t. the question latent.

    Parameters
    ----------
    question_latent : Tensor [1, L_question, d_model]
        The latent representation of the question (possibly perturbed by Dδ).

    Returns
    -------
    full_embeds : Tensor [1, L_total, d_model]
    """
    if device is None:
        device = model.device

    # Convert prefix and suffix text → token IDs → embeddings
    prefix_ids = tokenizer(
        prefix_text, return_tensors="pt", add_special_tokens=False,
    ).input_ids.to(device)
    suffix_ids = tokenizer(
        suffix_text, return_tensors="pt", add_special_tokens=False,
    ).input_ids.to(device)

    embed_layer = model.model.embed_tokens
    prefix_embeds = embed_layer(prefix_ids)   # [1, L_prefix, d_model]
    suffix_embeds = embed_layer(suffix_ids)   # [1, L_suffix, d_model]

    # Make sure the question latent matches device and dtype
    question_latent = question_latent.to(
        device=device, dtype=prefix_embeds.dtype,
    )

    # Concatenate: [prefix | question | suffix]
    return torch.cat([prefix_embeds, question_latent, suffix_embeds], dim=1)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Free-Form Response Generation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_full_response(
    model, tokenizer, input_text: str, max_new_tokens: int = 512,
) -> str:
    """Generate a complete free-form text response from the target LLM.

    Used in the "reasoning model" evaluation setting, where instead of
    just predicting A/B/C/D, the model writes a full explanation.
    We then check this explanation for hallucinations.

    Note: This uses greedy decoding (no sampling) for deterministic results.
    """
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy = deterministic
        )

    # Only decode the NEW tokens (skip the input prompt)
    new_tokens = output_ids[0, inputs.input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)
