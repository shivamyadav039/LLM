"""
🏗️ Dictionary Construction — Building Edit Dictionaries from Scratch
=====================================================================

If you don't have pre-computed dictionaries for a particular MMLU question,
this module builds them from the ground up. Here's the pipeline:

  Step 1: WordNet Adjectives
    Pull ~29,000 adjective concepts from WordNet (e.g. "concise", "formal",
    "indirect", "passive", "elaborate").

  Step 2: Embed & Score
    Embed both the concepts and the question using Qwen3-Embedding-8B.
    Score each concept on:
      - Relevance: cosine similarity to the question
      - Editability: "Can this concept guide a meaningful rewrite?" (1-5)

  Step 3: Select Best Concepts
    Greedy diverse selection: pick concepts that are relevant, editable,
    AND different from each other. We don't want 20 near-synonyms.

  Step 4: Generate Rephrasings
    For each selected concept, ask an LLM:
      "Rewrite this question in a more [concept] style."
    → Get 5 rephrasings per concept.

  Step 5: Compute Latent Directions
    For each rephrasing:
      z^(i) = φ(rephrasing_i) − φ(original_question)
    → Each direction = "how to push the latent to get that style."

     📚 WordNet ──→ 🔢 Embed ──→ 🎯 Select ──→ ✍️ Rephrase ──→ 🧮 Latent Dirs
     ~29k adj.      + score       top-n          via LLM         z^(i) = φ(r) - φ(x₀)
"""
import json
import os
import pickle
import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F

from src.config import LAYER_NUM_REGISTRY
from src.dictionary_utils import get_original_latent


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Editability Scoring Prompt
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# This prompt is sent to an LLM to judge: "Can this concept guide a
# meaningful question rewrite while preserving the original meaning?"
#
# Key insight from the paper: not all concepts are useful for editing.
#   - "chemisorptive" → Score 1 (too domain-specific, can't guide a rewrite)
#   - "concise" → Score 5 (excellent editing operator: "make it shorter")
#
# We filter out low-editability concepts to avoid wasting compute on
# directions that would produce gibberish or trivial rewrites.

EDITABILITY_INSTRUCTION = '''You are an expert evaluator of semantically equivalent prompt rewriting.

Your task is to judge the editability of a concept. Editability measures how suitable a concept is as an editing instruction that can guide a language model to rewrite a prompt while preserving its original meaning.

We define editability as follows:

A concept is considered editable if, when used as an editing instruction, it can reliably guide a language model to produce a rewritten prompt that:
(1) preserves the original intent and correct answer,
(2) remains coherent, grammatical, and natural, and
(3) meaningfully changes the surface form (i.e., it is not a trivial copy or minor wording change).

Important clarifications:
- Concepts that describe topical content or domain-specific attributes (e.g., medical terms, scientific descriptors, historical periods) are generally NOT good editing concepts.
- Concepts that describe linguistic, logical, or structural transformations (e.g., negation, contrastive framing, indirect questioning, counterfactual reasoning) are generally GOOD editing concepts.
- Relevance to the topic does NOT imply editability.
- Your judgment should focus only on whether the concept can function as a reliable rewrite operator.

You are given Concept: {concept}

Task:
Judge how suitable this concept is as an editing instruction for producing a semantically equivalent rewrite of the original prompt.

Scoring rubric (1-5):
- 1: Not editable at all. The concept is purely a content/topic descriptor and does not provide a meaningful rewrite operation.
- 2: Weakly editable. The concept is vague or unreliable and rarely leads to valid semantic-preserving rewrites.
- 3: Moderately editable. The concept can sometimes guide rewriting, but often fails to preserve intent or coherence.
- 4: Highly editable. The concept clearly functions as a rewrite operator and usually preserves meaning.
- 5: Excellent editability. The concept is a strong, reliable editing operator that consistently induces non-trivial, semantically equivalent rewrites.

Examples:

Concept: chemisorptive -> Score: 1
Concept: abaxial -> Score: 1
Concept: busy -> Score: 2
Concept: new -> Score: 2
Concept: accommodating -> Score: 3
Concept: accurate -> Score: 3
Concept: passive -> Score: 4
Concept: accessible -> Score: 4
Concept: abridged -> Score: 5
Concept: concrete -> Score: 5

Your output should be strictly an integer between 1 and 5, which is the score for the concept. DO NOT print anything else such as "Here are ...", "Sure, ...", "Certainly, ...". JUST RETURN ME THE SCORE.'''


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Step 1: WordNet Concept Pool
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_wordnet_adjective_concepts(pool_size: int | None = None, seed: int = 0):
    """Pull all unique adjective lemma names from WordNet.

    The paper uses the full pool (~29,000 adjectives). For quick demos,
    you can pass `pool_size=200` to randomly subsample a smaller set.

    Why adjectives?
      Adjectives describe properties and styles (e.g. "formal", "concise",
      "indirect"). These naturally map to ways you could rephrase a
      question while keeping the same meaning — which is exactly what
      we need for semantically equivalent edits.

    Parameters
    ----------
    pool_size : int or None — subsample to this many (None = all ~29k)
    seed      : int         — for reproducible subsampling

    Returns
    -------
    list[str] — concept names like ["formal", "concise", "indirect", ...]
    """
    import nltk
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    from nltk.corpus import wordnet as wn

    concepts = []
    seen = set()

    # WordNet has two adjective POS tags: 'a' (main) and 's' (satellite)
    for synset in list(wn.all_synsets("a")) + list(wn.all_synsets("s")):
        lemma = synset.lemma_names()[0].replace("_", " ")
        if lemma not in seen:
            seen.add(lemma)
            concepts.append(lemma)

    if pool_size is not None and pool_size < len(concepts):
        concepts = random.Random(seed).sample(concepts, pool_size)

    return concepts


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Step 2a: Embedding Model
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_embedding_model(
    model_name: str = "Qwen/Qwen3-Embedding-8B",
    device: str = "cuda",
):
    """Load the embedding model used to measure concept-question similarity.

    The paper uses Qwen3-Embedding-8B for computing:
      - Relevance scores (cosine similarity between concept & question)
      - Diversity scores (cosine similarity between selected concepts)

    Loaded in fp16 to keep VRAM manageable alongside the target LLM.
    """
    from sentence_transformers import SentenceTransformer

    if device == "cuda":
        return SentenceTransformer(
            model_name,
            device="cuda:0",
            model_kwargs={"torch_dtype": torch.float16},
        )
    return SentenceTransformer(model_name, device=device)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Step 3: Concept Selection (Diversity + Relevance + Editability)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def select_concepts(
    question_embedding: np.ndarray,
    concept_embeddings: np.ndarray,
    concepts: list[str],
    editability_scores: np.ndarray,
    n_select: int = 20,
    relevance_floor: float = 0.3,
    editability_floor: int = 3,
    diversity_weight: float = 0.5,
) -> list[str]:
    """Pick the best concepts for building an edit dictionary.

    The goal: find concepts that are:
      ✅ Relevant to the question (cosine similarity ≥ 0.3)
      ✅ Actually editable (editability score ≥ 3)
      ✅ Diverse from each other (we don't want 20 synonyms!)

    Algorithm (greedy):
      1. Filter out concepts below the relevance and editability floors.
      2. Start with the MOST relevant concept.
      3. Iteratively add the concept that maximizes:
         score = relevance − diversity_weight × similarity_to_nearest_selected
      4. Stop when we have n_select concepts.

    This balances "pick things relevant to the question" with
    "pick things different from what we already have."
    """
    # Normalize embeddings for cosine similarity
    q_normalized = question_embedding / (np.linalg.norm(question_embedding) + 1e-12)
    c_normalized = concept_embeddings / (
        np.linalg.norm(concept_embeddings, axis=1, keepdims=True) + 1e-12
    )

    # Relevance = cosine similarity between question and each concept
    relevance_scores = c_normalized @ q_normalized  # [N]

    # Apply both filters
    passes_filter = (
        (relevance_scores >= relevance_floor) &
        (editability_scores >= editability_floor)
    )
    valid_indices = np.where(passes_filter)[0]

    # Fallback: if nothing passes both, relax the relevance requirement
    if len(valid_indices) == 0:
        print("  ⚠️ No concepts pass both floors — relaxing relevance filter.")
        valid_indices = np.where(editability_scores >= editability_floor)[0]

    # If we have fewer valid concepts than requested, just return them all
    if len(valid_indices) <= n_select:
        return [concepts[i] for i in valid_indices]

    # ── Greedy diverse selection ──────────────────────────────────────
    selected = []
    remaining = set(valid_indices.tolist())

    # Seed the selection with the single most relevant concept
    first_idx = valid_indices[np.argmax(relevance_scores[valid_indices])]
    selected.append(first_idx)
    remaining.discard(first_idx)

    # Greedily add concepts that are relevant BUT different from selection
    for _ in range(n_select - 1):
        if not remaining:
            break

        best_score = -float("inf")
        best_idx = None

        for candidate in remaining:
            # How similar is this candidate to the MOST similar selected concept?
            max_sim_to_selected = max(
                float(c_normalized[candidate] @ c_normalized[s])
                for s in selected
            )

            # We want: high relevance, low similarity to existing selection
            combined_score = (
                float(relevance_scores[candidate])
                - diversity_weight * max_sim_to_selected
            )

            if combined_score > best_score:
                best_score = combined_score
                best_idx = candidate

        if best_idx is not None:
            selected.append(best_idx)
            remaining.discard(best_idx)

    return [concepts[i] for i in selected]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Step 4: Rephrasing Generation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_concept_rephrasings(
    question_text: str,
    concept: str,
    choices: list[str],
    subject: str,
    llm_generator,
    n_rephrasings: int = 5,
) -> list[str]:
    """Ask an LLM to rephrase a question using a specific concept.

    The prompt tells the LLM:
      "Rewrite this question using the concept 'formal' as a guide.
       Keep the same meaning and correct answer."

    We get back rephrasings like:
      Original: "What is supervised learning?"
      Concept "formal": "In the context of machine learning paradigms,
         which of the following best characterizes supervised learning?"

    Parameters
    ----------
    concept : str — the editing concept (e.g. "concise", "formal")
    llm_generator  — anything with a .generate(messages, ...) method

    Returns
    -------
    list[str] — the rephrasings (may be fewer than requested on parse failure)
    """
    choice_block = "\n".join(
        f"{chr(65 + i)}. {c}" for i, c in enumerate(choices)
    )

    prompt = f"""You are an expert in {subject.replace('_', ' ')}.

Rewrite the following multiple-choice question using the concept "{concept}" as a guiding style or transformation. The rewrite must:
1. Preserve the original meaning and correct answer.
2. Be grammatically correct and fluent.
3. Meaningfully change the surface form (not a trivial copy).

Original question:
{question_text}

Answer choices:
{choice_block}

Provide exactly {n_rephrasings} rewritten versions of ONLY the question (not the choices). Return them as a JSON list of strings.
Example output: ["rewrite 1", "rewrite 2", ...]"""

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    raw_response = llm_generator.generate(messages, max_new_tokens=1024, temperature=0.7)

    # Parse the JSON array from the LLM's response
    try:
        import re
        match = re.search(r"\[.*\]", raw_response, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception as e:
        print(f"  ⚠️ Failed to parse rephrasings: {e}")

    return []


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Step 5: Latent Direction Computation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_latent_directions(
    model,
    tokenizer,
    question_text: str,
    rephrasings: list[str],
    model_type: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute latent editing directions: z^(i) = φ(rephrasing_i) − φ(original).

    In plain English: for each rephrasing, we measure "which direction did
    the latent representation move?" These directions become the columns of
    our edit dictionary D(z₀).

    During the attack, we'll combine these directions with weights δ:
      z = z₀ + δ₁·z^(1) + δ₂·z^(2) + ... + δₙ·z^(n)

    The simplex constraint on δ ensures we stay close to the original.

    Note: rephrasings may have different token lengths than the original.
    We pad/truncate to match z₀'s sequence length.

    Parameters
    ----------
    question_text : str        — the original question
    rephrasings   : list[str]  — semantically equivalent rephrasings

    Returns
    -------
    directions : Tensor [n_rephrasings, L, d_model]  — editing directions
    z0         : Tensor [1, L, d_model]               — original latent
    """
    # Get the original question's latent representation
    z0 = get_original_latent(model, tokenizer, question_text, model_type)
    original_seq_len = z0.shape[1]

    directions = []
    for rephrasing in rephrasings:
        # Get the rephrasing's latent
        z_rephrased = get_original_latent(model, tokenizer, rephrasing, model_type)

        # Handle length mismatch: pad short rephrasings, truncate long ones
        rephrased_len = z_rephrased.shape[1]
        if rephrased_len < original_seq_len:
            # Pad with zeros (neutral — won't push the latent in any direction)
            padding = torch.zeros(
                1, original_seq_len - rephrased_len, z_rephrased.shape[2],
                device=z_rephrased.device, dtype=z_rephrased.dtype,
            )
            z_rephrased = torch.cat([z_rephrased, padding], dim=1)
        elif rephrased_len > original_seq_len:
            # Truncate to match (keep the first L tokens)
            z_rephrased = z_rephrased[:, :original_seq_len, :]

        # The direction = where the rephrasing "lives" relative to the original
        direction = z_rephrased - z0  # [1, L, d_model]
        directions.append(direction.squeeze(0))  # [L, d_model]

    if directions:
        # Stack into a single tensor: [n_rephrasings, L, d_model]
        return torch.stack(directions, dim=0), z0
    else:
        return torch.empty(0), z0
