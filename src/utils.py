"""
🎲 Reproducibility Helpers
==========================

One tiny function, but it's crucial: fixing all random number
generators so that the same seed always produces the same attack.
"""
import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Lock down ALL sources of randomness for reproducible experiments.

    This seeds:
      - Python's built-in `random` module
      - NumPy's random number generator
      - PyTorch CPU and all GPU random generators

    Why does this matter?
    Gumbel-Softmax decoding and Langevin noise injection both depend
    on random sampling. Without a fixed seed, you'd get different
    adversarial prompts every run, making debugging impossible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
