# ── Base: NVIDIA CUDA 12.4 + Python 3.11 ─────────────────────────
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

# Don't buffer Python stdout/stderr
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install basic development utilities
RUN apt-get update && apt-get install -y git wget curl && rm -rf /var/lib/apt/lists/*

# Install python dependencies required by REALISTA
RUN pip install --no-cache-dir \
    transformers>=4.40.0 \
    sentence-transformers \
    datasets \
    numpy>=1.24.0 \
    tqdm>=4.65.0 \
    nltk \
    openai>=1.0.0 \
    python-dotenv \
    zstandard \
    accelerate>=0.26.0 \
    pypdf

# Pre-download NLTK data (needed for WordNet adjective concept pool extraction)
RUN python -c "import nltk; nltk.download('wordnet', quiet=True); nltk.download('omw-1.4', quiet=True)"

# Set up working directory inside the container (standard SWE/Harbor workspaces mount here)
WORKDIR /workspace
