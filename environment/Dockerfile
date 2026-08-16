# ── Base: Python 3.11 slim ────────────────────────────────────────
# Using the official slim Python image for fast, reliable builds on Modal/Harbor.
# GPU drivers are injected at runtime by the orchestrator (Modal, Harbor, etc.).
FROM python:3.11-slim

# Don't buffer Python stdout/stderr
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install basic OS utilities
RUN apt-get update && \
    apt-get install -y --no-install-recommends git wget curl && \
    rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU (sufficient for verifier tests; GPU is injected at runtime)
# and all REALISTA dependencies pinned with ==
RUN pip install --no-cache-dir \
    torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
    transformers==4.43.3 \
    sentence-transformers==3.0.1 \
    datasets==2.20.0 \
    numpy==1.26.4 \
    tqdm==4.66.4 \
    nltk==3.8.1 \
    openai==1.37.1 \
    python-dotenv==1.0.1 \
    zstandard==0.22.0 \
    accelerate==0.33.0 \
    pypdf==4.3.1 \
    pytest==8.2.2

# Pre-download NLTK data (needed for WordNet adjective concept pool extraction)
RUN python -c "import nltk; nltk.download('wordnet', quiet=True); nltk.download('omw-1.4', quiet=True)"

# Set up working directory
WORKDIR /workspace
