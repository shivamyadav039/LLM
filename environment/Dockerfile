# ── Base: NVIDIA CUDA 12.4 + Python 3.11 ─────────────────────────
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

# Don't buffer Python stdout/stderr
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install basic development utilities
RUN apt-get update && apt-get install -y git wget curl && rm -rf /var/lib/apt/lists/*

# Install python dependencies required by REALISTA (pinned with == for integrity)
RUN pip install --no-cache-dir \
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

# Stage environment/paper/ into /workspace/paper/
COPY environment/paper/ /workspace/paper/

# Set up working directory inside the container (standard SWE/Harbor workspaces mount here)
WORKDIR /workspace
