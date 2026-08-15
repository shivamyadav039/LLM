# ─────────────────────────────────────────────────────────────────
#  REALISTA — GPU-accelerated Docker image
# ─────────────────────────────────────────────────────────────────
#
#  This image runs the full REALISTA attack pipeline on an NVIDIA GPU.
#
#  Build:
#    docker build -t realista .
#
#  Run (requires nvidia-container-toolkit):
#    docker run --gpus all -it realista \
#      --model_type llama3_3b --trial_num 1 --pld_iterations 10
#
#  With OpenAI judges:
#    docker run --gpus all --env-file .env -it realista \
#      --model_type llama3_8b --trial_num 10
#
#  Interactive shell:
#    docker run --gpus all --env-file .env -it realista bash
# ─────────────────────────────────────────────────────────────────

# ── Base: NVIDIA CUDA 12.4 + Python 3.11 ─────────────────────────
# Using the official PyTorch base image with CUDA pre-installed.
# This avoids 30+ minutes of compiling CUDA from scratch.
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

# Don't buffer Python stdout/stderr (so logs appear in real-time)
ENV PYTHONUNBUFFERED=1
# Don't write .pyc files (saves disk in container)
ENV PYTHONDONTWRITEBYTECODE=1

# Where HuggingFace models get cached inside the container.
# Mount a volume here to persist downloads between runs:
#   docker run --gpus all -v ~/.cache/huggingface:/root/.cache/huggingface ...
ENV HF_HOME=/root/.cache/huggingface
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface

WORKDIR /app

# ── Install Python dependencies ──────────────────────────────────
# Copy just the dependency files first for better Docker layer caching.
# If requirements.txt hasn't changed, this layer is reused even
# when your code changes.
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir accelerate>=0.26.0

# ── Download NLTK data (needed for WordNet concepts) ──────────────
RUN python -c "import nltk; nltk.download('wordnet', quiet=True); nltk.download('omw-1.4', quiet=True)"

# ── Copy the full project ────────────────────────────────────────
COPY . .

# ── Health check: verify imports work ─────────────────────────────
RUN python -c "\
from src.config import MODEL_REGISTRY; \
from src.arguments import RealistaArgs; \
from src.realista import project_onto_simplex; \
print('✅ Container build verified')"

# ── Default command: run the attack ───────────────────────────────
# Override args with: docker run ... realista --model_type llama3_3b
ENTRYPOINT ["python", "run_demo.py"]

# Default: smallest model, 1 trial, 10 iterations (quick smoke test)
CMD ["--model_type", "llama3_3b", "--trial_num", "1", "--pld_iterations", "10"]
