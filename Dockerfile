# =============================================================================
# ACE-Step 1.5 FastAPI Server - Multi-stage Dockerfile
# =============================================================================
# This image includes the ACE-Step models (~15GB total)
# Build with: docker build --secret id=HF_TOKEN,env=HF_TOKEN -t acestep-api:latest .
#
# Build args:
#   LM_MODEL_SIZE - Language model size: "0.6B" (default) or "1.7B"
#     0.6B: smaller, fits 8GB VRAM GPUs (~13GB image)
#     1.7B: higher quality, needs more VRAM (~20GB image)
#   DIT_VARIANT - Diffusion transformer: "turbo" (default) or "base"
#     turbo: 8 diffusion steps, ~7s generation (~6x faster)
#     base:  50 diffusion steps, ~40s generation (higher quality)
# =============================================================================

# LM model size: "0.6B" or "1.7B"
ARG LM_MODEL_SIZE=0.6B
# DiT variant: "base" (default, 50 steps) or "turbo" (8 steps, faster)
ARG DIT_VARIANT=base

# -----------------------------------------------------------------------------
# Stage 1: Model Downloader - Download models from HuggingFace
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS model-downloader

ARG LM_MODEL_SIZE
ARG DIT_VARIANT

WORKDIR /models

# Install huggingface-hub with hf_transfer for faster downloads
RUN pip install --no-cache-dir "huggingface-hub[cli,hf_transfer]"

# Enable fast transfers
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# Download main model package (shared components: VAE, Qwen3-Embedding, text encoder)
# Exclude models we don't need based on LM_MODEL_SIZE and DIT_VARIANT
RUN --mount=type=secret,id=HF_TOKEN \
    IGNORE=""; \
    if [ "$DIT_VARIANT" != "base" ]; then IGNORE="$IGNORE acestep-v15-base/*"; fi; \
    if [ "$LM_MODEL_SIZE" != "1.7B" ]; then IGNORE="$IGNORE acestep-5Hz-lm-1.7B/*"; fi; \
    python -c "from huggingface_hub import snapshot_download; token=open('/run/secrets/HF_TOKEN').read().strip(); ignore=[p for p in '$IGNORE'.split() if p]; snapshot_download('ACE-Step/Ace-Step1.5', local_dir='/models/checkpoints', token=token, ignore_patterns=ignore)"

# Download turbo DiT model weights separately (only for turbo builds — it's a separate HF repo)
RUN --mount=type=secret,id=HF_TOKEN \
    if [ "$DIT_VARIANT" = "turbo" ]; then \
      python -c "from huggingface_hub import snapshot_download; token=open('/run/secrets/HF_TOKEN').read().strip(); snapshot_download('ACE-Step/acestep-v15-turbo', local_dir='/models/checkpoints/acestep-v15-turbo', token=token)"; \
    else \
      echo "Skipping turbo DiT download (building $DIT_VARIANT variant)"; \
    fi

# Download 0.6B LM separately (only for 0.6B builds — it's a separate HF repo)
RUN --mount=type=secret,id=HF_TOKEN \
    if [ "$LM_MODEL_SIZE" = "0.6B" ]; then \
      python -c "from huggingface_hub import snapshot_download; token=open('/run/secrets/HF_TOKEN').read().strip(); snapshot_download('ACE-Step/acestep-5Hz-lm-0.6B', local_dir='/models/checkpoints/acestep-5Hz-lm-0.6B', token=token)"; \
    else \
      echo "Skipping 0.6B LM download (building $LM_MODEL_SIZE variant)"; \
    fi

# Reorganize models into separate dirs so each becomes its own Docker layer.
# This makes image pulls resilient: each ~2-5GB layer is independently resumable,
# instead of one monolithic ~15GB layer that restarts from scratch on failure.
RUN mkdir -p /models/dit-model /models/lm-model /models/shared-models && \
    if [ -d /models/checkpoints/acestep-v15-base ]; then mv /models/checkpoints/acestep-v15-base /models/dit-model/; fi && \
    if [ -d /models/checkpoints/acestep-v15-turbo ]; then mv /models/checkpoints/acestep-v15-turbo /models/dit-model/; fi && \
    if [ -d /models/checkpoints/acestep-5Hz-lm-1.7B ]; then mv /models/checkpoints/acestep-5Hz-lm-1.7B /models/lm-model/; fi && \
    if [ -d /models/checkpoints/acestep-5Hz-lm-0.6B ]; then mv /models/checkpoints/acestep-5Hz-lm-0.6B /models/lm-model/; fi && \
    mv /models/checkpoints/* /models/shared-models/ 2>/dev/null || true

# -----------------------------------------------------------------------------
# Stage 2: Runtime - Install ACE-Step and run from /app
# -----------------------------------------------------------------------------
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04 AS runtime

ARG LM_MODEL_SIZE
ARG DIT_VARIANT

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # ACE-Step configuration
    ACESTEP_PROJECT_ROOT=/app \
    ACESTEP_OUTPUT_DIR=/app/outputs \
    ACESTEP_TMPDIR=/app/outputs \
    ACESTEP_DEVICE=cuda \
    # ACE-Step API model paths (full paths to pre-baked models)
    ACESTEP_CONFIG_PATH=/app/checkpoints/acestep-v15-${DIT_VARIANT} \
    ACESTEP_LM_MODEL_PATH=/app/checkpoints/acestep-5Hz-lm-${LM_MODEL_SIZE} \
    ACESTEP_LM_BACKEND=pt \
    # Server configuration
    ACESTEP_API_HOST=0.0.0.0 \
    ACESTEP_API_PORT=8000

WORKDIR /app

# Install system dependencies including Python, pip, git, and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    curl \
    build-essential \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

# Install uv for faster dependency resolution
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Clone ACE-Step directly into /app and install.
# Pinned to a specific upstream commit (rather than a moving `main`) so
# rebuilds are reproducible and staleness has to be an explicit bump here,
# not something that silently drifts until someone notices.
COPY patches/vae-float32.patch /tmp/vae-float32.patch
ARG ACESTEP_UPSTREAM_REF=ca1e85fe9430179831e6bc6be790c332190a3866
RUN git clone https://github.com/ace-step/ACE-Step-1.5.git /app && \
    cd /app && git checkout "$ACESTEP_UPSTREAM_REF" && cd / && \
    git -C /app apply --check /tmp/vae-float32.patch && \
    git -C /app apply /tmp/vae-float32.patch && \
    rm -rf /app/.git /tmp/vae-float32.patch && \
    uv pip install --system --no-cache /app

# Create symlink so ACE-Step's model discovery finds /app/checkpoints
# ACE-Step uses __file__ to locate checkpoints relative to its install path
RUN ln -s /app/checkpoints /usr/local/lib/python3.11/dist-packages/checkpoints

# Copy models as separate Docker layers for resilient image pulls.
# Each layer is ~2-5GB instead of one monolithic ~15GB layer, so containerd
# can resume individual layers on network failure instead of restarting everything.
COPY --from=model-downloader /models/shared-models/ /app/checkpoints/
COPY --from=model-downloader /models/dit-model/ /app/checkpoints/
COPY --from=model-downloader /models/lm-model/ /app/checkpoints/

# Create placeholder dirs for excluded models to satisfy check_main_model_exists()
RUN if [ "$DIT_VARIANT" != "base" ]; then mkdir -p /app/checkpoints/acestep-v15-base; fi && \
    if [ "$DIT_VARIANT" != "turbo" ]; then mkdir -p /app/checkpoints/acestep-v15-turbo; fi && \
    if [ "$LM_MODEL_SIZE" != "1.7B" ]; then mkdir -p /app/checkpoints/acestep-5Hz-lm-1.7B; fi && \
    if [ "$LM_MODEL_SIZE" != "0.6B" ]; then mkdir -p /app/checkpoints/acestep-5Hz-lm-0.6B; fi

# Create non-root user (UID 1001) and fix ownership
RUN groupadd -g 1001 appuser && \
    useradd -u 1001 -g 1001 -m -s /bin/bash appuser

# Copy startup script
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Fix ownership for all dirs ACE-Step writes to at runtime:
#   /app/outputs        — generated audio, logs
#   /app/checkpoints    — model checkpoints (symlinked into site-packages)
#   /home/appuser/.cache — PyTorch/HF cache
#   site-packages/      — ACE-Step creates gradio_outputs/, .cache/, and other
#                         dirs inside its install path at import time. Rather than
#                         chasing each one, make the entire dist-packages tree
#                         writable by appuser.
RUN mkdir -p /app/outputs /home/appuser/.cache && \
    chown -R 1001:1001 /app/outputs /app/checkpoints /home/appuser/.cache \
    /usr/local/lib/python3.11/dist-packages

# PyTorch cache dir: getpass.getuser() needs the user in /etc/passwd (done above),
# and the cache dir must be writable by UID 1001
ENV TORCHINDUCTOR_CACHE_DIR=/home/appuser/.cache/torch_inductor

# Expose ports (8000 for API, 7860 for Gradio UI)
EXPOSE 8000 7860

USER 1001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run both API server and Gradio UI
CMD ["/app/start.sh"]
