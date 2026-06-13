# ============================================================================
# THE RAG API — Dockerfile for Google Cloud Run
# ============================================================================
# Multi-stage build:
#   Stage 1 (builder)  — Install Python dependencies with uv into a venv.
#   Stage 2 (runtime)  — Slim Debian image with only system libs + the venv.
# ============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Builder — install Python deps
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

# Install uv (fast Python package manager)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy only dependency manifests first for better layer caching
COPY pyproject.toml uv.lock .python-version ./

# Install dependencies into a project venv (no dev deps, frozen lockfile)
RUN uv sync --frozen --no-install-project --no-dev

# Copy the application source code
COPY . .

# Install the project itself (picks up the local package)
RUN uv sync --frozen --no-dev


# ---------------------------------------------------------------------------
# Stage 2: Runtime — lean image with system dependencies
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# ── System dependencies required by any-to-markdown[all] ──
#
#   tesseract-ocr          → pytesseract (OCR on images)
#   tesseract-ocr-eng      → English language data for Tesseract
#   ffmpeg                 → faster-whisper / av (audio & video processing)
#   libsm6 libxext6        → OpenCV / Pillow image backend libraries
#   libgl1                 → OpenGL stub needed by some image libraries
#   libglib2.0-0           → GLib (common C library dependency)
#
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-eng \
        ffmpeg \
        libsm6 \
        libxext6 \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the pre-built virtual environment from the builder stage
COPY --from=builder /app/.venv /app/.venv

# Copy application source code
COPY --from=builder /app/main.py /app/main.py

# Put the venv on PATH so `python` and all CLI tools resolve from it
ENV PATH="/app/.venv/bin:$PATH"

# Cloud Run injects the PORT env var (default 8080)
ENV PORT=8080

# Ensure Python output is sent straight to the container logs (no buffering)
ENV PYTHONUNBUFFERED=1

# Expose the port (informational — Cloud Run uses $PORT)
EXPOSE ${PORT}

# Healthcheck for local / docker-compose testing (Cloud Run ignores this)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/docs')" || exit 1

# ── Entrypoint ──
# Run uvicorn directly. Cloud Run sets $PORT automatically.
# --workers 1 is recommended for Cloud Run (it scales via container instances).
# --timeout-keep-alive 610 prevents 504s behind Cloud Run's 600s ALB timeout.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers 1 --timeout-keep-alive 610"]
