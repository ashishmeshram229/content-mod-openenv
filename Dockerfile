FROM python:3.11-slim

# Metadata
LABEL maintainer="OpenEnv Hackathon"
LABEL description="Content Moderation OpenEnv — Trust & Safety Agent Environment"
LABEL version="1.0.0"

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (HF Spaces requirement)
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install Python deps first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project (preserves env/ package structure)
COPY --chown=appuser:appuser . .

# Verify the env package is importable at build time
RUN python -c "from env.environment import ContentModerationEnv; print('env import OK')"

USER appuser

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=5 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1", "--timeout-keep-alive", "75"]