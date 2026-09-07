# CPU image. For CUDA, see Dockerfile.cuda.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VISION_MCP_DEVICE=cpu

# OpenCV runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN uv pip install --system --no-cache .

COPY scripts ./scripts
RUN python scripts/download_models.py

# stdio MCP server
ENTRYPOINT ["python", "-m", "vision_mcp.server"]
