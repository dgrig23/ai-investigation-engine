# Container for Hugging Face Spaces (Docker SDK, free CPU tier: 2 vCPU, 16 GB RAM).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FASTEMBED_CACHE_PATH=/app/models \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app

COPY pyproject.toml ./
RUN pip install --upgrade pip && pip install \
    "fastapi>=0.115" "uvicorn[standard]>=0.30" "pydantic>=2.7" "pydantic-settings>=2.3" "python-multipart>=0.0.9" \
    "httpx>=0.27" "qdrant-client>=1.12" "fastembed>=0.4" "openai>=1.40" "pypdf>=4.0" "python-docx>=1.1" \
    "beautifulsoup4>=4.12" "numpy>=1.26"

# Bake the ONNX models into the image so a cold start does not download them.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5'); \
from fastembed.rerank.cross_encoder import TextCrossEncoder; TextCrossEncoder('Xenova/ms-marco-MiniLM-L-6-v2')"

COPY app ./app

# Spaces run the container as user 1000; give it a writable data dir (LLM response cache).
RUN mkdir -p /app/data/cache && chown -R 1000:1000 /app/data /app/models
USER 1000

EXPOSE 7860
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
