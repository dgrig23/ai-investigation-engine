# Production container (used by Render's free web service; works on any Docker host).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FASTEMBED_CACHE_PATH=/app/models \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app

RUN pip install --upgrade pip && pip install \
    "fastapi>=0.115" "uvicorn[standard]>=0.30" "pydantic>=2.7" "pydantic-settings>=2.3" "python-multipart>=0.0.9" \
    "httpx>=0.27" "qdrant-client>=1.12" "fastembed>=0.4" "openai>=1.40" "pypdf>=4.0" "python-docx>=1.1" \
    "beautifulsoup4>=4.12" "numpy>=1.26"

# Bake the ONNX models into the image so a cold start does not download them.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5'); \
from fastembed.rerank.cross_encoder import TextCrossEncoder; TextCrossEncoder('Xenova/ms-marco-MiniLM-L-6-v2')"

COPY app ./app
RUN mkdir -p /app/data/cache && chown -R 1000:1000 /app/data /app/models
USER 1000

# Render (and most hosts) pass the port in $PORT.
EXPOSE 10000
CMD ["sh", "-c", "uvicorn app.api.main:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1"]
