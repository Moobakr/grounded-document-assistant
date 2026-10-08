FROM python:3.12-slim-bookworm@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    HOME=/tmp

WORKDIR /app
COPY requirements.txt requirements-runtime.txt requirements-web.txt requirements-linux.lock ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.7.1+cpu -c requirements-linux.lock --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements-web.txt -c requirements-linux.lock \
    && pip check
RUN groupadd --gid 10001 assistant \
    && useradd --uid 10001 --gid assistant --no-create-home assistant \
    && mkdir -p /app/data /app/models/embeddings /app/documents \
    && chown -R assistant:assistant /app/data /app/models
COPY src ./src
USER 10001:10001
EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/api/health', timeout=2)" || exit 1
CMD ["python", "-m", "src.web", "--host", "0.0.0.0", "--port", "8765"]
