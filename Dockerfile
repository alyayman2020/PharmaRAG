# Multi-stage. The index is MOUNTED or fetched at startup, never baked in —
# a 300MB+ index makes builds slow and blows past free-tier image limits.
FROM python:3.12-slim AS base
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app

COPY pyproject.toml uv.lock* ./
COPY packages ./packages
RUN uv sync --no-dev --no-install-project || uv sync --no-dev

COPY src ./src
COPY app ./app
COPY scripts ./scripts

ENV PYTHONPATH=/app/src:/app/packages/spl_parser/src \
    DEVICE=cpu \
    STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0
EXPOSE 7860
CMD ["uv", "run", "streamlit", "run", "app/main.py"]
