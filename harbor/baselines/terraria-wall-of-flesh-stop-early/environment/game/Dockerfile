FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TERRARIA_BENCH_IMAGE_DIGEST=terrariabench-harbor-game

RUN pip install --no-cache-dir "mcp>=1.12,<2" "uvicorn>=0.27" "starlette>=0.37"

WORKDIR /srv
COPY terraria_bench ./terraria_bench
COPY artifacts ./artifacts
COPY server.py ./server.py

EXPOSE 8000
CMD ["python", "server.py"]
