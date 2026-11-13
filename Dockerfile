FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TERRARIA_BENCH_IMAGE_DIGEST=terrariabench-local-py311

WORKDIR /app

COPY pyproject.toml README.md Makefile ./
COPY terraria_bench ./terraria_bench
COPY dashboard ./dashboard
COPY tests ./tests

RUN pip install --no-cache-dir -e ".[dev]"

EXPOSE 8000 8080 8765

CMD ["python", "-m", "terraria_bench.harness.runner", "--eval-standard"]
