# Reproducible environment for the harness.
#
# The image carries Python, the locked dependencies and the code. It does not carry
# the corpus or the models — both are mounted or downloaded at run time, for reasons
# worth stating:
#
#   corpus  LegalBench-RAG's underlying documents come from ContractNLI, CUAD, MAUD
#           and PrivacyQA, each with its own usage policy. Redistribution rights were
#           never established, so the corpus is mounted from the host rather than
#           baked into an image that could be pushed to a registry.
#
#   models  bge-small is 130 MB and bge-reranker-base is 1.1 GB. Baking them in would
#           make the image enormous for no benefit — they are cached in a named volume
#           on first run and reused thereafter.
#
# No GPU here. The image is CPU-only, which makes it portable at the cost of speed:
# embedding the full corpus took 12 minutes on a GTX 1650 and around an hour on CPU.
# That is acceptable because the results are cached, and because the point of the
# container is reproducibility rather than throughput.

FROM python:3.12-slim

# uv, for the same reason it is used locally: uv.lock pins every transitive dependency,
# so the container resolves to the exact versions the results were produced with.
COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /uvx /bin/

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    # HuggingFace caches models here; mounted as a named volume so a rebuild does not
    # re-download 1.2 GB.
    HF_HOME=/models \
    # Windows redirects default to cp1252 and the corpus contains characters that
    # cannot encode; Phase 7 hit this writing case files.
    PYTHONIOENCODING=utf-8

# Dependencies first, so editing source does not invalidate the layer that took
# minutes to build.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

COPY src/ ./src/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY prompts/ ./prompts/
COPY evals/ ./evals/
COPY results/baseline.json ./results/baseline.json

RUN uv sync --frozen --no-dev

# Default to the gate rather than the full pipeline: it needs no corpus, no API key
# and no network, so `docker compose up` does something useful on a clean clone even
# before the corpus is downloaded.
CMD ["uv", "run", "pytest", "tests/", "-v", "--tb=short"]
