"""Configuration and run manifests.

Every eval run writes a manifest recording exactly what produced it: the git SHA,
a hash of the effective config, the model IDs, and the installed versions of the
libraries that affect results. A number without a manifest is not a result.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root, resolved from this file's location: src/rag_eval_harness/config.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Libraries whose version can change a metric. Recorded in every manifest.
TRACKED_PACKAGES = (
    "numpy",
    "scikit-learn",
    "rank-bm25",
    "groq",
    "sentence-transformers",
    "ragas",
)


class Settings(BaseSettings):
    """Runtime settings. Values come from .env, with the defaults below as fallback."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Paths -------------------------------------------------------------
    data_dir: Path = PROJECT_ROOT / "data"
    results_dir: Path = PROJECT_ROOT / "results"
    cache_path: Path = PROJECT_ROOT / ".cache" / "llm_cache.sqlite"

    # --- API keys ----------------------------------------------------------
    groq_api_key: str = ""
    google_api_key: str = ""

    # --- Models ------------------------------------------------------------
    # Pinned exactly. A model ID change invalidates comparisons, so it lands
    # in the manifest and the cache key.
    generator_model: str = "openai/gpt-oss-120b"
    judge_model: str = "qwen/qwen3.6-27b"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"

    # --- Postgres (Phase 5+, not needed before then) -----------------------
    postgres_user: str = "raguser"
    postgres_password: str = "ragpass"
    postgres_db: str = "rageval"
    postgres_host: str = "localhost"
    postgres_port: int = 5433

    # --- Retrieval defaults ------------------------------------------------
    chunk_size: int = 256
    chunk_overlap: int = 0
    top_k: int = 10
    random_seed: int = 42

    @property
    def corpus_dir(self) -> Path:
        return self.data_dir / "corpus"

    @property
    def benchmarks_dir(self) -> Path:
        return self.data_dir / "benchmarks"

    @property
    def runs_dir(self) -> Path:
        return self.results_dir / "runs"

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    def public_dict(self) -> dict[str, Any]:
        """Config as a dict with secrets removed. Safe to write to disk."""
        secret_fields = {"groq_api_key", "google_api_key", "postgres_password"}
        out: dict[str, Any] = {}
        for key, value in self.model_dump().items():
            if key in secret_fields:
                continue
            out[key] = str(value) if isinstance(value, Path) else value
        return out


def git_sha(short: bool = False) -> str:
    """Current commit SHA, or 'unknown' outside a git checkout."""
    args = ["git", "rev-parse", "--short" if short else "HEAD"]
    try:
        result = subprocess.run(
            args, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def git_is_dirty() -> bool:
    """True if there are uncommitted changes. A dirty run is not reproducible."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0 and bool(result.stdout.strip())


def package_versions() -> dict[str, str]:
    """Installed versions of the packages that can move a metric."""
    versions: dict[str, str] = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def config_hash(payload: dict[str, Any]) -> str:
    """Stable 12-char hash of a config dict. Same config -> same hash, always."""
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def write_manifest(
    settings: Settings,
    run_params: dict[str, Any] | None = None,
    run_id: str | None = None,
) -> Path:
    """Create results/runs/<run_id>/ and write manifest.json into it.

    run_params holds anything specific to this run that isn't in Settings, e.g.
    {"retriever": "bm25", "chunker": "fixed"}. It feeds the config hash, so two
    runs that differ only in retriever get different directories.
    """
    run_params = run_params or {}
    effective = {"settings": settings.public_dict(), "run_params": run_params}
    chash = config_hash(effective)
    run_id = run_id or f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{chash}"

    run_dir = settings.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "git_dirty": git_is_dirty(),
        "config_hash": chash,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": package_versions(),
        **effective,
    }

    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest_path


# Single shared instance. Import this rather than constructing Settings().
settings = Settings()
