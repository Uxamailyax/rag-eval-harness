"""On-disk cache for LLM responses.

Every LLM call is keyed by sha256(provider + model + params + prompt). A repeated
call costs nothing, which is what makes re-runs free under a rate-limited free tier.
Token counts are stored alongside each response, so cost and usage reporting later
needs no extra instrumentation.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_cache (
    key                TEXT PRIMARY KEY,
    provider           TEXT NOT NULL,
    model              TEXT NOT NULL,
    prompt             TEXT NOT NULL,
    params_json        TEXT NOT NULL,
    response           TEXT NOT NULL,
    prompt_tokens      INTEGER,
    completion_tokens  INTEGER,
    latency_ms         REAL,
    created_at_utc     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_llm_cache_model ON llm_cache(provider, model);
"""


@dataclass(frozen=True)
class CachedResponse:
    """A response served from cache, with the accounting captured at call time."""

    response: str
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_ms: float | None
    created_at_utc: str


def make_key(provider: str, model: str, prompt: str, params: dict[str, Any]) -> str:
    """Cache key. Any change to provider, model, params, or prompt is a new key.

    Params are sorted so that {"temperature": 0, "seed": 1} and
    {"seed": 1, "temperature": 0} produce the same key.
    """
    payload = json.dumps(
        {
            "provider": provider,
            "model": model,
            "prompt": prompt,
            "params": params,
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class LLMCache:
    """SQLite-backed cache. Safe to use from multiple threads in one process."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # WAL lets reads proceed during writes and survives an interrupted run.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)
        self._conn.commit()
        self.hits = 0
        self.misses = 0

    def get(
        self, provider: str, model: str, prompt: str, params: dict[str, Any]
    ) -> CachedResponse | None:
        key = make_key(provider, model, prompt, params)
        with self._lock:
            row = self._conn.execute(
                "SELECT response, prompt_tokens, completion_tokens, latency_ms,"
                " created_at_utc FROM llm_cache WHERE key = ?",
                (key,),
            ).fetchone()
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        return CachedResponse(
            response=row["response"],
            prompt_tokens=row["prompt_tokens"],
            completion_tokens=row["completion_tokens"],
            latency_ms=row["latency_ms"],
            created_at_utc=row["created_at_utc"],
        )

    def put(
        self,
        provider: str,
        model: str,
        prompt: str,
        params: dict[str, Any],
        response: str,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        latency_ms: float | None = None,
    ) -> str:
        key = make_key(provider, model, prompt, params)
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO llm_cache (key, provider, model, prompt,"
                " params_json, response, prompt_tokens, completion_tokens,"
                " latency_ms, created_at_utc) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    key,
                    provider,
                    model,
                    prompt,
                    json.dumps(params, sort_keys=True, default=str),
                    response,
                    prompt_tokens,
                    completion_tokens,
                    latency_ms,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            self._conn.commit()
        return key

    def stats(self) -> dict[str, Any]:
        """Hit/miss counts for this session, plus totals across the whole cache."""
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n, COALESCE(SUM(prompt_tokens), 0) AS pt,"
                " COALESCE(SUM(completion_tokens), 0) AS ct FROM llm_cache"
            ).fetchone()
        lookups = self.hits + self.misses
        return {
            "session_hits": self.hits,
            "session_misses": self.misses,
            "session_hit_rate": round(self.hits / lookups, 4) if lookups else None,
            "cached_entries": row["n"],
            "total_prompt_tokens": row["pt"],
            "total_completion_tokens": row["ct"],
        }

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "LLMCache":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
