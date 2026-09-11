"""Phase 0 gate: prove the LLM cache works.

Calls Groq twice with an identical prompt. The first call should hit the network;
the second should be served from SQLite with no API traffic at all.

Run with:  uv run python scripts/phase0_gate.py
"""

from __future__ import annotations

import json
import sys

from rag_eval_harness.config import settings, write_manifest
from rag_eval_harness.llm import LLMClient

PROMPT = "Reply with exactly one word: ping"


def main() -> int:
    print(f"cache file : {settings.cache_path}")
    print(f"model      : {settings.generator_model}\n")

    with LLMClient() as client:
        print("--- call 1 (expect: cached=False, real API call) ---")
        first = client.complete(PROMPT, temperature=0.0, max_tokens=512)
        print(f"  text        : {first.text.strip()!r}")
        print(f"  cached      : {first.cached}")
        print(f"  tokens      : {first.prompt_tokens} in / {first.completion_tokens} out")
        print(f"  latency_ms  : {first.latency_ms:.1f}" if first.latency_ms else "")

        print("\n--- call 2, identical prompt (expect: cached=True, no API call) ---")
        second = client.complete(PROMPT, temperature=0.0, max_tokens=512)
        print(f"  text        : {second.text.strip()!r}")
        print(f"  cached      : {second.cached}")

        print("\n--- call 3, one word changed (expect: cached=False, new API call) ---")
        third = client.complete(
            PROMPT.replace("ping", "pong"), temperature=0.0, max_tokens=512
        )
        print(f"  text        : {third.text.strip()!r}")
        print(f"  cached      : {third.cached}")

        stats = client.stats()
        print("\n--- stats ---")
        print(json.dumps(stats, indent=2))

        checks = {
            "call 1 was a real API call": first.cached is False,
            "call 2 was served from cache": second.cached is True,
            "call 2 returned identical text": first.text == second.text,
            "changed prompt was a cache miss": third.cached is False,
            "exactly 2 API calls were made": stats["api_calls_this_session"] == 2,
            "token accounting was captured": first.prompt_tokens is not None,
        }

        print("\n--- gate ---")
        for label, ok in checks.items():
            print(f"  [{'PASS' if ok else 'FAIL'}] {label}")

        passed = all(checks.values())
        write_manifest(
            settings,
            {"phase": 0, "gate": "llm_cache", "passed": passed, "stats": stats},
        )

        print(
            "\nPHASE 0 GATE: PASS" if passed else "\nPHASE 0 GATE: FAIL"
        )
        return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
