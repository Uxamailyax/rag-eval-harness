"""Cache-aware Groq client.

Every LLM call in this project goes through `complete()`. That means caching,
latency timing, and token accounting happen in one place instead of being
re-implemented at each call site.

Cache-first is not an optimisation here, it is what makes the project runnable:
Groq's free tier allows roughly 200K tokens/day per model, and judge calibration
alone needs more than that. Paying once and replaying from disk is the difference
between a re-run costing nothing and costing a day.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, asdict
from typing import Any

from groq import Groq
from groq import APIStatusError, APIConnectionError, RateLimitError

from rag_eval_harness.cache import LLMCache
from rag_eval_harness.config import Settings, settings as default_settings

PROVIDER = "groq"

# Retry policy for 429s and transient network failures. Groq's free tier caps at
# ~30 requests/minute, so a short backoff usually clears it.
MAX_ATTEMPTS = 5
BASE_BACKOFF_S = 2.0
MAX_BACKOFF_S = 60.0


@dataclass(frozen=True)
class LLMResponse:
    """Result of a completion, whether served from cache or from the API."""

    text: str
    model: str
    cached: bool
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: float | None = None

    @property
    def total_tokens(self) -> int | None:
        if self.prompt_tokens is None or self.completion_tokens is None:
            return None
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LLMClient:
    """Groq client that checks the cache before making a network call.

    One client can serve both the generator and the judge; pass `model` per call.
    """

    def __init__(
        self,
        cfg: Settings | None = None,
        cache: LLMCache | None = None,
        api_key: str | None = None,
    ):
        self.cfg = cfg or default_settings
        key = api_key or self.cfg.groq_api_key
        if not key:
            raise ValueError(
                "No Groq API key. Set GROQ_API_KEY in .env at the project root."
            )
        self._client = Groq(api_key=key)
        self.cache = cache or LLMCache(self.cfg.cache_path)
        self.api_calls = 0

    def complete(
        self,
        prompt: str,
        model: str | None = None,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        seed: int | None = None,
        force_refresh: bool = False,
    ) -> LLMResponse:
        """Return a completion for `prompt`, from cache when available.

        `temperature=0` and a fixed `seed` are the defaults because evaluation runs
        should be as close to deterministic as the provider allows. Anything that
        changes the request — model, system prompt, temperature, seed — changes the
        cache key, so a cached answer is never served for a different question.
        """
        model = model or self.cfg.generator_model
        params: dict[str, Any] = {
            "system": system,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": seed,
        }

        if not force_refresh:
            hit = self.cache.get(PROVIDER, model, prompt, params)
            if hit is not None:
                return LLMResponse(
                    text=hit.response,
                    model=model,
                    cached=True,
                    prompt_tokens=hit.prompt_tokens,
                    completion_tokens=hit.completion_tokens,
                    latency_ms=hit.latency_ms,
                )

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        started = time.perf_counter()
        completion = self._call_with_retry(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        self.api_calls += 1

        text = completion.choices[0].message.content or ""
        usage = getattr(completion, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)

        self.cache.put(
            provider=PROVIDER,
            model=model,
            prompt=prompt,
            params=params,
            response=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
        )

        return LLMResponse(
            text=text,
            model=model,
            cached=False,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
        )

    def _call_with_retry(self, **kwargs: Any) -> Any:
        """Call Groq, backing off on rate limits and transient failures.

        Backoff is exponential with jitter. Jitter matters when several calls are
        throttled at once: without it they all retry at the same instant and get
        throttled again together.
        """
        last_error: Exception | None = None

        for attempt in range(MAX_ATTEMPTS):
            try:
                return self._client.chat.completions.create(**kwargs)
            except RateLimitError as exc:
                last_error = exc
            except APIConnectionError as exc:
                last_error = exc
            except APIStatusError as exc:
                # 5xx is worth retrying; 4xx (bad request, auth) is not.
                if exc.status_code < 500:
                    raise
                last_error = exc

            if attempt == MAX_ATTEMPTS - 1:
                break

            delay = min(BASE_BACKOFF_S * (2**attempt), MAX_BACKOFF_S)
            delay += random.uniform(0, delay * 0.25)
            print(
                f"  [llm] attempt {attempt + 1}/{MAX_ATTEMPTS} failed "
                f"({type(last_error).__name__}); retrying in {delay:.1f}s"
            )
            time.sleep(delay)

        raise RuntimeError(
            f"Groq call failed after {MAX_ATTEMPTS} attempts: {last_error}"
        ) from last_error

    def stats(self) -> dict[str, Any]:
        """Cache stats plus the number of real API calls made this session."""
        return {"api_calls_this_session": self.api_calls, **self.cache.stats()}

    def close(self) -> None:
        self.cache.close()

    def __enter__(self) -> "LLMClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
