"""Load versioned prompts from disk.

Prompts live in `prompts/<version>/` as YAML rather than as string literals in Python,
because a prompt is code: it changes every answer the system produces, and a change
buried in a source file is an untracked deploy.

Keeping them as files buys three things:

  - the version is recorded in every answer record and run manifest, so a score can be
    traced to the exact text that produced it
  - a change appears in a diff and is reviewable in a pull request
  - the CI gate can invalidate cached answers automatically, because the prompt text is
    part of the LLM cache key

**Scores produced under different prompt versions are not comparable.** This matters
most for the judge: its calibration — Cohen's kappa 0.87 against 30 human labels —
applies to one specific rubric. Changing that rubric invalidates the calibration until
the 30 cases are re-judged.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from rag_eval_harness.config import PROJECT_ROOT

PROMPTS_DIR = PROJECT_ROOT / "prompts"
DEFAULT_VERSION = "v1"


@dataclass(frozen=True)
class Prompt:
    """A loaded prompt, with everything needed to reproduce a call."""

    name: str
    version: str
    system: str
    user_template: str
    parameters: dict[str, Any]
    metadata: dict[str, Any]

    @property
    def fingerprint(self) -> str:
        """Stable hash of the prompt text and parameters.

        Recorded alongside results so a score can be matched to the exact prompt that
        produced it, even if the version label is reused by mistake.
        """
        blob = f"{self.system}\u0000{self.user_template}\u0000{sorted(self.parameters.items())}"
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]

    def render(self, **kwargs: str) -> str:
        """Fill the user template.

        Raises rather than silently producing a prompt with a literal `{context}` in
        it, which would be sent to the model and quietly ruin the result.
        """
        try:
            return self.user_template.format(**kwargs)
        except KeyError as exc:
            raise KeyError(
                f"prompt '{self.name}' v{self.version} needs placeholder {exc}"
            ) from exc


@lru_cache(maxsize=32)
def load(name: str, version: str = DEFAULT_VERSION) -> Prompt:
    """Load `prompts/<version>/<name>.yaml`."""
    path = PROMPTS_DIR / version / f"{name}.yaml"
    if not path.is_file():
        available = sorted(p.stem for p in (PROMPTS_DIR / version).glob("*.yaml")) \
            if (PROMPTS_DIR / version).is_dir() else []
        raise FileNotFoundError(
            f"no prompt '{name}' in version '{version}' ({path})\n"
            f"available: {available or 'none — is prompts/ present?'}"
        )

    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    for field in ("system", "user"):
        if field not in data:
            raise ValueError(f"{path} is missing required field '{field}'")

    metadata = {
        k: v for k, v in data.items()
        if k not in {"system", "user", "parameters", "version"}
    }

    return Prompt(
        name=name,
        version=str(data.get("version", version)),
        system=data["system"].rstrip(),
        user_template=data["user"].rstrip(),
        parameters=data.get("parameters", {}),
        metadata=metadata,
    )


def available_versions() -> list[str]:
    if not PROMPTS_DIR.is_dir():
        return []
    return sorted(p.name for p in PROMPTS_DIR.iterdir() if p.is_dir())
