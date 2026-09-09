"""On-disk response cache for the harness.

Every model call (the toy assistant's reply, the judge's verdict, an attacker
variation) is cached to a small JSON file so the whole matrix reproduces offline.
Cache entries are keyed by ``provider__model__scenario`` -- for example
``ollama__gemma3:latest__app|both|direct_injection_001|a1``. The committed cache
lets ``make run`` rebuild the exact same matrix without a running model, while
``make run-llm`` refreshes it live.

Three modes control how the cache is consulted:

``FROM_CACHE``
    Strictly offline. A miss is an error (the user needs to run live first).
``REFRESH``
    Always call the model and overwrite the stored entry.
``AUTO``
    Use a stored entry when present, otherwise call the model and store it.

Scope reminder: the cached exchanges come from the lab's OWN isolated toy
assistant, recorded to make the defensive results reproducible.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .corpus import REPO_ROOT

# Committed cache directory (not git-ignored) so ``make run`` works offline.
DEFAULT_CACHE_DIR = REPO_ROOT / "cache"

# Characters that are awkward in a file name (``gemma3:latest`` has a colon,
# scenarios use ``|``). The full, unsanitized key is still stored inside the file.
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


class CacheMode(str, Enum):
    """How the runner should consult the response cache."""

    FROM_CACHE = "from-cache"
    REFRESH = "refresh"
    AUTO = "auto"


class CacheMiss(RuntimeError):
    """Raised in ``FROM_CACHE`` mode when a required entry is absent."""


def make_key(provider: str, model: str, scenario: str) -> str:
    """Build the canonical ``provider__model__scenario`` cache key."""

    return f"{provider}__{model}__{scenario}"


def _filename(key: str) -> str:
    """Map a cache key to a safe, still-recognisable file name.

    A short digest of the full key is appended so two keys that sanitise to the
    same string never collide.
    """

    safe = _UNSAFE.sub("_", key).strip("_")
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]
    return f"{safe}.{digest}.json"


@dataclass
class ResponseCache:
    """A directory of JSON entries, one per model call."""

    directory: Path = DEFAULT_CACHE_DIR
    mode: CacheMode = CacheMode.AUTO

    def __post_init__(self) -> None:
        self.directory = Path(self.directory)

    def _path(self, key: str) -> Path:
        return self.directory / _filename(key)

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Return the stored payload for ``key``, or ``None`` if not cached."""

        path = self._path(key)
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):  # pragma: no cover - defensive
            return None

    def put(self, key: str, payload: Dict[str, Any]) -> None:
        """Store ``payload`` for ``key`` (creating the cache dir if needed)."""

        self.directory.mkdir(parents=True, exist_ok=True)
        record = {"key": key, **payload}
        self._path(key).write_text(
            json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def resolve(self, key: str, compute: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
        """Return the payload for ``key`` honouring the active :class:`CacheMode`.

        ``compute`` is the (potentially expensive) model call, invoked only when
        the mode and cache state require it. It must return a JSON-serialisable
        payload dict; the same dict is stored and returned.
        """

        if self.mode is CacheMode.REFRESH:
            payload = compute()
            self.put(key, payload)
            return payload

        cached = self.get(key)
        if cached is not None:
            # Drop the bookkeeping "key" field before handing back the payload.
            return {k: v for k, v in cached.items() if k != "key"}

        if self.mode is CacheMode.FROM_CACHE:
            raise CacheMiss(
                f"no cached response for {key!r}. Run `make run-llm` (live) once "
                "to populate the committed cache, then `make run` reproduces it "
                "offline."
            )

        payload = compute()
        self.put(key, payload)
        return payload


__all__ = [
    "DEFAULT_CACHE_DIR",
    "CacheMode",
    "CacheMiss",
    "ResponseCache",
    "make_key",
]
