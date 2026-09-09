"""Combined defense: heuristic input filter *and* hardened system prompt.

The two layers are complementary. The input filter is a cheap pre-model gate
that blocks obviously hostile input before it ever reaches the assistant; the
hardened system prompt defends whatever gets through. This contour runs the
filter first and, only if the input is allowed, applies the prompt hardening.
"""

from __future__ import annotations

from typing import Any

from . import input_filter, system_prompt
from .base import Block, Outcome


def defend(user_input: str, system_prompt_text: str) -> tuple[Outcome, dict[str, Any]]:
    """Gate with the input filter, then harden the surviving input."""

    filtered, filter_meta = input_filter.defend(user_input, system_prompt_text)
    if isinstance(filtered, Block):
        meta: dict[str, Any] = {
            "defense": "both",
            "action": "block",
            "filter": filter_meta,
        }
        return filtered, meta

    hardened, prompt_meta = system_prompt.defend(user_input, system_prompt_text)
    meta = {
        "defense": "both",
        "action": "harden",
        "filter": filter_meta,
        "system_prompt": prompt_meta,
    }
    return hardened, meta


__all__ = ["defend"]
