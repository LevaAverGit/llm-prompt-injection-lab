"""Baseline defense: no protection at all.

This is the control column of the results matrix. The user input is passed to
the model unchanged, alongside the original (unhardened) system prompt. Its
breach rate is the reference the other contours are measured against.
"""

from __future__ import annotations

from typing import Any

from .base import EffectivePrompt, Outcome


def defend(user_input: str, system_prompt: str) -> tuple[Outcome, dict[str, Any]]:
    """Pass the input straight through with no changes."""

    outcome: Outcome = EffectivePrompt(
        system_prompt=system_prompt, user_input=user_input
    )
    meta: dict[str, Any] = {"defense": "none", "action": "passthrough"}
    return outcome, meta


__all__ = ["defend"]
