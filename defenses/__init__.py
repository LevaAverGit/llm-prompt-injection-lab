"""Defensive contours for the prompt-injection resilience lab.

Every defense implements the same interface::

    defend(user_input: str, system_prompt: str) -> tuple[Outcome, dict]

where ``Outcome`` is an :class:`EffectivePrompt` (let it through, with the
possibly-hardened prompt and possibly-transformed input), a :class:`Block`
(reject before the model runs, with a canned response), or a :class:`Skipped`
(the defense is unavailable in this environment).

Defenses are registered by name so the runner can iterate the matrix columns:

===============  ============================================================
Name             Idea
===============  ============================================================
``none``         Baseline, no protection -- the control column.
``system_prompt``Hardened system prompt: delimiters, spotlighting, refusal to
                 change role or reveal the instructions / canary.
``input_filter`` Heuristic: decode base64 / rot13 / leetspeak / reversal /
                 spacing first, then match injection signatures and block.
``both``         Input filter gate followed by the hardened system prompt.
``llm_guard``    Protect AI's LLM Guard PromptInjection scanner (optional
                 extra; gracefully skipped when not installed).
===============  ============================================================

Usage::

    from defenses import get_defense, available_defenses

    defend = get_defense("both")
    outcome, meta = defend(user_input, system_prompt)

All contours guard the lab's OWN isolated toy assistant. They exist to measure
and harden that single fictional target, never to attack real systems.
"""

from __future__ import annotations

from typing import Any

from . import both as _both
from . import input_filter as _input_filter
from . import llm_guard as _llm_guard
from . import none as _none
from . import system_prompt as _system_prompt
from .base import (
    BLOCKED_RESPONSE,
    INPUT_CLOSE,
    INPUT_OPEN,
    Block,
    Defense,
    EffectivePrompt,
    Outcome,
    Skipped,
    _Registry,
    wrap_untrusted,
)

_registry = _Registry()
_registry.register("none", _none.defend)
_registry.register("system_prompt", _system_prompt.defend)
_registry.register("input_filter", _input_filter.defend)
_registry.register("both", _both.defend)
_registry.register("llm_guard", _llm_guard.defend)


def get_defense(name: str) -> Defense:
    """Return the ``defend`` callable registered under ``name``.

    Raises ``KeyError`` for an unknown name (the message lists the known ones).
    """

    return _registry.get(name)


def available_defenses() -> list[str]:
    """Return the registered defense names in registration order.

    ``llm_guard`` is always listed; whether it actually runs is decided at call
    time (it returns :class:`Skipped` when its optional dependency is missing).
    """

    return _registry.names()


def defend(name: str, user_input: str, system_prompt: str) -> tuple[Outcome, dict[str, Any]]:
    """Look up ``name`` and run it -- convenience over :func:`get_defense`."""

    return get_defense(name)(user_input, system_prompt)


__all__ = [
    "get_defense",
    "available_defenses",
    "defend",
    "Defense",
    "Outcome",
    "EffectivePrompt",
    "Block",
    "Skipped",
    "wrap_untrusted",
    "INPUT_OPEN",
    "INPUT_CLOSE",
    "BLOCKED_RESPONSE",
]
