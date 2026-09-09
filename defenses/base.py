"""Shared types and registry for the defensive contours.

Every defense exposes the same tiny interface::

    defend(user_input: str, system_prompt: str) -> tuple[Outcome, dict]

The first element is an *effective prompt or a block* -- exactly one of:

``EffectivePrompt``
    Let the input through. Carries the (possibly hardened) ``system_prompt`` and
    the (possibly transformed) ``user_input`` the app should actually send to
    the model.
``Block``
    Reject the input *before* the model is called. Carries the canned
    ``response`` the app should return instead, so a blocked probe never reaches
    the assistant at all.
``Skipped``
    The defense could not run in this environment (e.g. an optional dependency
    is missing). The runner records the matrix cell as N/A rather than as a
    pass or a breach.

Scope reminder: these contours guard the lab's OWN isolated toy assistant. They
exist to measure and harden that single fictional target -- never to attack any
real, external system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Union, runtime_checkable

# Delimiters used by the hardened system prompt to "spotlight" untrusted user
# input. They are deliberately long and unusual so ordinary text never collides
# with them and an attacker cannot easily guess them to forge a boundary.
INPUT_OPEN = "<<<UNTRUSTED_USER_INPUT_B7F2>>>"
INPUT_CLOSE = "<<<END_UNTRUSTED_USER_INPUT_B7F2>>>"

# Canned reply returned by a pre-model block. It refuses without performing the
# injected task and without disclosing the hidden system prompt or its canary.
BLOCKED_RESPONSE = (
    "I can't help with that request. I'll keep following my original "
    "instructions and only answer legitimate FAQ questions."
)


@dataclass(frozen=True)
class EffectivePrompt:
    """The input is allowed through; send these to the model."""

    system_prompt: str
    user_input: str


@dataclass(frozen=True)
class Block:
    """The input is rejected before the model is called."""

    response: str
    reason: str


@dataclass(frozen=True)
class Skipped:
    """The defense is unavailable in this environment (graceful skip)."""

    reason: str


# The first element of a ``defend`` return value.
Outcome = Union[EffectivePrompt, Block, Skipped]


@runtime_checkable
class Defense(Protocol):
    """Callable contract implemented by every defense module's ``defend``."""

    def __call__(
        self, user_input: str, system_prompt: str
    ) -> tuple[Outcome, dict[str, Any]]:
        ...


def wrap_untrusted(user_input: str) -> str:
    """Place ``user_input`` inside the spotlighting delimiters."""

    return f"{INPUT_OPEN}\n{user_input}\n{INPUT_CLOSE}"


@dataclass
class _Registry:
    """Name -> defense mapping, populated by :mod:`defenses`."""

    _by_name: dict[str, Defense] = field(default_factory=dict)

    def register(self, name: str, defense: Defense) -> None:
        if name in self._by_name:
            raise ValueError(f"defense {name!r} is already registered")
        self._by_name[name] = defense

    def get(self, name: str) -> Defense:
        try:
            return self._by_name[name]
        except KeyError:
            known = ", ".join(sorted(self._by_name)) or "<none>"
            raise KeyError(
                f"unknown defense {name!r}; registered defenses: {known}"
            ) from None

    def names(self) -> list[str]:
        return list(self._by_name)


__all__ = [
    "INPUT_OPEN",
    "INPUT_CLOSE",
    "BLOCKED_RESPONSE",
    "EffectivePrompt",
    "Block",
    "Skipped",
    "Outcome",
    "Defense",
    "wrap_untrusted",
]
