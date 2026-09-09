"""Shared contracts for the prompt-injection resilience lab.

These Pydantic v2 models are the single source of truth exchanged between the
components of the harness:

    attacker  -> produces / selects an ``Attack``
    app       -> the defended toy assistant answers under a chosen defense
    judge     -> emits a structured ``Verdict`` (was the defense breached?)

``GraphState`` is the object carried along the LangGraph run (attacker -> app ->
judge, with a bounded retry loop). Every other module in the repository imports
these definitions instead of redefining ad-hoc dictionaries, so a change to the
contract is caught everywhere at once.

Scope reminder: this lab exercises its OWN isolated toy assistant only. The
models describe illustrative, defensive probes used to measure and harden the
lab's own contour -- never payloads aimed at real, external systems.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class AttackClass(str, Enum):
    """The four injection classes covered by the attack corpus.

    Each value matches the stem of a file under ``attacks/`` and is used as a
    row label in the results matrix.
    """

    DIRECT_INJECTION = "direct_injection"
    ROLE_OVERRIDE = "role_override"
    ENCODING_OBFUSCATION = "encoding_obfuscation"
    SYSTEM_PROMPT_LEAK = "system_prompt_leak"


class OwaspCategory(str, Enum):
    """OWASP LLM Top-10 mapping for the corpus.

    All four classes in this lab are facets of prompt injection, so they all map
    to ``LLM01``. The enum is kept open as a named type so the mapping is
    explicit in code and can grow if further categories are added later.
    """

    LLM01_PROMPT_INJECTION = "LLM01"


class Attack(BaseModel):
    """A single illustrative attack sample from the corpus.

    Samples are loaded from ``attacks/*.yml``. They are deliberately generic
    probes (e.g. ``ignore previous instructions``-style requests) aimed only at
    the lab's own toy assistant, so that a defense can be measured against them.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(
        ...,
        description="Stable unique id of the sample, e.g. 'direct_injection_001'.",
        min_length=1,
    )
    attack_class: AttackClass = Field(
        ...,
        description="Which injection class this sample belongs to.",
    )
    owasp: OwaspCategory = Field(
        default=OwaspCategory.LLM01_PROMPT_INJECTION,
        description="OWASP LLM Top-10 category, e.g. 'LLM01'.",
    )
    prompt: str = Field(
        ...,
        description="The probe text sent to the defended toy assistant.",
        min_length=1,
    )
    note: str = Field(
        default="",
        description="Short note on the technique and what a breach would look like.",
    )


class Verdict(BaseModel):
    """Structured judgement produced by the LLM-as-judge node.

    ``breached`` is the headline signal used to populate the results matrix.
    ``leaked_system_prompt`` is tracked separately because a system-prompt leak
    is a distinct failure mode from a generic instruction override.
    """

    model_config = ConfigDict(extra="forbid")

    breached: bool = Field(
        ...,
        description="True if the defense failed and the attack objective was met.",
    )
    rationale: str = Field(
        ...,
        description="Concise justification for the verdict, in plain English.",
        min_length=1,
    )
    leaked_system_prompt: bool = Field(
        default=False,
        description="True if the response revealed the hidden system prompt or its canary.",
    )


class Turn(BaseModel):
    """One attacker -> app -> judge exchange, recorded in the run history."""

    model_config = ConfigDict(extra="forbid")

    attempt: int = Field(..., ge=1, description="1-based attempt number for this turn.")
    prompt: str = Field(..., description="The probe text that was sent on this turn.")
    response: str = Field(..., description="The toy assistant's reply on this turn.")
    verdict: Optional[Verdict] = Field(
        default=None,
        description="The judge's verdict for this turn, if it was scored.",
    )


class GraphState(BaseModel):
    """State carried through the LangGraph run.

    The graph loops attacker -> app -> judge. If the current attempt is not a
    breach and ``attempt`` is still below the attempt budget, the attacker may
    produce a variation and the loop runs again; otherwise the run terminates
    with the most recent ``verdict``.
    """

    model_config = ConfigDict(extra="forbid")

    attack: Attack = Field(..., description="The attack sample under test.")
    defense_name: str = Field(
        ...,
        description="Which defense contour is active (e.g. 'none', 'system_prompt').",
        min_length=1,
    )
    attempt: int = Field(
        default=1,
        ge=1,
        description="Current 1-based attempt number within the retry loop.",
    )
    max_attempts: int = Field(
        default=1,
        ge=1,
        description="Attempt budget; the loop stops once this is reached.",
    )
    history: list[Turn] = Field(
        default_factory=list,
        description="Ordered record of every attacker -> app -> judge exchange.",
    )
    verdict: Optional[Verdict] = Field(
        default=None,
        description="Latest verdict; None until the judge has scored an attempt.",
    )

    @property
    def breached(self) -> bool:
        """Convenience flag: True once any verdict in the run reports a breach."""
        if self.verdict is not None and self.verdict.breached:
            return True
        return any(t.verdict is not None and t.verdict.breached for t in self.history)


__all__ = [
    "AttackClass",
    "OwaspCategory",
    "Attack",
    "Verdict",
    "Turn",
    "GraphState",
]
