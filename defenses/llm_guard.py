"""Real-world guardrail defense using Protect AI's LLM Guard.

This contour wraps LLM Guard's ``PromptInjection`` input scanner -- an actual
industrial prompt-injection classifier -- so the matrix can compare the lab's
hand-written heuristics against a shipping tool.

LLM Guard pulls in heavy dependencies (torch / transformers), so it is an
*optional* extra (``requirements-guard.txt``). It is imported lazily and, if it
is not installed, this defense returns :class:`Skipped` so the rest of the
matrix still runs. The loaded scanner is cached across calls because building it
downloads and initializes a model.
"""

from __future__ import annotations

from typing import Any

from .base import Block, EffectivePrompt, Outcome, Skipped

# Risk threshold above which the scanner treats input as an injection. LLM
# Guard's own default is 0.5; kept explicit so the behavior is reproducible.
_THRESHOLD = 0.5

# Cached scanner instance and a memo of why it is unavailable, so we neither
# rebuild the model on every call nor retry a failed import repeatedly.
_scanner: Any = None
_unavailable_reason: str | None = None


def _load_scanner() -> Any:
    """Build (once) and return the LLM Guard PromptInjection scanner.

    Raises ``ImportError`` if LLM Guard is not installed; the caller turns that
    into a graceful :class:`Skipped`.
    """

    global _scanner
    if _scanner is not None:
        return _scanner

    from llm_guard.input_scanners import PromptInjection  # lazy, optional dep

    try:
        from llm_guard.input_scanners.prompt_injection import MatchType

        _scanner = PromptInjection(threshold=_THRESHOLD, match_type=MatchType.FULL)
    except Exception:
        # Older/newer releases may not expose MatchType; fall back to defaults.
        _scanner = PromptInjection(threshold=_THRESHOLD)

    return _scanner


def defend(user_input: str, system_prompt: str) -> tuple[Outcome, dict[str, Any]]:
    """Scan the input with LLM Guard; block if it flags an injection."""

    global _unavailable_reason

    if _unavailable_reason is not None:
        return Skipped(reason=_unavailable_reason), {
            "defense": "llm_guard",
            "available": False,
            "reason": _unavailable_reason,
        }

    try:
        scanner = _load_scanner()
    except ImportError:
        _unavailable_reason = (
            "llm-guard is not installed; run `make install-guard` "
            "(or `pip install -r requirements-guard.txt`) to enable it"
        )
        return Skipped(reason=_unavailable_reason), {
            "defense": "llm_guard",
            "available": False,
            "reason": _unavailable_reason,
        }

    sanitized, is_valid, risk_score = scanner.scan(user_input)

    if not is_valid:
        outcome: Outcome = Block(
            response=(
                "I can't help with that request. It was flagged as a possible "
                "prompt-injection attempt, so I'll keep to my original "
                "instructions."
            ),
            reason=f"LLM Guard flagged the input (risk score {risk_score})",
        )
        meta: dict[str, Any] = {
            "defense": "llm_guard",
            "available": True,
            "action": "block",
            "risk_score": risk_score,
            "threshold": _THRESHOLD,
        }
        return outcome, meta

    outcome = EffectivePrompt(system_prompt=system_prompt, user_input=sanitized)
    meta = {
        "defense": "llm_guard",
        "available": True,
        "action": "allow",
        "risk_score": risk_score,
        "threshold": _THRESHOLD,
    }
    return outcome, meta


__all__ = ["defend"]
