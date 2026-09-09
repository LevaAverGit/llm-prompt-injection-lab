"""The judge's raw output is coerced into a Verdict across messy inputs.

Real model output is not always clean JSON: it may be wrapped in prose or
markdown fences, omit the optional field, or be truncated / malformed. The
parser must recover a :class:`Verdict` from the recoverable cases and fail
gracefully (return ``None`` rather than raise) on the rest.
"""

from __future__ import annotations

from graph import Verdict
from tests.support import parse_verdict


def test_parses_clean_json_object() -> None:
    raw = '{"breached": true, "rationale": "obeyed the injection", "leaked_system_prompt": true}'
    verdict = parse_verdict(raw)
    assert isinstance(verdict, Verdict)
    assert verdict.breached is True
    assert verdict.leaked_system_prompt is True
    assert verdict.rationale == "obeyed the injection"


def test_parses_json_embedded_in_prose_and_fences() -> None:
    raw = (
        "Here is my assessment.\n\n"
        "```json\n"
        '{"breached": false, "rationale": "stayed in role", "leaked_system_prompt": false}\n'
        "```\n"
        "Let me know if you need more detail."
    )
    verdict = parse_verdict(raw)
    assert isinstance(verdict, Verdict)
    assert verdict.breached is False
    assert verdict.rationale == "stayed in role"


def test_partial_json_uses_default_for_optional_field() -> None:
    # leaked_system_prompt omitted -> defaults to False.
    raw = '{"breached": false, "rationale": "safe refusal"}'
    verdict = parse_verdict(raw)
    assert isinstance(verdict, Verdict)
    assert verdict.breached is False
    assert verdict.leaked_system_prompt is False


def test_missing_required_field_returns_none() -> None:
    # No "breached" field -> cannot build a valid Verdict.
    assert parse_verdict('{"rationale": "no verdict field"}') is None
    # No "rationale" field.
    assert parse_verdict('{"breached": true}') is None
    # Empty object.
    assert parse_verdict("{}") is None


def test_empty_rationale_is_rejected() -> None:
    # rationale has min_length=1 in the contract.
    assert parse_verdict('{"breached": true, "rationale": ""}') is None


def test_unexpected_field_is_rejected() -> None:
    # The Verdict contract forbids extra fields.
    raw = '{"breached": true, "rationale": "x", "leaked_system_prompt": false, "confidence": 0.9}'
    assert parse_verdict(raw) is None


def test_broken_and_missing_json_returns_none() -> None:
    assert parse_verdict("this is not json at all") is None
    assert parse_verdict("") is None
    # Truncated object (no matching close brace) is unrecoverable.
    assert parse_verdict('{"breached": true, "rationale": "x"') is None
    # A stray opening brace with no valid content.
    assert parse_verdict("{ oops") is None


def test_first_object_is_used_when_several_present() -> None:
    raw = (
        '{"breached": true, "rationale": "first", "leaked_system_prompt": false} '
        'then junk {"breached": false, "rationale": "second"}'
    )
    verdict = parse_verdict(raw)
    assert isinstance(verdict, Verdict)
    assert verdict.rationale == "first"
    assert verdict.breached is True
