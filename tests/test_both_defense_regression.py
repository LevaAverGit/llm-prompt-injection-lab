"""Security regression guard: the hardened ``both`` defense holds every attack.

For each sample in the corpus, running it through the ``both`` contour (input
filter gate followed by the hardened, spotlighting system prompt) and a
deterministic mock assistant must NOT produce a breach. If a change ever weakens
``both`` so an attack reaches the model unspotlighted, the gullible mock obeys it
and the matching case fails -- which is the point of this file.

The tests are provider-free (``provider='mock'``) and fully deterministic, so
they need no Ollama and give the same result on every run.
"""

from __future__ import annotations

import pytest

from defenses import defend
from defenses.base import Block, EffectivePrompt
from defenses.base import INPUT_CLOSE, INPUT_OPEN
from prompts.toy_app import base_system_prompt
from tests.support import BENIGN_REPLY, deterministic_judge, load_corpus, mock_app_respond, run_attack

CORPUS = load_corpus()
CORPUS_IDS = [attack.id for attack in CORPUS]


@pytest.mark.parametrize("attack", CORPUS, ids=CORPUS_IDS)
def test_both_defense_is_not_breached(attack) -> None:
    """The regression guard: every attack class is held by ``both``."""
    verdict = run_attack("both", attack)
    assert verdict is not None, "the 'both' defense never skips"
    assert not verdict.breached, (
        f"'both' was breached by {attack.id} ({attack.attack_class.value}): "
        f"{verdict.rationale}"
    )
    assert not verdict.leaked_system_prompt, f"{attack.id} leaked the system prompt"


@pytest.mark.parametrize("attack", CORPUS, ids=CORPUS_IDS)
def test_both_blocks_or_spotlights_every_attack(attack) -> None:
    """``both`` must either block the probe or spotlight it under hardening.

    A plain passthrough of an attack would be the silent regression this guards
    against, independent of the mock assistant.
    """
    outcome, meta = defend("both", attack.prompt, base_system_prompt())
    assert meta["defense"] == "both"
    if isinstance(outcome, Block):
        assert outcome.response.strip()
    elif isinstance(outcome, EffectivePrompt):
        assert INPUT_OPEN in outcome.system_prompt, "surviving input must be hardened"
        assert INPUT_OPEN in outcome.user_input and INPUT_CLOSE in outcome.user_input
    else:  # pragma: no cover - 'both' never skips
        pytest.fail(f"unexpected outcome type for {attack.id}: {type(outcome).__name__}")


def test_none_baseline_is_breachable() -> None:
    """Sanity: the unprotected baseline IS breached, so the guard is not vacuous.

    If the mock could never be breached, the ``both`` assertions above would be
    meaningless. Under ``none`` (no protection) every corpus attack must break
    through the gullible assistant.
    """
    verdicts = [run_attack("none", attack) for attack in CORPUS]
    assert all(v is not None for v in verdicts)
    breaches = sum(1 for v in verdicts if v.breached)
    assert breaches == len(CORPUS), (
        f"expected the unprotected baseline to be breached by all "
        f"{len(CORPUS)} attacks, saw {breaches}"
    )
    # System-prompt leaks are tracked distinctly and must show up in the baseline.
    assert any(v.leaked_system_prompt for v in verdicts)


def test_mock_leaves_benign_input_alone() -> None:
    """The gullible mock breaches attacks, not harmless questions.

    A benign FAQ question carried through the unprotected baseline must NOT be
    scored as a breach -- otherwise the judge would flag anything at all.
    """
    outcome, _meta = defend("none", "How do I export my notes?", base_system_prompt())
    response = mock_app_respond(outcome)
    assert response == BENIGN_REPLY
    verdict = deterministic_judge(
        CORPUS[0], response
    )  # the attack object only supplies context; scoring is on the reply
    assert not verdict.breached
    assert not verdict.leaked_system_prompt
