"""Toy assistant and prompt templates for the prompt-injection resilience lab.

This package holds the lab's OWN minimal "protected assistant" -- a harmless,
fictional FAQ support bot whose hidden system prompt carries the canary marker
used to test leak resistance -- plus the prompt templates for the attacker and
judge nodes. Everything is self-contained and defensive: it exercises only the
lab's isolated contour and names no real product or service as a target.

Layout:
    * :mod:`prompts.toy_app` -- the assistant's identity, FAQ knowledge, the
      canary, the baseline system prompt, and a shared leak-detection helper.
    * :mod:`prompts.attacker` -- template for varying a failed probe within its
      class during the bounded retry loop.
    * :mod:`prompts.judge` -- template for the LLM-as-judge that scores an
      exchange into a :class:`~graph.state.Verdict`.

Convenience re-exports let callers do ``from prompts import base_system_prompt``.
"""

from prompts.attacker import (
    ATTACKER_SYSTEM_PROMPT,
    attacker_messages,
    attacker_user_prompt,
)
from prompts.judge import (
    BREACH_CRITERIA,
    JUDGE_SYSTEM_PROMPT,
    judge_messages,
    judge_user_prompt,
)
from prompts.toy_app import (
    CANARY,
    FAQ,
    IDENTITY,
    PRODUCT_NAME,
    SECRET_NOTE_BASELINE,
    TASK,
    base_system_prompt,
    compose_system_prompt,
    contains_canary,
    faq_block,
)

__all__ = [
    # toy app
    "CANARY",
    "PRODUCT_NAME",
    "IDENTITY",
    "TASK",
    "FAQ",
    "faq_block",
    "SECRET_NOTE_BASELINE",
    "compose_system_prompt",
    "base_system_prompt",
    "contains_canary",
    # attacker
    "ATTACKER_SYSTEM_PROMPT",
    "attacker_user_prompt",
    "attacker_messages",
    # judge
    "BREACH_CRITERIA",
    "JUDGE_SYSTEM_PROMPT",
    "judge_user_prompt",
    "judge_messages",
]
