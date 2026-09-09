"""LangGraph assembly and entry point for the attacker -> app -> judge loop.

:func:`build_harness` compiles the stateful graph once; :func:`run_attack` runs a
single attack sample against a single defense, looping up to the attempt budget
until the defense is breached. The graph carries the shared
:class:`~graph.state.GraphState` as its channel state, so the object handed back
is exactly the contract every other component speaks.

Graph shape::

    START -> attacker -> app -> (judge | END)
    judge -> (END | retry)
    retry -> attacker

The chat client is passed per-run through the config rather than baked into the
graph, so the same compiled graph serves the live Ollama model and any injected
stand-in. Everything here targets the lab's OWN isolated toy assistant.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from langgraph.graph import END, START, StateGraph

from graph.client import ChatClient, OllamaChatClient
from graph.nodes import (
    app_node,
    attacker_node,
    judge_node,
    retry_node,
    route_after_app,
    route_after_judge,
)
from graph.state import Attack, GraphState


def build_harness():
    """Build and compile the attacker -> app -> judge graph over ``GraphState``."""

    builder = StateGraph(GraphState)
    builder.add_node("attacker", attacker_node)
    builder.add_node("app", app_node)
    builder.add_node("judge", judge_node)
    builder.add_node("retry", retry_node)

    builder.add_edge(START, "attacker")
    builder.add_edge("attacker", "app")
    builder.add_conditional_edges(
        "app", route_after_app, {"judge": "judge", "end": END}
    )
    builder.add_conditional_edges(
        "judge", route_after_judge, {"retry": "retry", "end": END}
    )
    builder.add_edge("retry", "attacker")

    return builder.compile()


@lru_cache(maxsize=1)
def _default_harness():
    """Compile the harness once and reuse it across :func:`run_attack` calls."""

    return build_harness()


def run_attack(
    attack: Attack,
    defense_name: str,
    *,
    max_attempts: int = 1,
    client: Optional[ChatClient] = None,
    harness=None,
) -> GraphState:
    """Run one attack against one defense and return the final ``GraphState``.

    Args:
        attack: The corpus sample to probe with.
        defense_name: Which defense contour to activate (``none``,
            ``system_prompt``, ``input_filter``, ``both``, ``llm_guard``).
        max_attempts: Attempt budget. With more than one, a probe that fails to
            breach is rephrased into a same-class variation and retried until it
            breaches or the budget is exhausted.
        client: The chat client the nodes use. Defaults to a local
            :class:`~graph.client.OllamaChatClient` (``gemma3:latest``).
        harness: A pre-compiled graph to reuse; defaults to a shared instance.

    Returns:
        The terminal :class:`GraphState`, whose ``history`` holds every attempt
        and whose ``verdict`` / ``breached`` report the outcome.
    """

    graph = harness or _default_harness()
    chat = client or OllamaChatClient()
    initial = GraphState(
        attack=attack, defense_name=defense_name, max_attempts=max_attempts
    )
    # Each attempt costs at most 4 super-steps (attacker, app, judge, retry);
    # give the run enough headroom plus a small buffer.
    recursion_limit = max_attempts * 4 + 6
    result = graph.invoke(
        initial,
        config={
            "configurable": {"client": chat},
            "recursion_limit": recursion_limit,
        },
    )
    if isinstance(result, GraphState):
        return result
    return GraphState.model_validate(result)


__all__ = ["build_harness", "run_attack"]
