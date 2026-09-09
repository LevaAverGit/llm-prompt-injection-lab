"""Chat-model client used by the harness nodes.

The three harness nodes (attacker, app, judge) all need to talk to a chat model.
Rather than let each node import a concrete backend, they depend on the small
:class:`ChatClient` protocol below and receive a concrete client through the
LangGraph run config. This keeps the graph backend-agnostic: the real runs use
:class:`OllamaChatClient` (local ``gemma3:latest`` via Ollama), while the offline
runner and the tests can inject any object that implements ``complete``.

Everything is aimed at the lab's OWN isolated toy assistant; the client is just
the transport that carries a node's prompt to the local model and back.
"""

from __future__ import annotations

from typing import List, Optional, Protocol, Sequence, Tuple, runtime_checkable

# A single chat message as a ``(role, content)`` pair. Roles follow the prompt
# modules' convention ("system"/"user"/"assistant"); the Ollama client maps them
# onto LangChain's message types.
Message = Tuple[str, str]

# Default local model. Chosen because it is small, key-free, and reproducible
# under a fixed seed / zero temperature.
DEFAULT_MODEL = "gemma3:latest"


@runtime_checkable
class ChatClient(Protocol):
    """Minimal chat interface the harness nodes rely on.

    Implementations must be deterministic-friendly: given the same messages they
    should return the same text whenever the backend allows it (temperature 0,
    fixed seed). ``json_mode`` asks the backend to emit a single JSON object,
    used by the judge node; backends that cannot enforce it may ignore the flag,
    since the judge parses defensively either way.
    """

    def complete(self, messages: Sequence[Message], *, json_mode: bool = False) -> str:
        ...


# Map the prompt modules' role names onto the role tags LangChain understands.
_ROLE_ALIASES = {
    "system": "system",
    "user": "human",
    "human": "human",
    "assistant": "ai",
    "ai": "ai",
}


def _to_langchain(messages: Sequence[Message]) -> List[Tuple[str, str]]:
    """Normalize ``(role, content)`` pairs into LangChain tuple messages."""

    return [(_ROLE_ALIASES.get(role, "human"), content) for role, content in messages]


class OllamaChatClient:
    """A :class:`ChatClient` backed by a local Ollama model via LangChain.

    Two underlying chat models are built lazily and cached: one plain and one in
    JSON mode (Ollama's ``format="json"``), so the judge can request structured
    output without reconfiguring the model on every call. Determinism is pursued
    with ``temperature=0`` and a fixed ``seed``; the local model is still not
    guaranteed bit-for-bit reproducible, which is why the runner caches replies.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        temperature: float = 0.0,
        seed: Optional[int] = 0,
        base_url: Optional[str] = None,
        num_ctx: Optional[int] = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.seed = seed
        self.base_url = base_url
        self.num_ctx = num_ctx
        # Cache: json_mode flag -> built ChatOllama instance.
        self._models: dict = {}

    def _build(self, json_mode: bool):
        """Construct (once) a ChatOllama configured for the requested mode."""

        from langchain_ollama import ChatOllama  # local, optional at import time

        kwargs = {"model": self.model, "temperature": self.temperature}
        if self.seed is not None:
            kwargs["seed"] = self.seed
        if self.base_url is not None:
            kwargs["base_url"] = self.base_url
        if self.num_ctx is not None:
            kwargs["num_ctx"] = self.num_ctx
        if json_mode:
            kwargs["format"] = "json"
        return ChatOllama(**kwargs)

    def _model(self, json_mode: bool):
        if json_mode not in self._models:
            self._models[json_mode] = self._build(json_mode)
        return self._models[json_mode]

    def complete(self, messages: Sequence[Message], *, json_mode: bool = False) -> str:
        """Send ``messages`` to the model and return the reply text."""

        result = self._model(json_mode).invoke(_to_langchain(messages))
        content = getattr(result, "content", result)
        if isinstance(content, list):
            # Some backends return content parts; join their text.
            parts = [
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            ]
            return "".join(parts)
        return str(content)


__all__ = ["Message", "ChatClient", "OllamaChatClient", "DEFAULT_MODEL"]
