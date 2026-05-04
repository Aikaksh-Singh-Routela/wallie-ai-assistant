"""Shared LLM provider protocol.

Every provider adapter exposes a single async streaming method. Orchestrator and
vision loop only talk to this interface, so adding a new backend is a single file.
"""
from __future__ import annotations

from typing import Any, AsyncIterator, Protocol


class LLMError(RuntimeError):
    """Raised when a provider fails to produce a stream."""


class LLMProvider(Protocol):
    name: str
    model: str
    supports_vision: bool

    async def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.85,
        top_p: float = 0.95,
        max_tokens: int = 500,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
    ) -> AsyncIterator[str]:
        """Yield text tokens as they arrive. Stops when the provider closes the stream."""
        ...

    async def aclose(self) -> None:
        """Release any long-lived resources (HTTP clients, etc.)."""
        ...
