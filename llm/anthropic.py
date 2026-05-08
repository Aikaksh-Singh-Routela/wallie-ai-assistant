"""Anthropic Claude provider."""
from __future__ import annotations

import base64
from typing import Any, AsyncIterator

from anthropic import AsyncAnthropic

from .base import LLMError, LLMProvider


class AnthropicProvider(LLMProvider):
    def __init__(self, *, model: str, api_key: str, supports_vision: bool = True) -> None:
        if not api_key:
            raise LLMError("anthropic: missing API key")
        self.name = "anthropic"
        self.model = model
        self.supports_vision = supports_vision
        self._client = AsyncAnthropic(api_key=api_key)

    async def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.85,
        top_p: float = 0.95,
        max_tokens: int = 500,
        presence_penalty: float = 0.0,  # unsupported by anthropic; accepted for signature parity
        frequency_penalty: float = 0.0,  # same
    ) -> AsyncIterator[str]:
        system_prompt, rest = self._split_system(messages)
        try:
            async with self._client.messages.stream(
                model=self.model,
                system=system_prompt,
                messages=[self._encode_message(m) for m in rest],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            ) as stream:
                async for text in stream.text_stream:
                    if text:
                        yield text
        except Exception as e:
            raise LLMError(f"anthropic request failed: {e}") from e

    async def aclose(self) -> None:
        await self._client.close()

    # ----- helpers -----
    @staticmethod
    def _split_system(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
        system = ""
        rest: list[dict[str, Any]] = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"] if isinstance(m["content"], str) else ""
            else:
                rest.append(m)
        return system, rest

    def _encode_message(self, m: dict[str, Any]) -> dict[str, Any]:
        content = m.get("content")
        if isinstance(content, list):
            blocks: list[dict[str, Any]] = []
            for b in content:
                if b.get("type") == "text":
                    blocks.append({"type": "text", "text": b.get("text", "")})
                elif b.get("type") == "image":
                    b64 = base64.b64encode(b["data"]).decode("ascii")
                    blocks.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": b.get("mime", "image/jpeg"),
                                "data": b64,
                            },
                        }
                    )
            return {"role": m["role"], "content": blocks}
        return {"role": m["role"], "content": content or ""}
