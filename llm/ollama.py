"""Ollama provider — local LLM via Ollama's OpenAI-compatible endpoint."""
from __future__ import annotations

from typing import Any, AsyncIterator, Optional

import httpx
from loguru import logger

from .base import LLMError, LLMProvider
from .openai_compat import OpenAICompatProvider


class OllamaProvider(LLMProvider):
    def __init__(
        self,
        *,
        model: str,
        base_url: str = "http://localhost:11434",
        keep_alive: str = "5m",
        supports_vision: bool = False,
    ) -> None:
        self.name = "ollama"
        self.model = model
        self.supports_vision = supports_vision

        # Strip trailing /v1 — we add it ourselves.
        base = base_url.rstrip("/")
        if base.endswith("/v1"):
            base = base[: -len("/v1")]
        self._native_base = base

        # Delegate to OpenAICompatProvider. Placeholder key — Ollama ignores it.
        self._inner = OpenAICompatProvider(
            name="ollama",
            model=model,
            api_key="ollama",
            base_url=f"{base}/v1",
            supports_vision=supports_vision,
        )
        self._keep_alive = keep_alive
        self._verified = False
        self._warmup_started = False

    async def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.85,
        top_p: float = 0.95,
        max_tokens: int = 150,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
    ) -> AsyncIterator[str]:
        if not self._verified:
            await self._verify_model()
        if not self._warmup_started:
            self._warmup_started = True
            try:
                import asyncio
                asyncio.create_task(self._warmup(), name="ollama-warmup")
            except Exception:
                pass

        async for tok in self._inner.stream(
            messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
        ):
            yield tok

    async def aclose(self) -> None:
        await self._inner.aclose()

    async def _verify_model(self) -> None:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._native_base}/api/tags")
                if resp.status_code != 200:
                    raise LLMError(
                        f"Ollama at {self._native_base} returned HTTP {resp.status_code}. "
                        "Is `ollama serve` running and reachable?"
                    )
                tags = (resp.json() or {}).get("models") or []
        except httpx.ConnectError as e:
            raise LLMError(
                f"Ollama at {self._native_base} unreachable: {e}. "
                "Start it with: ollama serve  (or check your URL/port)."
            ) from e
        except httpx.HTTPError as e:
            logger.warning(f"ollama: pre-flight tag fetch failed: {e}; proceeding")
            self._verified = True
            return

        names = {(t.get("name") or "") for t in tags}
        bare = {n.split(":", 1)[0] for n in names if n}
        wanted_bare = self.model.split(":", 1)[0]
        if self.model not in names and wanted_bare not in bare:
            available = sorted(bare)[:10]
            pretty = ", ".join(available) if available else "(no models pulled)"
            raise LLMError(
                f"Ollama model '{self.model}' is not pulled.\n"
                f"Available locally: {pretty}\n"
                f"Pull it with: ollama pull {self.model}"
            )
        self._verified = True
        logger.info(
            f"ollama: verified, {len(names)} model(s) loaded; using '{self.model}' "
            f"(keep_alive={self._keep_alive})"
        )

    async def _warmup(self) -> None:
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                await client.post(
                    f"{self._native_base}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": "",
                        "stream": False,
                        "keep_alive": self._keep_alive,
                    },
                )
        except Exception as e:
            logger.debug(f"ollama: warmup ping failed (non-fatal): {e}")
