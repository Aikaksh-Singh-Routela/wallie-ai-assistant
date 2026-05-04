"""Fish Audio TTS adapter (streaming PCM).

Uses the HTTP streaming endpoint which yields audio bytes as the model generates
them. Requesting PCM output removes the MP3 decode step from the hot path.
"""
from __future__ import annotations

from typing import AsyncIterator

import httpx

from .base import TTSError, TTSProvider

_ENDPOINT = "https://api.fish.audio/v1/tts"


class FishTTS(TTSProvider):
    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str,
        sample_rate: int = 24000,
        latency_mode: str = "balanced",  # "normal" | "balanced"
    ) -> None:
        if not api_key:
            raise TTSError("fish: missing FISH_API_KEY")
        self.name = "fish"
        self.sample_rate = sample_rate
        self.channels = 1
        self._api_key = api_key
        self._voice_id = voice_id
        self._latency = latency_mode
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=5.0))

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        if not text.strip():
            return
        payload: dict = {
            "text": text,
            "format": "pcm",
            "sample_rate": self.sample_rate,
            "chunk_length": 200,
            "latency": self._latency,
        }
        if self._voice_id:
            payload["reference_id"] = self._voice_id
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "model": "speech-1.6",
        }
        try:
            async with self._client.stream("POST", _ENDPOINT, json=payload, headers=headers) as resp:
                if resp.status_code >= 400:
                    body = await resp.aread()
                    raise TTSError(f"fish {resp.status_code}: {body[:200]!r}")
                async for chunk in resp.aiter_bytes():
                    if chunk:
                        yield chunk
        except httpx.HTTPError as e:
            raise TTSError(f"fish network error: {e}") from e

    async def aclose(self) -> None:
        await self._client.aclose()
