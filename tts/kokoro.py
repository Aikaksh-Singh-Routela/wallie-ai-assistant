"""Kokoro TTS — local, high-quality neural TTS (free, CPU or GPU).

A modern open-weight voice that sounds far better than Piper while still running
locally with no API key. Output is 24 kHz mono PCM16, streamed per segment.
"""
from __future__ import annotations

import asyncio
from typing import AsyncIterator

from .base import TTSError, TTSProvider

_SAMPLE_RATE = 24000  # Kokoro generates at 24 kHz


class KokoroTTS(TTSProvider):
    name = "kokoro"

    def __init__(
        self,
        *,
        voice: str = "af_heart",
        lang_code: str = "a",
        speed: float = 1.0,
    ) -> None:
        try:
            from kokoro import KPipeline  # type: ignore
        except ImportError as e:  # pragma: no cover - optional dep
            raise TTSError(
                "kokoro: not installed. Install with: pip install kokoro soundfile"
            ) from e

        try:
            self._pipeline = KPipeline(lang_code=lang_code)
        except Exception as e:
            raise TTSError(f"kokoro: failed to init pipeline (lang_code={lang_code!r}): {e}") from e

        self._voice = voice or "af_heart"
        self._speed = float(speed) if speed else 1.0
        self.sample_rate = _SAMPLE_RATE
        self.channels = 1

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        text = text.strip()
        if not text:
            return
        loop = asyncio.get_event_loop()
        chunks = await loop.run_in_executor(None, self._synthesize_blocking, text)
        for c in chunks:
            if c:
                yield c

    def _synthesize_blocking(self, text: str) -> list[bytes]:
        import numpy as np

        out: list[bytes] = []
        try:
            for result in self._pipeline(text, voice=self._voice, speed=self._speed):
                # KPipeline yields (graphemes, phonemes, audio); audio is a float
                # tensor/array in [-1, 1]. Support tuple or object forms.
                audio = result[2] if isinstance(result, (tuple, list)) else getattr(result, "audio", None)
                if audio is None:
                    continue
                arr = np.asarray(audio, dtype=np.float32)
                if arr.size == 0:
                    continue
                pcm = np.clip(arr, -1.0, 1.0)
                out.append((pcm * 32767.0).astype(np.int16).tobytes())
        except Exception as e:
            raise TTSError(f"kokoro: synthesis failed: {e}") from e
        return out

    async def aclose(self) -> None:
        self._pipeline = None  # type: ignore
