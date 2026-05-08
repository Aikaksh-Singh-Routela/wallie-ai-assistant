"""Piper TTS — local CPU-based neural TTS adapter."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, AsyncIterator

from .base import TTSError, TTSProvider


class PiperTTS(TTSProvider):
    name = "piper"

    def __init__(
        self,
        *,
        model_path: str,
        length_scale: float = 1.0,
        noise_scale: float = 0.667,
        noise_w: float = 0.8,
    ) -> None:
        if not model_path:
            raise TTSError("piper: model_path is required (download a .onnx voice file)")
        path = Path(model_path)
        if not path.is_file():
            raise TTSError(f"piper: model file not found: {path}")

        try:
            from piper import PiperVoice  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise TTSError(
                "piper: 'piper-tts' is not installed. Install with: pip install piper-tts"
            ) from e

        try:
            self._voice = PiperVoice.load(str(path))
        except Exception as e:
            raise TTSError(f"piper: failed to load model {path.name}: {e}") from e

        # Sample rate from voice config, default if missing.
        sr = getattr(getattr(self._voice, "config", None), "sample_rate", None)
        self.sample_rate = int(sr) if sr else 22050
        self.channels = 1
        self._length_scale = length_scale
        self._noise_scale = noise_scale
        self._noise_w = noise_w

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
        out: list[bytes] = []
        try:
            iterator = self._voice.synthesize(
                text,
                length_scale=self._length_scale,
                noise_scale=self._noise_scale,
                noise_w=self._noise_w,
            )
            for chunk in iterator:
                buf: Any = getattr(chunk, "audio_int16_bytes", None)
                if buf is None:
                    if isinstance(chunk, (bytes, bytearray)):
                        buf = bytes(chunk)
                    else:
                        arr = getattr(chunk, "audio_float_array", None)
                        if arr is not None:
                            import numpy as np
                            buf = (arr * 32767).astype(np.int16).tobytes()
                if buf:
                    out.append(bytes(buf))
            return out
        except (AttributeError, TypeError):
            pass
        # Legacy fallback.
        try:
            for chunk in self._voice.synthesize_stream_raw(
                text, length_scale=self._length_scale
            ):
                if chunk:
                    out.append(bytes(chunk))
        except Exception as e:
            raise TTSError(f"piper: synthesis failed: {e}") from e
        return out

    async def aclose(self) -> None:
        self._voice = None  # type: ignore
