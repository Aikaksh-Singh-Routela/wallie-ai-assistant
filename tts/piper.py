"""Piper TTS — local CPU-based neural TTS adapter.

Why this exists
---------------
LLM + cloud TTS adds up to ~$3-8 per streaming hour. For hobbyists, a
zero-recurring-cost path matters more than top-tier voice quality. Piper runs
entirely on CPU, has decent prosody, and pairs well with Llama on Ollama or
Gemini's free tier for a pure-local / pure-free Wallie setup.

Setup
-----
1. ``pip install piper-tts onnxruntime`` (already optional in requirements.txt)
2. Download a voice model (``.onnx`` + ``.onnx.json``):
   ``python scripts/download_piper_voice.py en_US-amy-medium``
3. In the dashboard set TTS provider to ``piper`` and point ``piper_model_path``
   at the downloaded file.
"""
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

        # Sample rate is exposed on the voice config. Use a default if missing.
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
        # Piper is synchronous; offload to a worker thread so we don't block
        # the event loop (which would freeze the rest of the audio pipeline).
        chunks = await loop.run_in_executor(None, self._synthesize_blocking, text)
        for c in chunks:
            if c:
                yield c

    def _synthesize_blocking(self, text: str) -> list[bytes]:
        """Run the actual synthesis. Returns a list of PCM16-LE chunks."""
        # piper-tts API has shifted between releases. Try the modern API first
        # (synthesize() yielding AudioChunk objects with .audio_int16_bytes),
        # then fall back to the legacy synthesize_stream_raw() byte iterator.
        out: list[bytes] = []
        try:
            iterator = self._voice.synthesize(
                text,
                length_scale=self._length_scale,
                noise_scale=self._noise_scale,
                noise_w=self._noise_w,
            )
            for chunk in iterator:
                # Modern API: AudioChunk with .audio_int16_bytes
                buf: Any = getattr(chunk, "audio_int16_bytes", None)
                if buf is None:
                    # Some versions: bytes directly
                    if isinstance(chunk, (bytes, bytearray)):
                        buf = bytes(chunk)
                    else:
                        # Possibly a numpy array or AudioFloat — best-effort convert.
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
        # Voice object holds an ONNX session; let GC handle it.
        self._voice = None  # type: ignore
