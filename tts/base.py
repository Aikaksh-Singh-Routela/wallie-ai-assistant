"""TTS provider protocol.

All adapters return raw PCM16 little-endian bytes at a known sample rate. Keeping
decode out of the pipeline removes a significant source of latency and failure.
"""
from __future__ import annotations

from typing import AsyncIterator, Protocol


class TTSError(RuntimeError):
    pass


class TTSProvider(Protocol):
    name: str
    sample_rate: int  # PCM sample rate of the bytes returned
    channels: int = 1

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        """Yield PCM16-LE chunks for the given text."""
        ...

    async def aclose(self) -> None:
        ...
