"""TTS provider protocol."""
from __future__ import annotations

from typing import AsyncIterator, Protocol


class TTSError(RuntimeError):
    pass


class TTSProvider(Protocol):
    name: str
    sample_rate: int  # PCM sample rate of the bytes returned
    channels: int = 1

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        ...

    async def aclose(self) -> None:
        ...
