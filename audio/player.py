"""Low-latency PCM audio player.

The TTS pipeline feeds PCM16-LE chunks into a byte ring buffer. A sounddevice
OutputStream pulls from it in its audio callback, so playback starts as soon as
the first TTS chunk arrives. interrupt() empties the queue instantly for barge-in.

Alignment safety
----------------
PCM16 samples are 2 bytes each. If a single ``write()`` call ever leaves an odd
number of bytes in the buffer, every subsequent sample is read split between
two adjacent samples — which produces sustained, ear-melting static that
*never recovers* until the buffer is flushed. Real-world TTS responses can
end on an odd byte if the HTTP body is truncated or the server returns a
malformed payload. We therefore:

  * Carry any trailing odd byte forward into the next ``write()`` so chunks
    of one stream stitch together cleanly.
  * Provide ``boundary()`` to drop that carry-over between independent TTS
    streams (the leftover from sentence N must NOT be mixed with sentence N+1).
  * Provide ``reset()`` for explicit panic recovery (dashboard button).
"""
from __future__ import annotations

import asyncio
import threading
from collections import deque
from typing import Optional

import numpy as np
import sounddevice as sd
from loguru import logger


class AudioPlayer:
    def __init__(
        self,
        *,
        sample_rate: int = 24000,
        channels: int = 1,
        blocksize: int = 960,  # ~20ms at 48kHz, ~40ms at 24kHz
        device: Optional[int | str] = None,
    ) -> None:
        self._sr = sample_rate
        self._channels = channels
        self._blocksize = blocksize
        self._device = device

        # Byte-level ring buffer. PCM16 = 2 bytes per sample per channel.
        self._buf = bytearray()
        self._lock = threading.Lock()
        self._finished_event = asyncio.Event()
        self._finished_event.set()
        # Carry-over odd byte for alignment-safe writes (see module docstring).
        self._pending_odd: Optional[int] = None

        self._stream: Optional[sd.RawOutputStream] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    # ----- lifecycle -----
    def start(self) -> None:
        if self._stream is not None:
            return
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = asyncio.get_event_loop()
        self._stream = sd.RawOutputStream(
            samplerate=self._sr,
            channels=self._channels,
            dtype="int16",
            blocksize=self._blocksize,
            device=self._device,
            callback=self._callback,
        )
        self._stream.start()
        logger.info(f"audio: playing at {self._sr} Hz, {self._channels}ch, block={self._blocksize}")

    def close(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                logger.warning(f"audio stream close: {e}")
            self._stream = None

    # ----- writing -----
    async def write(self, pcm: bytes) -> None:
        """Enqueue PCM16-LE bytes for playback. Maintains 2-byte alignment so
        a truncated TTS response can't permanently desync the stream."""
        if not pcm:
            return
        # Stitch in any odd byte left over from the previous write.
        if self._pending_odd is not None:
            pcm = bytes((self._pending_odd,)) + pcm
            self._pending_odd = None
        # If we still have an odd-byte tail, hold it for the next call.
        if len(pcm) & 1:
            self._pending_odd = pcm[-1]
            pcm = pcm[:-1]
        if not pcm:
            return
        with self._lock:
            self._buf.extend(pcm)
        self._finished_event.clear()
        await asyncio.sleep(0)

    def boundary(self) -> None:
        """Mark the end of an independent TTS stream (e.g. a sentence). Drops
        any carry-over odd byte so the next stream starts cleanly aligned."""
        if self._pending_odd is not None:
            logger.debug(f"audio: dropping leftover odd byte at sentence boundary")
            self._pending_odd = None

    def reset(self) -> None:
        """Panic recovery: clear the buffer AND any alignment carry-over."""
        with self._lock:
            dropped = len(self._buf)
            self._buf.clear()
        self._pending_odd = None
        if self._loop is not None:
            try:
                self._loop.call_soon_threadsafe(self._finished_event.set)
            except RuntimeError:
                pass
        logger.info(f"audio: hard reset, dropped {dropped} bytes")

    def interrupt(self) -> None:
        """Stop playback instantly by discarding queued audio."""
        with self._lock:
            dropped = len(self._buf)
            self._buf.clear()
        self._pending_odd = None
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._finished_event.set)
        if dropped:
            logger.info(f"audio: interrupt, dropped {dropped} bytes")

    async def wait_drained(self) -> None:
        """Resolve when the queue is empty."""
        await self._finished_event.wait()

    def seconds_queued(self) -> float:
        with self._lock:
            return len(self._buf) / (self._sr * self._channels * 2)

    # ----- audio callback -----
    def _callback(self, outdata, frames: int, time_info, status) -> None:  # noqa: ARG002
        if status:
            # xruns are normal under load; log once at debug level.
            logger.debug(f"audio status: {status}")
        needed = frames * self._channels * 2
        with self._lock:
            available = len(self._buf)
            take = min(needed, available)
            if take:
                chunk = bytes(self._buf[:take])
                del self._buf[:take]
            else:
                chunk = b""
            empty_after = len(self._buf) == 0

        if take < needed:
            chunk = chunk + b"\x00" * (needed - take)
        outdata[: len(chunk)] = chunk

        if empty_after and self._loop is not None:
            # Notify the async world that we drained.
            try:
                self._loop.call_soon_threadsafe(self._finished_event.set)
            except RuntimeError:
                pass


def list_output_devices() -> list[dict]:
    """Utility for the dashboard: enumerate selectable output devices."""
    out = []
    for i, d in enumerate(sd.query_devices()):
        if d.get("max_output_channels", 0) > 0:
            out.append(
                {
                    "index": i,
                    "name": d.get("name", ""),
                    "hostapi": d.get("hostapi", 0),
                    "default_samplerate": d.get("default_samplerate", 0),
                }
            )
    return out
