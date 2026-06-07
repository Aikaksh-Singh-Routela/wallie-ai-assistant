"""System-audio capture via WASAPI loopback (soundcard), continuous + lag-free.

A background thread drains the loopback device non-stop into a rolling ring buffer,
so the buffer never overflows while transcription is running. The processing loop
just grabs the most RECENT `window` seconds whenever it wants — always current audio,
no "data discontinuity", no stale/echoed audio. This is Wallie's ear.
"""
from __future__ import annotations

import threading

import numpy as np

try:
    import soundcard as sc
except Exception:  # pragma: no cover - optional dep
    sc = None


class SystemAudioCapture:
    """Continuous loopback capture into a rolling buffer (thread-backed)."""

    def __init__(self, samplerate: int = 16000, channels: int = 1, buffer_sec: float = 10.0) -> None:
        if sc is None:
            raise RuntimeError(
                "soundcard not installed — hearing needs it. Install: pip install soundcard"
            )
        self._sr = samplerate
        self._ch = channels
        self._ring = np.zeros(int(samplerate * buffer_sec), dtype="float32")
        self._lock = threading.Lock()
        self._thread: "threading.Thread | None" = None
        self._running = False

    def open(self) -> None:
        if self._thread is not None:
            return
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, name="audio-capture", daemon=True)
        self._thread.start()

    def _capture_loop(self) -> None:
        spk = sc.default_speaker()
        mic = sc.get_microphone(str(spk.name), include_loopback=True)
        chunk = max(1, int(self._sr * 0.25))  # drain in 250 ms slices
        with mic.recorder(samplerate=self._sr, channels=self._ch, blocksize=chunk) as rec:
            while self._running:
                try:
                    data = rec.record(numframes=chunk)
                except Exception:
                    break
                data = data.flatten().astype("float32")
                n = data.shape[0]
                if n == 0:
                    continue
                with self._lock:
                    if n >= self._ring.shape[0]:
                        self._ring[:] = data[-self._ring.shape[0]:]
                    else:
                        self._ring[:-n] = self._ring[n:]
                        self._ring[-n:] = data

    def latest(self, seconds: float) -> "np.ndarray":
        """Most recent `seconds` of audio from the ring (always current)."""
        k = int(self._sr * seconds)
        with self._lock:
            return self._ring[-k:].copy()

    def close(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    @property
    def sample_rate(self) -> int:
        return self._sr
