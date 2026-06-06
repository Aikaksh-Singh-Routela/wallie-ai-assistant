"""System-audio capture via WASAPI loopback (soundcard).

Captures whatever is playing on the default output device — game audio, video,
music, voice chat — without a virtual cable. Mono, 16 kHz, float32 (what Whisper
wants). This is Wallie's "ear": the auditory counterpart to vision.capture.
"""
from __future__ import annotations

import numpy as np

try:
    import soundcard as sc
except Exception:  # pragma: no cover - optional dep
    sc = None


class SystemAudioCapture:
    """Persistent loopback recorder. Open once, then read() successive windows."""

    def __init__(self, samplerate: int = 16000, channels: int = 1) -> None:
        if sc is None:
            raise RuntimeError(
                "soundcard not installed — hearing needs it. Install: pip install soundcard"
            )
        self._sr = samplerate
        self._ch = channels
        self._rec = None

    def open(self) -> None:
        spk = sc.default_speaker()
        mic = sc.get_microphone(str(spk.name), include_loopback=True)
        self._rec = mic.recorder(samplerate=self._sr, channels=self._ch)
        self._rec.__enter__()

    def read(self, seconds: float) -> "np.ndarray":
        """Block until `seconds` of audio is captured; return mono float32."""
        if self._rec is None:
            raise RuntimeError("capture not open")
        data = self._rec.record(numframes=int(self._sr * seconds))
        return data.flatten().astype("float32")

    def close(self) -> None:
        if self._rec is not None:
            try:
                self._rec.__exit__(None, None, None)
            except Exception:
                pass
            self._rec = None

    @property
    def sample_rate(self) -> int:
        return self._sr
