"""Periodic hearing loop — captures system audio windows, transcribes speech,
measures loudness, and emits HearingEvents.

Mirrors vision.vision_loop: a background task that turns a raw stream (audio)
into discrete, meaningful events the orchestrator can fuse with vision. Silence
is skipped so Wallie only "hears" when there's something to hear.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from loguru import logger

from .capture import SystemAudioCapture


@dataclass
class HearingEvent:
    transcript: str            # what was said (may be "" for non-speech sound)
    loudness: float            # RMS 0..1 of the window
    has_speech: bool
    captured_at: float = field(default_factory=time.time)


class HearingLoop:
    def __init__(self, cfg, out_queue: "asyncio.Queue[HearingEvent]") -> None:
        self._cfg = cfg
        self._queue = out_queue
        self._capture = SystemAudioCapture(samplerate=16000)
        self._model = None
        self._task: Optional[asyncio.Task] = None

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name="hearing-loop")
            logger.info(
                "hearing: loop started (window={}s, model={}, silence<{})".format(
                    self._cfg.window_sec, self._cfg.model_size, self._cfg.silence_threshold
                )
            )

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._capture.close()

    # --- internal ---
    async def _run(self) -> None:
        loop = asyncio.get_event_loop()
        try:
            from faster_whisper import WhisperModel
            self._model = await loop.run_in_executor(
                None,
                lambda: WhisperModel(self._cfg.model_size, device="cpu", compute_type="int8"),
            )
            self._capture.open()
            window = self._cfg.window_sec
            silence = self._cfg.silence_threshold
            while True:
                audio = await loop.run_in_executor(None, self._capture.read, window)
                rms = float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0.0
                if rms < silence:
                    continue  # nothing worth hearing
                text = await loop.run_in_executor(None, self._transcribe, audio)
                if text or rms >= self._cfg.sound_event_threshold:
                    ev = HearingEvent(transcript=text, loudness=rms, has_speech=bool(text))
                    self._enqueue(ev)
                    logger.info(f"hear> [{rms:.2f}] {text[:90]}" if text else f"hear> [{rms:.2f}] (sound)")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"hearing: loop crashed: {e}")

    def _transcribe(self, audio: "np.ndarray") -> str:
        lang = self._cfg.language or None
        segs, _info = self._model.transcribe(
            audio, language=lang, vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=400),
        )
        return " ".join(s.text.strip() for s in segs).strip()

    def _enqueue(self, ev: HearingEvent) -> None:
        try:
            self._queue.put_nowait(ev)
        except asyncio.QueueFull:
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                self._queue.put_nowait(ev)
            except asyncio.QueueFull:
                pass
