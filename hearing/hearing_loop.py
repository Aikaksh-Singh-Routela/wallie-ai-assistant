"""Periodic hearing loop — captures system audio windows, transcribes speech,
measures loudness, and emits HearingEvents.

Mirrors vision.vision_loop: a background task that turns a raw stream (audio)
into discrete, meaningful events the orchestrator can fuse with vision. Silence
is skipped so Wallie only "hears" when there's something to hear.
"""
from __future__ import annotations

import asyncio
import glob
import os
import site
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from loguru import logger

from .capture import SystemAudioCapture


def _register_cuda_dll_dirs() -> None:
    """Put the pip-installed NVIDIA cuBLAS/cuDNN DLLs on Windows' DLL search path so
    CTranslate2 (faster-whisper) can load them. The wheels drop their DLLs under
    site-packages/nvidia/*/bin, which isn't searched by default — without this the
    GPU path dies at first inference with 'cublas64_12.dll not found'."""
    if not hasattr(os, "add_dll_directory"):
        return
    roots = list(site.getsitepackages())
    sp = getattr(site, "getusersitepackages", lambda: None)()
    if sp:
        roots.append(sp)
    for root in roots:
        for bindir in glob.glob(os.path.join(root, "nvidia", "*", "bin")):
            try:
                os.add_dll_directory(bindir)
            except OSError:
                pass


def _cudnn_present() -> bool:
    """True only if the cuDNN DLLs are actually installed. CTranslate2's CUDA path
    needs them; attempting CUDA without cuDNN can HANG model load, so we gate on this."""
    roots = list(site.getsitepackages())
    sp = getattr(site, "getusersitepackages", lambda: None)()
    if sp:
        roots.append(sp)
    for root in roots:
        if glob.glob(os.path.join(root, "nvidia", "cudnn", "bin", "cudnn*.dll")):
            return True
    return False


@dataclass
class HearingEvent:
    transcript: str            # what was said (may be "" for non-speech sound)
    loudness: float            # RMS 0..1 of the window
    has_speech: bool
    sound_type: str = "speech"  # speech | music | sound | quiet
    descriptor: str = ""        # for music/sound: e.g. "upbeat, energetic, bright"
    captured_at: float = field(default_factory=time.time)


class HearingLoop:
    def __init__(self, cfg, out_queue: "asyncio.Queue[HearingEvent]",
                 is_self_speaking: Optional[Callable[[], bool]] = None,
                 is_self_echo: Optional[Callable[[str], bool]] = None) -> None:
        self._cfg = cfg
        self._queue = out_queue
        self._capture = SystemAudioCapture(samplerate=16000)
        self._model = None
        self._task: Optional[asyncio.Task] = None
        # Returns True if Wallie is currently speaking — those windows are skipped
        # so Wallie never transcribes/reacts to its own TTS (timing-based guard).
        self._is_self_speaking = is_self_speaking
        # Returns True if a transcript matches something Wallie recently SAID — the
        # definitive self-echo guard, immune to capture lag (content, not timing).
        self._is_self_echo = is_self_echo

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
            self._model = await loop.run_in_executor(None, self._load_model)
            self._capture.open()
            window = self._cfg.window_sec
            silence = self._cfg.silence_threshold
            from .audio_analysis import analyze_window, music_character
            # Let the ring buffer fill once before the first read.
            await asyncio.sleep(window)
            while True:
                # The capture thread drains the device non-stop, so we just grab the
                # most recent window — always current, never a stale/overflowed chunk.
                audio = self._capture.latest(window)
                if self._is_self_speaking is not None and self._is_self_speaking():
                    await asyncio.sleep(window)
                    continue  # Wallie was speaking — don't hear ourselves
                rms = float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0.0
                if rms < silence:
                    await asyncio.sleep(window)
                    continue  # nothing worth hearing
                text = await loop.run_in_executor(None, self._transcribe, audio)
                # Guard against Whisper hallucinating a stray word over music/noise.
                has_speech = bool(text) and len(text.split()) >= 2
                # Definitive self-echo guard: if what we "heard" matches something
                # Wallie just said, it's our own voice bleeding through the loopback —
                # drop it. Content-based, so it works even if capture lags seconds behind.
                if has_speech and self._is_self_echo is not None and self._is_self_echo(text):
                    logger.debug(f"hearing: muted self-echo — {text[:60]!r}")
                    await asyncio.sleep(window)
                    continue
                # Analyze the MUSICAL character over a longer slice (more stable tempo/key)
                # than the STT window, pulled straight from the rolling buffer.
                music_audio = self._capture.latest(min(window * 1.8, 9.0))
                sound_type, descriptor = analyze_window(
                    music_audio, 16000, has_speech=has_speech, silence=silence
                )
                if sound_type == "quiet":
                    await asyncio.sleep(window)
                    continue
                # Even when there are lyrics (→ speech), tag the musical vibe so Wallie
                # knows it's a *song* and can react to the mood, not just the words.
                vibe = ""
                if sound_type == "speech":
                    vibe = music_character(music_audio, 16000)
                ev = HearingEvent(
                    transcript=text if has_speech else "",
                    loudness=rms, has_speech=has_speech,
                    sound_type=sound_type, descriptor=(vibe if sound_type == "speech" else descriptor),
                )
                self._enqueue(ev)
                if sound_type == "speech":
                    tag = f" ♪({vibe})" if vibe else ""
                    logger.info(f"hear> [{rms:.2f}]{tag} {text[:90]}")
                elif sound_type == "music":
                    logger.info(f"hear> [{rms:.2f}] ♪ music: {descriptor}")
                else:
                    logger.info(f"hear> [{rms:.2f}] (sound: {descriptor})")
                await asyncio.sleep(window)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"hearing: loop crashed: {e}")

    def _load_model(self):
        """Load Whisper on the best available device — GPU (fast, lets you run a
        bigger/more accurate model) with a clean fallback to CPU int8.

        The GPU path is *probed* with a tiny inference: CTranslate2 builds the model
        handle lazily, so a missing cublas/cudnn DLL only blows up on the first real
        transcribe. We trigger that here and fall back to CPU instead of crashing the
        loop mid-session."""
        from faster_whisper import WhisperModel
        size = self._cfg.model_size
        _register_cuda_dll_dirs()

        # Only attempt CUDA when the GPU AND cuDNN are both genuinely available —
        # a half-installed CUDA stack can HANG model load and silently kill hearing.
        use_cuda = False
        if _cudnn_present():
            try:
                import ctranslate2
                use_cuda = ctranslate2.get_cuda_device_count() > 0
            except Exception:
                use_cuda = False

        if use_cuda:
            try:
                m = WhisperModel(size, device="cuda", compute_type="float16")
                list(m.transcribe(np.zeros(16000, dtype="float32"))[0])  # force CUDA init
                logger.info(f"hearing: Whisper '{size}' on CUDA (float16)")
                return m
            except Exception as e:
                logger.warning(f"hearing: CUDA load failed ({str(e)[:80]}) — using CPU")

        m = WhisperModel(size, device="cpu", compute_type="int8")
        logger.info(f"hearing: Whisper '{size}' on CPU (int8)")
        return m

    def _transcribe(self, audio: "np.ndarray") -> str:
        lang = self._cfg.language or None
        # Sung vocals sit low under the instrumental — lift the window toward full
        # scale so Whisper gets a strong signal instead of a faint mumble.
        if audio.size:
            peak = float(np.abs(audio).max())
            if 0.0 < peak < 0.9:
                audio = (audio * (0.9 / peak)).astype("float32")
        segs, _info = self._model.transcribe(
            audio, language=lang,
            beam_size=5, best_of=5,
            temperature=[0.0, 0.2, 0.4, 0.6],   # fall back to softer decoding if stuck
            # Each window is independent — don't carry prior text, which makes Whisper
            # loop/hallucinate lyrics over music. Big accuracy win for songs.
            condition_on_previous_text=False,
            # Drop hallucinated garbage that music/noise provokes.
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=300, threshold=0.4),
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
