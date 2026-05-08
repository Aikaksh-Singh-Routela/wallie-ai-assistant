"""Factory — resolves TTSConfig + Secrets to a concrete TTSProvider."""
from __future__ import annotations

from config import Secrets, TTSConfig

from .base import TTSError, TTSProvider


def _missing_pkg(provider: str, pkg: str) -> TTSError:
    return TTSError(
        f"{provider} provider selected but '{pkg}' is not installed. "
        f"Install with: pip install {pkg}"
    )


def build_tts(cfg: TTSConfig, secrets: Secrets) -> TTSProvider:
    if cfg.provider == "fish":
        try:
            from .fish import FishTTS
        except ModuleNotFoundError as e:
            raise _missing_pkg("fish", "httpx") from e
        return FishTTS(
            api_key=secrets.fish_api_key,
            voice_id=cfg.voice_id,
            sample_rate=cfg.sample_rate,
            latency_mode=cfg.fish_latency_mode,
        )

    if cfg.provider == "elevenlabs":
        try:
            from .elevenlabs import ElevenLabsTTS
        except ModuleNotFoundError as e:
            raise _missing_pkg("elevenlabs", "httpx") from e
        return ElevenLabsTTS(
            api_key=secrets.elevenlabs_api_key,
            voice_id=cfg.voice_id,
            sample_rate=cfg.sample_rate,
            model_id=cfg.el_model_id,
            stability=cfg.el_stability,
            similarity_boost=cfg.el_similarity_boost,
            style=cfg.el_style,
        )

    if cfg.provider == "piper":
        try:
            from .piper import PiperTTS
        except ModuleNotFoundError as e:
            raise _missing_pkg("piper", "piper-tts") from e
        return PiperTTS(
            model_path=cfg.piper_model_path,
            length_scale=cfg.piper_length_scale,
        )

    raise TTSError(f"Unknown TTS provider: {cfg.provider}")
