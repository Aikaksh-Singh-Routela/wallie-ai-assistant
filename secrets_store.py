"""Secrets management — UI-editable API keys, written to .env safely.

Security model
--------------
* Keys live in ``BASE_DIR/.env`` (already git-ignored). We never commit them
  to ``config.yaml`` (the user-shareable profile file).
* The dashboard NEVER receives the raw value of a stored key. ``list_secrets``
  returns a masked preview only ("sk-•••abc") so a screenshot can't leak it.
* Writes go through ``python-dotenv``'s ``set_key`` for proper escaping, then
  the file is chmod-restricted to 600 on POSIX so other local users can't read
  it. (Windows ACLs are left to the OS — the dashboard is bound to 127.0.0.1
  by default, so the local machine boundary is the security boundary.)
* Empty values delete the key from .env rather than store an empty string.
* The set of writable env names is hard-coded; arbitrary writes are rejected.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from config import BASE_DIR

ENV_FILE = BASE_DIR / ".env"


# Field metadata: env name → (display label, kind, help text/url).
# kind drives grouping in the dashboard. "test"-able kinds: "llm", "tts".
SECRET_FIELDS: dict[str, dict[str, str]] = {
    "OPENAI_API_KEY": {
        "label": "OpenAI", "kind": "llm",
        "url": "https://platform.openai.com/api-keys",
        "hint": "starts with sk-…",
    },
    "GROQ_API_KEY": {
        "label": "Groq", "kind": "llm",
        "url": "https://console.groq.com/keys",
        "hint": "starts with gsk_…",
    },
    "OPENROUTER_API_KEY": {
        "label": "OpenRouter", "kind": "llm",
        "url": "https://openrouter.ai/keys",
        "hint": "starts with sk-or-…",
    },
    "ANTHROPIC_API_KEY": {
        "label": "Anthropic (Claude)", "kind": "llm",
        "url": "https://console.anthropic.com/settings/keys",
        "hint": "starts with sk-ant-…",
    },
    "GEMINI_API_KEY": {
        "label": "Google Gemini", "kind": "llm",
        "url": "https://aistudio.google.com/apikey",
        "hint": "free tier available",
    },
    "FISH_API_KEY": {
        "label": "Fish Audio", "kind": "tts",
        "url": "https://fish.audio/go-api/api-keys/",
        "hint": "from fish.audio dashboard",
    },
    "ELEVENLABS_API_KEY": {
        "label": "ElevenLabs", "kind": "tts",
        "url": "https://elevenlabs.io/app/settings/api-keys",
        "hint": "starts with sk_…",
    },
    "YOUTUBE_API_KEY": {
        "label": "YouTube API", "kind": "stream",
        "url": "https://console.cloud.google.com/apis/credentials",
        "hint": "Google Cloud Console",
    },
    "YOUTUBE_LIVE_CHAT_ID": {
        "label": "YouTube Live Chat ID", "kind": "stream",
        "url": "",
        "hint": "auto-detected when blank",
    },
    "TWITCH_OAUTH_TOKEN": {
        "label": "Twitch OAuth Token", "kind": "stream",
        "url": "https://twitchtokengenerator.com",
        "hint": "needs chat:read scope",
    },
    "TWITCH_CHANNEL": {
        "label": "Twitch Channel", "kind": "stream",
        "url": "",
        "hint": "the channel you stream on",
    },
    "TWITCH_NICK": {
        "label": "Twitch Nick", "kind": "stream",
        "url": "",
        "hint": "leave blank for anonymous",
    },
    "KICK_CHANNEL": {
        "label": "Kick Channel", "kind": "stream",
        "url": "",
        "hint": "channel slug",
    },
}


def mask(value: str) -> str:
    """Render a 'safe' preview of a secret. Never returns the full value."""
    if not value:
        return ""
    v = value.strip()
    if len(v) <= 8:
        return "•" * len(v)
    return f"{v[:3]}{'•' * 6}{v[-3:]}"


def list_secrets() -> list[dict[str, object]]:
    """Return metadata + masked previews. Never includes raw values."""
    out: list[dict[str, object]] = []
    for env_name, meta in SECRET_FIELDS.items():
        raw = os.getenv(env_name, "") or ""
        out.append(
            {
                "env": env_name,
                "label": meta["label"],
                "kind": meta["kind"],
                "url": meta["url"],
                "hint": meta["hint"],
                "is_set": bool(raw.strip()),
                "masked": mask(raw),
            }
        )
    return out


def set_secret(env_name: str, value: str) -> None:
    """Write a secret to .env and reload. Empty value removes the entry.

    Raises ``ValueError`` if env_name is not in the allowed set — prevents the
    UI from being tricked into writing arbitrary process environment.
    """
    if env_name not in SECRET_FIELDS:
        raise ValueError(f"refused write to unknown env name: {env_name!r}")

    value = (value or "").strip()
    _ensure_env_file()

    from dotenv import load_dotenv, set_key, unset_key

    if value:
        # quote_mode='auto' lets dotenv decide based on whitespace / specials.
        set_key(str(ENV_FILE), env_name, value, quote_mode="auto")
    else:
        try:
            unset_key(str(ENV_FILE), env_name)
        except Exception:
            # If the key wasn't there, that's fine.
            pass

    _harden_perms(ENV_FILE)
    # Make the new value visible to subsequent Secrets() reads in this process.
    load_dotenv(str(ENV_FILE), override=True)


def update_many(values: dict[str, str]) -> list[str]:
    """Update several secrets in one call. Returns the list of envs accepted."""
    accepted: list[str] = []
    for env, val in values.items():
        if env not in SECRET_FIELDS:
            continue
        set_secret(env, val)
        accepted.append(env)
    return accepted


def envs_for_kind(kind: str) -> Iterable[str]:
    return [env for env, meta in SECRET_FIELDS.items() if meta["kind"] == kind]


# ---------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------
def _ensure_env_file() -> None:
    if not ENV_FILE.exists():
        ENV_FILE.write_text("", encoding="utf-8")
        _harden_perms(ENV_FILE)


def _harden_perms(path: Path) -> None:
    """Best-effort: restrict .env to the owner on POSIX. Windows is OS-managed."""
    try:
        if os.name != "nt":
            os.chmod(path, 0o600)
    except Exception:
        pass
