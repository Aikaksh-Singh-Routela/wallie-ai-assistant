"""Cross-session persistent memory: streamer notes + viewer interaction log.

Stored as a single JSON file next to the active profile, named
``{profile_name}.memory.json``. Loaded at session start, saved at graceful
stop. All failures are non-fatal — a missing or corrupted store just means
the session starts fresh with no cross-session context.

Two components
--------------
notes
    A compact string (usually 6-12 bullets) summarising what the streamer has
    done across *past* sessions. Updated by the rolling summariser each time it
    runs, and persisted on stop. Injected into the system prompt as a second
    memory layer below the current-session ``session_notes`` so the persona
    keeps continuity across stream days.

viewer_log
    Append-only record of every viewer chat message the streamer replied to.
    Capped at ``max_log`` entries (FIFO). Dashboard exposes a read endpoint so
    the user can see "who talked to Wallie today".
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from loguru import logger

_MAX_LOG = 500


class MemoryStore:
    def __init__(self, path: Path) -> None:
        self._path = path
        self.notes: str = ""
        self.viewer_log: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def load(self) -> None:
        if not self._path.exists():
            logger.debug(f"memory: no store at {self._path.name} — starting fresh")
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self.notes = (data.get("notes") or "").strip()
            raw_log = data.get("viewer_log") or []
            self.viewer_log = raw_log[-_MAX_LOG:]
            logger.info(
                f"memory: loaded {self._path.name} — "
                f"{len(self.notes)} chars notes, {len(self.viewer_log)} viewer entries"
            )
        except Exception as e:
            logger.warning(f"memory: load failed (non-fatal): {e}")

    def save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            data: dict[str, Any] = {
                "notes": self.notes,
                "viewer_log": self.viewer_log[-_MAX_LOG:],
                "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            self._path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            logger.info(
                f"memory: saved {self._path.name} — "
                f"{len(self.notes)} chars notes, {len(self.viewer_log)} viewer entries"
            )
        except Exception as e:
            logger.warning(f"memory: save failed (non-fatal): {e}")

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------
    def update_notes(self, new_notes: str) -> None:
        """Replace persistent cross-session notes (called by rolling summariser)."""
        self.notes = (new_notes or "").strip()

    def log_viewer(self, *, username: str, platform: str, text: str) -> None:
        """Append a viewer interaction entry."""
        self.viewer_log.append(
            {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "username": username,
                "platform": platform,
                "text": text[:300],
            }
        )
        # Rolling trim — keep only the most recent entries in memory.
        if len(self.viewer_log) > _MAX_LOG * 2:
            self.viewer_log = self.viewer_log[-_MAX_LOG:]

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def recent_viewers(self, n: int = 50) -> list[dict[str, Any]]:
        """Return the *n* most recent viewer log entries (newest last)."""
        return self.viewer_log[-n:]

    def summary_for_prompt(self, max_chars: int = 800) -> str:
        """Return notes truncated to ``max_chars`` for injection into the
        system prompt. Returns empty string when there are no notes."""
        if not self.notes:
            return ""
        return self.notes[:max_chars]
