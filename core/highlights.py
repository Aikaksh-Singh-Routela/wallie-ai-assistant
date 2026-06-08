"""Auto-highlight detection — flags Wallie's own viral moments while it streams.

This is the brain of the clip pipeline: as Wallie talks, every segment is scored
on how clip-worthy it is (excitement, strong reactions, punchy/hype delivery). When
a moment crosses the bar, a timestamped marker is written to a per-session JSONL file.

Those markers tell you *exactly* where the good clips are — no scrubbing a 30-minute
recording. A companion exporter can then cut vertical Shorts at each marker.
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger

HIGHLIGHTS_DIR = Path("highlights")

_HYPE_PHRASES = (
    "no way", "let's go", "lets go", "insane", "i can't", "i cant", "oh my",
    "what the", "hold on", "wait", "stop", "bro", "actually", "called it",
    "lock it", "no notes", "trust me", "i told you", "deadass", "lowkey",
)


def _score(text: str, arousal: float, reaction: str) -> tuple[float, str]:
    """Return (score 0..~1.2, short reason). Heuristic, tuned for short reactions."""
    s = max(0.0, min(1.0, arousal)) * 0.45  # mood excitement
    reasons: list[str] = []
    if arousal >= 0.6:
        reasons.append("hyped")

    r = (reaction or "").lower()
    if "deep" in r or "tangent" in r:
        s += 0.18
        reasons.append(r)

    t = text or ""
    excls = t.count("!")
    if excls:
        s += min(0.25, excls * 0.09)
        reasons.append("!")
    caps = sum(1 for w in t.split() if len(w) >= 3 and w.isupper())
    if caps:
        s += min(0.2, caps * 0.08)
        reasons.append("CAPS")
    tl = t.lower()
    if any(h in tl for h in _HYPE_PHRASES):
        s += 0.16
        reasons.append("hype")

    return s, "+".join(reasons) or "—"


class HighlightTracker:
    def __init__(self, profile_name: str, *, enabled: bool = False, threshold: float = 0.55) -> None:
        self._profile = profile_name or "default"
        self._enabled = bool(enabled)
        self._threshold = float(threshold)
        self._path: Optional[Path] = None
        self._started = 0.0
        self._count = 0

    def start(self) -> None:
        if not self._enabled:
            return
        HIGHLIGHTS_DIR.mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self._path = HIGHLIGHTS_DIR / f"{self._profile}-{stamp}.jsonl"
        self._started = time.time()
        self._count = 0
        logger.info(f"highlights: auto-detect on → {self._path} (threshold {self._threshold})")

    def note(self, *, text: str, arousal: float, reaction: str = "") -> None:
        """Score a just-spoken segment; log a marker if it's clip-worthy."""
        if not self._enabled or self._path is None or not text.strip():
            return
        score, reason = _score(text, arousal, reaction)
        if score < self._threshold:
            return
        now = time.time()
        marker = {
            "ts": round(now, 3),                     # wall-clock epoch (align to recording)
            "iso": datetime.now().isoformat(timespec="seconds"),
            "t_session": round(now - self._started, 2),  # seconds since session start
            "score": round(score, 3),
            "reason": reason,
            "text": text.strip()[:300],
        }
        try:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(marker, ensure_ascii=False) + "\n")
            self._count += 1
            logger.info(f"highlight> [{score:.2f} {reason}] {text.strip()[:70]}")
        except OSError as e:
            logger.warning(f"highlights: write failed: {e}")

    def stop(self) -> None:
        if self._enabled and self._path is not None:
            logger.info(f"highlights: {self._count} marker(s) saved → {self._path}")
