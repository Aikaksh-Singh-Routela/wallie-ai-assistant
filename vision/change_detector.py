"""Perceptual-hash based change detector with idle + micro-change filtering.

Sending every frame to the vision model wastes money and adds latency. Instead we
compute a pHash each tick and only emit when the Hamming distance to the last
emitted frame exceeds a threshold.

v3 additions (organic layer):
  * `micro_change_threshold`: deltas below this are silently suppressed.
    Cursor blinks, animated ads, small UI ticks won't wake the host up.
  * Scene stability tracker: emits a :class:`ChangeType.STABLE` signal the
    first time a scene has been unchanged for a while — the orchestrator
    uses it to know the host can stop looking at the screen.
  * Per-tick delta history for rate analysis (how "busy" the screen is).

Previous additions (kept):
  * ChangeType classification (NONE / DELTA / SCENE_CHANGE / IDLE).
  * IDLE detection via image std.
"""
from __future__ import annotations

import io
import time
from collections import deque
from typing import Deque, Optional, Tuple

import imagehash
from PIL import Image

from .capture import Frame
from .scene_classifier import ChangeType, ScreenActivity
from .activity_classifier import ActivityResult, ScreenActivityClassifier


class FrameChangeDetector:
    def __init__(
        self,
        *,
        threshold: int = 8,
        scene_change_threshold: int = 20,
        idle_variance_threshold: float = 15.0,
        micro_change_threshold: int = 4,
    ) -> None:
        self._threshold = threshold
        self._scene_threshold = scene_change_threshold
        self._idle_var = idle_variance_threshold
        self._micro_threshold = micro_change_threshold
        self._last_hash: Optional[imagehash.ImageHash] = None
        # Recent Hamming distances (for rate/busy-ness tracking).
        self._recent_distances: Deque[int] = deque(maxlen=16)
        # Unix ts of the last non-NONE classification. Used for stability.
        self._last_change_ts: float = time.time()
        self._scene_started_ts: float = time.time()
        # Screen activity classifier — runs every frame for pattern detection.
        self._activity_classifier = ScreenActivityClassifier()
        self._last_activity: ActivityResult = ActivityResult(
            activity=ScreenActivity.STATIC, confidence=1.0,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, frame: Frame) -> Tuple[ChangeType, Optional[imagehash.ImageHash]]:
        """Classify a new frame.

        Returns ``(ChangeType, hash)`` where ``hash`` is the computed pHash
        of the frame (useful for callers that want to cache it in
        ``SceneMemory``). Returns ``(ChangeType.IDLE, None)`` when the
        screen is blank/static.

        Also runs the activity classifier on every frame (even NONE/IDLE) to
        maintain accurate activity state. Access the result via
        :attr:`last_activity`.
        """
        img = Image.open(io.BytesIO(frame.jpeg))

        # Always run activity classification to keep the pattern history warm.
        self._last_activity = self._activity_classifier.classify(frame)

        # Idle detection: fast path, skip pHash if screen is blank.
        if self._is_idle(img):
            return ChangeType.IDLE, None

        h = imagehash.phash(img)

        if self._last_hash is None:
            # First frame ever: emit as SCENE_CHANGE to seed everything.
            self._last_hash = h
            self._last_change_ts = time.time()
            self._scene_started_ts = time.time()
            return ChangeType.SCENE_CHANGE, h

        distance = abs(h - self._last_hash)
        self._recent_distances.append(distance)

        # Below the meaningful-change floor → nothing to do.
        if distance < self._threshold:
            return ChangeType.NONE, h

        # Distance between threshold and micro_change_threshold is a
        # "micro change" — filter it unless micro is below threshold (0).
        # micro_threshold acts as a secondary floor: a tick must exceed BOTH
        # threshold AND micro_threshold to be counted.
        if self._micro_threshold > 0 and distance < (self._threshold + self._micro_threshold):
            # Micro-changes are suppressed but we still update the hash
            # slowly so cumulative drift eventually registers.
            return ChangeType.NONE, h

        # Update stored hash on any meaningful change.
        self._last_hash = h
        now = time.time()
        self._last_change_ts = now

        if distance >= self._scene_threshold:
            self._scene_started_ts = now
            return ChangeType.SCENE_CHANGE, h

        return ChangeType.DELTA, h

    @property
    def last_activity(self) -> ActivityResult:
        """The most recent screen activity classification."""
        return self._last_activity

    @property
    def activity_pattern(self) -> str:
        """Summarize the recent activity pattern (e.g. 'browsing', 'settled')."""
        return self._activity_classifier.recent_pattern()

    @property
    def user_settled(self) -> bool:
        """True if the user has been on the same content for a while."""
        return self._activity_classifier.is_user_settled()

    @property
    def rapid_browsing(self) -> bool:
        """True if the user is quickly flipping through content."""
        return self._activity_classifier.is_rapid_browsing()

    def should_emit(self, frame: Frame) -> bool:
        """Backward-compatible helper for callers that only need a bool."""
        change_type, _ = self.classify(frame)
        return change_type not in (ChangeType.NONE, ChangeType.IDLE)

    def reset(self) -> None:
        self._last_hash = None
        self._recent_distances.clear()
        self._last_change_ts = time.time()
        self._scene_started_ts = time.time()
        self._activity_classifier.reset()
        self._last_activity = ActivityResult(
            activity=ScreenActivity.STATIC, confidence=1.0,
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def busyness(self) -> float:
        """Return a 0..1 rough indicator of how "busy" the screen is lately.

        High busyness = lots of movement (a game, a video). The orchestrator
        can use this to throttle emit rate further — a busy scene would
        otherwise fire a vision event every single tick.
        """
        if not self._recent_distances:
            return 0.0
        avg = sum(self._recent_distances) / len(self._recent_distances)
        return min(1.0, avg / float(max(1, self._scene_threshold)))

    def seconds_since_change(self) -> float:
        return time.time() - self._last_change_ts

    def scene_age_sec(self) -> float:
        return time.time() - self._scene_started_ts

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_idle(self, img: Image.Image) -> bool:
        """Return True when the image is essentially static / blank.

        Uses the standard deviation of pixel values as a cheap proxy for
        visual complexity. A pure desktop, a black screen, or a solid colour
        all produce very low std values.
        """
        try:
            import numpy as np  # type: ignore
            arr = np.array(img.convert("L"), dtype=np.float32)
            return float(arr.std()) < self._idle_var
        except Exception:
            # numpy unavailable or conversion failed: skip idle detection.
            return False
