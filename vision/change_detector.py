"""Perceptual-hash based change detector with idle + micro-change filtering."""
from __future__ import annotations

import time
from collections import deque
from typing import Deque, Optional, Tuple

import imagehash
import numpy as np
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
        self._recent_distances: Deque[int] = deque(maxlen=16)
        self._last_change_ts: float = time.time()
        self._scene_started_ts: float = time.time()
        self._activity_classifier = ScreenActivityClassifier()
        self._last_activity: ActivityResult = ActivityResult(
            activity=ScreenActivity.STATIC, confidence=1.0,
        )

    def classify(self, frame: Frame) -> Tuple[ChangeType, Optional[imagehash.ImageHash]]:
        img = frame.to_pil()

        self._last_activity = self._activity_classifier.classify(frame)

        if self._is_idle(img):
            return ChangeType.IDLE, None

        h = imagehash.phash(img)

        if self._last_hash is None:
            # First frame.
            self._last_hash = h
            self._last_change_ts = time.time()
            self._scene_started_ts = time.time()
            return ChangeType.SCENE_CHANGE, h

        distance = abs(h - self._last_hash)
        self._recent_distances.append(distance)

        # Below threshold.
        if distance < self._threshold:
            return ChangeType.NONE, h

        # Micro-change suppression.
        if self._micro_threshold > 0 and distance < (self._threshold + self._micro_threshold):
            return ChangeType.NONE, h

        # Meaningful change.
        self._last_hash = h
        now = time.time()
        self._last_change_ts = now

        if distance >= self._scene_threshold:
            self._scene_started_ts = now
            return ChangeType.SCENE_CHANGE, h

        return ChangeType.DELTA, h

    @property
    def last_activity(self) -> ActivityResult:
        return self._last_activity

    @property
    def activity_pattern(self) -> str:
        return self._activity_classifier.recent_pattern()

    @property
    def user_settled(self) -> bool:
        return self._activity_classifier.is_user_settled()

    @property
    def rapid_browsing(self) -> bool:
        return self._activity_classifier.is_rapid_browsing()

    def reset(self) -> None:
        self._last_hash = None
        self._recent_distances.clear()
        self._last_change_ts = time.time()
        self._scene_started_ts = time.time()
        self._activity_classifier.reset()
        self._last_activity = ActivityResult(
            activity=ScreenActivity.STATIC, confidence=1.0,
        )

    def busyness(self) -> float:
        if not self._recent_distances:
            return 0.0
        avg = sum(self._recent_distances) / len(self._recent_distances)
        return min(1.0, avg / float(max(1, self._scene_threshold)))

    def seconds_since_change(self) -> float:
        return time.time() - self._last_change_ts

    def scene_age_sec(self) -> float:
        return time.time() - self._scene_started_ts

    def _is_idle(self, img: Image.Image) -> bool:
        arr = np.array(img.convert("L"), dtype=np.float32)
        return float(arr.std()) < self._idle_var
