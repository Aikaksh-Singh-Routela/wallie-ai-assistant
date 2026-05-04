"""Scene memory — tracks what the AI has already described so it can build on it.

Gives the orchestrator a single source of truth about:
  * what the screen looked like last time the AI reacted to it
  * how long we've been on the same scene
  * how many delta changes happened during that scene
  * what the AI said about it (so it doesn't repeat itself)

v4: UserBehaviorTracker — tracks user screen-control patterns so the AI can
    adapt its responses organically (e.g. suppress during rapid browsing,
    react when settled, acknowledge scrolling naturally).
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Optional

from .scene_classifier import ScreenActivity


@dataclass
class SceneMemory:
    # The last natural-language description the AI produced for a vision turn.
    last_description: str = ""
    # pHash of the frame that triggered the last scene-change event.
    last_scene_hash: Optional[Any] = None
    # Wall-clock time when the current scene was first seen.
    scene_started_at: float = field(default_factory=time.time)
    # Number of DELTA changes detected within the current scene.
    scene_change_count: int = 0

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def record_scene_change(self, h: Any) -> None:
        """Call when a SCENE_CHANGE event is processed."""
        self.last_scene_hash = h
        self.scene_started_at = time.time()
        self.scene_change_count = 0

    def record_delta(self) -> None:
        """Call when a DELTA event is processed (same scene, small change)."""
        self.scene_change_count += 1

    def record_spoken(self, text: str) -> None:
        """Call after the AI has finished speaking a vision-related segment."""
        if text:
            self.last_description = text

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def is_same_scene(self, h: Any, threshold: int) -> bool:
        """Return True if *h* is close enough to the last scene hash to be
        considered the same scene (Hamming distance < threshold)."""
        if self.last_scene_hash is None:
            return False
        try:
            return abs(h - self.last_scene_hash) < threshold
        except Exception:
            return False

    def scene_age_sec(self) -> float:
        """How many seconds since the current scene was first seen."""
        return time.time() - self.scene_started_at


@dataclass
class UserBehaviorTracker:
    """Tracks user's screen-control patterns over time.

    The orchestrator feeds every VisionEvent's activity into this tracker.
    It builds a picture of what the user is doing so the AI can adapt:
      - During rapid browsing → suppress reactions, wait for user to settle
      - When user is settled → AI can comment more deeply
      - During scrolling → AI acknowledges movement naturally
      - During media playback → AI can comment on what's playing
    """

    # Recent activity types for pattern analysis.
    _recent_activities: Deque[ScreenActivity] = field(
        default_factory=lambda: deque(maxlen=20)
    )
    # Timestamps of recent activities for pace calculation.
    _activity_timestamps: Deque[float] = field(
        default_factory=lambda: deque(maxlen=20)
    )
    # Current dominant pattern label.
    current_pattern: str = "starting"
    # How long the user has been settled (seconds since last active change).
    _last_active_change_ts: float = field(default_factory=time.time)
    # Track scroll direction consistency.
    _recent_scroll_dirs: Deque[str] = field(
        default_factory=lambda: deque(maxlen=6)
    )
    # Consecutive settled frames.
    _settled_count: int = 0
    # Last activity type to detect transitions.
    _prev_activity: ScreenActivity = ScreenActivity.STATIC

    def record_activity(
        self,
        activity: ScreenActivity,
        pattern: str = "",
        scroll_direction: str = "",
    ) -> None:
        """Record a new screen activity observation."""
        now = time.time()
        self._recent_activities.append(activity)
        self._activity_timestamps.append(now)

        if pattern:
            self.current_pattern = pattern

        if activity == ScreenActivity.SCROLL and scroll_direction:
            self._recent_scroll_dirs.append(scroll_direction)

        # Track settled-ness.
        if activity in (ScreenActivity.STATIC, ScreenActivity.MICRO):
            self._settled_count += 1
        else:
            self._settled_count = 0
            self._last_active_change_ts = now

        self._prev_activity = activity

    @property
    def settled_seconds(self) -> float:
        """How many seconds since the user last actively changed something."""
        return time.time() - self._last_active_change_ts

    @property
    def is_settled(self) -> bool:
        """True if the user has been stationary for a meaningful period."""
        return self._settled_count >= 4 and self.settled_seconds > 8.0

    @property
    def is_rapid_browsing(self) -> bool:
        """True if the user is quickly flipping through content."""
        if len(self._recent_activities) < 4:
            return False
        recent = list(self._recent_activities)[-5:]
        active_count = sum(
            1 for a in recent
            if a in (ScreenActivity.SCROLL, ScreenActivity.NAVIGATION, ScreenActivity.APP_SWITCH)
        )
        return active_count >= 3

    @property
    def is_watching_media(self) -> bool:
        """True if the user appears to be watching a video or animation."""
        if len(self._recent_activities) < 3:
            return False
        recent = list(self._recent_activities)[-4:]
        return sum(1 for a in recent if a == ScreenActivity.MEDIA_PLAYING) >= 3

    @property
    def is_typing(self) -> bool:
        """True if the user appears to be typing."""
        if len(self._recent_activities) < 2:
            return False
        recent = list(self._recent_activities)[-3:]
        return sum(1 for a in recent if a == ScreenActivity.TYPING) >= 2

    @property
    def scroll_direction(self) -> str:
        """Dominant recent scroll direction, or '' if not scrolling."""
        if not self._recent_scroll_dirs:
            return ""
        recent = list(self._recent_scroll_dirs)[-3:]
        # If consistent direction, return it.
        if len(set(recent)) == 1:
            return recent[0]
        return recent[-1] if recent else ""

    @property
    def browsing_pace(self) -> float:
        """0..1 — how fast the user is changing things. 0=stationary, 1=rapid."""
        if len(self._activity_timestamps) < 3:
            return 0.0
        recent = list(self._recent_activities)[-8:]
        active = sum(
            1 for a in recent
            if a not in (ScreenActivity.STATIC, ScreenActivity.MICRO)
        )
        return min(1.0, active / max(1, len(recent)))

    def transition_detected(self) -> Optional[str]:
        """Detect meaningful activity transitions for the AI to acknowledge.

        Returns a transition label like "started_scrolling", "switched_app",
        "settled_down", "started_watching" — or None if no notable transition.
        """
        if len(self._recent_activities) < 2:
            return None
        prev = self._prev_activity
        curr = list(self._recent_activities)[-1]

        if curr == ScreenActivity.APP_SWITCH:
            return "switched_app"
        if curr == ScreenActivity.NAVIGATION and prev != ScreenActivity.NAVIGATION:
            return "navigated"
        if curr == ScreenActivity.SCROLL and prev not in (ScreenActivity.SCROLL,):
            return "started_scrolling"
        if curr == ScreenActivity.STATIC and self._settled_count == 4:
            return "settled_down"
        if curr == ScreenActivity.MEDIA_PLAYING and prev != ScreenActivity.MEDIA_PLAYING:
            return "started_watching"
        if curr == ScreenActivity.TYPING and prev != ScreenActivity.TYPING:
            return "started_typing"
        return None

    def adaptation_hint(self) -> str:
        """Generate a short hint for the AI about how to adapt to the user's behavior.

        This is injected into the prompt to help the AI sound like IT is in
        control of the screen, matching the user's actual actions.
        """
        if self.is_rapid_browsing:
            return (
                "You're quickly flipping through things — browsing, looking "
                "for something. Don't commit to any one thing until you land "
                "on something that catches your eye."
            )
        if self.is_typing:
            return (
                "You're typing something right now. If you mention it, keep it "
                "brief and natural — 'let me type this real quick' or similar."
            )
        if self.is_watching_media:
            return (
                "You're watching something. React to what's happening in the "
                "video/stream, not to the player UI."
            )
        if self.is_settled:
            settled = self.settled_seconds
            if settled > 30:
                return (
                    "You've been on this for a while now. Find a fresh angle or "
                    "move on — don't repeat what you already said about it."
                )
            return (
                "You've settled on this. Take your time with it — no need to "
                "rush to the next thing."
            )

        recent = list(self._recent_activities)[-1] if self._recent_activities else ScreenActivity.STATIC
        if recent == ScreenActivity.SCROLL:
            d = self.scroll_direction
            if d:
                return f"You're scrolling {d} through the content."
            return "You're scrolling through the content."
        if recent == ScreenActivity.NAVIGATION:
            return "You just navigated to something new."
        if recent == ScreenActivity.APP_SWITCH:
            return "You just switched to a different app/window."
        return ""
