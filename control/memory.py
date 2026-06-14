"""Working memory + goal state for the agent.

Carries context between decisions (a persistent scene note + sub-goal the VLM
maintains itself) and detects when the agent is STUCK — either looping the same
action or staring at a view that won't change — so we can force a strategy change
instead of turning left into a wall forever.
"""
from __future__ import annotations

from collections import deque


class AgentMemory:
    def __init__(self, *, history: int = 6, loop_threshold: int = 2,
                 change_dist: int = 6, stale_threshold: int = 3) -> None:
        self.note = ""        # persistent "where am I / what's going on" memory
        self.subgoal = ""     # current sub-goal (the plan, VLM-maintained)
        self.history: deque[tuple[str, str]] = deque(maxlen=history)
        self._last_hash = None
        self._last_action: str | None = None
        self._same_count = 0
        self._stale_count = 0
        self._loop_threshold = loop_threshold
        self._change_dist = change_dist
        self._stale_threshold = stale_threshold

    def on_frame(self, pil_image) -> None:
        """Call once per decision with the fresh frame — tracks view staleness."""
        try:
            import imagehash
            h = imagehash.phash(pil_image)
            if self._last_hash is not None and (h - self._last_hash) < self._change_dist:
                self._stale_count += 1
            else:
                self._stale_count = 0
            self._last_hash = h
        except Exception:  # noqa: BLE001
            self._stale_count = 0

    def stuck_reason(self) -> str:
        """Non-empty if we appear stuck (checked BEFORE the next decision)."""
        if self._same_count >= self._loop_threshold:
            return f"you've repeated the same action {self._same_count + 1} times with no progress"
        if self._stale_count >= self._stale_threshold:
            return "the view hasn't changed for several steps — you may be against a wall"
        return ""

    def on_action(self, action: str, steer: int, note: str, subgoal: str) -> None:
        """Record the decision the VLM just made and fold in its memory updates."""
        self._same_count = self._same_count + 1 if action == self._last_action else 0
        self._last_action = action
        if note:
            self.note = note.strip()[:180]
        if subgoal:
            self.subgoal = subgoal.strip()[:130]
        tag = action if not steer else f"{action}({steer})"
        self.history.append((tag, (note or "").strip()[:70]))

    def recent_text(self) -> str:
        return "  •  ".join(f"{a} ({n})" if n else a for a, n in self.history) or "(just started)"

    @property
    def last_action(self) -> str | None:
        return self._last_action
