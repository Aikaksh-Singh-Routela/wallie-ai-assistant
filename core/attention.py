"""AttentionEngine — picks how the streamer reacts to each vision event."""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class VisionReaction(str, Enum):
    DEEP = "deep"
    GLANCE = "glance"
    TANGENT = "tangent"
    IGNORE = "ignore"
    SILENCE = "silence"


@dataclass
class VisionDirective:
    reaction: VisionReaction
    target_sentences: int = 3
    tangent_seed: Optional[str] = None
    glance_style: str = "neutral"  # "neutral" | "amused" | "annoyed" | "curious"
    rationale: str = ""
    is_fallback: bool = False


@dataclass
class _AttentionState:
    """Rolling counters — nothing here escapes this module."""
    vision_reactions_in_a_row: int = 0
    last_vision_reaction_ts: float = 0.0
    ignored_scenes_in_a_row: int = 0
    last_scene_changed_at: float = 0.0
    scene_novelty_deadline: float = 0.0  # time until scene stops being "new"
    # Tangent pressure: rises when we haven't done a personal tangent in a while.
    tangent_pressure: float = 0.0
    # Silence pressure: rises when we've been talking non-stop.
    silence_pressure: float = 0.0


class AttentionEngine:

    def __init__(
        self,
        *,
        organicity: float = 0.75,
        deep_base: float = 0.22,
        glance_base: float = 0.28,
        tangent_base: float = 0.05,
        ignore_base: float = 0.27,
        silence_base: float = 0.18,
        min_vision_react_interval: float = 15.0,
        seed: Optional[int] = None,
    ) -> None:
        self._organicity = max(0.0, min(1.0, organicity))
        self._deep_base = deep_base
        self._glance_base = glance_base
        self._tangent_base = tangent_base
        self._ignore_base = ignore_base
        self._silence_base = silence_base
        self._min_vision_react_interval = min_vision_react_interval
        self._rng = random.Random(seed)
        self._state = _AttentionState()

    def decide_on_vision(
        self,
        *,
        change_kind: str,
        scene_age_sec: float,
        mood_arousal: float = 0.5,
        mood_focus: float = 0.5,
        mood_talkativity: float = 0.5,
        vision_engagement: float = 0.5,
        has_topic: bool = False,
        in_monologue_flow: bool = False,
        segments_total: int = 0,
        tangent_seeds: Optional[list[str]] = None,
        screen_activity: str = "",
        user_pattern: str = "",
        user_settled: bool = False,
        rapid_browsing: bool = False,
    ) -> VisionDirective:
        self._tick_pressures()

        if self._state.last_vision_reaction_ts == 0.0:
            if screen_activity in ("static", "micro", "typing"):
                chosen = VisionReaction.GLANCE
            else:
                chosen = VisionReaction.DEEP if change_kind == "scene" else VisionReaction.GLANCE
            directive = self._build_directive(
                chosen=chosen, mood_arousal=mood_arousal, mood_focus=mood_focus,
                change_kind=change_kind, tangent_seeds=tangent_seeds,
                weights={}, total=1.0,
            )
            directive.rationale = f"first-ever vision event → {chosen.value} (activity={screen_activity})"
            self._apply_state_update(chosen, change_kind)
            return directive

        w_deep = self._deep_base
        w_glance = self._glance_base
        w_tangent = self._tangent_base
        w_ignore = self._ignore_base
        w_silence = self._silence_base

        is_active_content = screen_activity in ("media", "app_switch")
        effective_cooldown = (
            self._min_vision_react_interval * 0.60 if is_active_content
            else self._min_vision_react_interval
        )
        since_last = time.time() - self._state.last_vision_reaction_ts
        if self._state.last_vision_reaction_ts > 0 and since_last < effective_cooldown:
            cooldown_ratio = 1.0 - (since_last / effective_cooldown)
            w_deep *= max(0.05, 1.0 - cooldown_ratio * 0.9)
            w_glance *= max(0.10, 1.0 - cooldown_ratio * 0.7)
            w_tangent *= max(0.10, 1.0 - cooldown_ratio * 0.6)
            w_ignore *= 1.0 + cooldown_ratio * 1.5
            w_silence *= 1.0 + cooldown_ratio * 1.0

        if change_kind == "delta":
            w_deep *= 0.35
            w_glance *= 1.30
            w_tangent *= 0.80
            w_ignore *= 1.60
            w_silence *= 1.20
        else:
            w_deep *= 1.10
            w_ignore *= 0.85

        streak = self._state.vision_reactions_in_a_row
        if streak >= 2:
            w_deep *= max(0.20, 1.0 - 0.25 * streak)
            w_glance *= max(0.30, 1.0 - 0.15 * streak)
            w_ignore *= 1.40
            w_silence *= 1.20
        if streak >= 4:
            w_deep *= 0.30
            w_ignore *= 1.50

        if self._state.ignored_scenes_in_a_row >= 2:
            w_ignore *= 0.5
            w_deep *= 1.4
            w_glance *= 1.3
        if self._state.ignored_scenes_in_a_row >= 4:
            w_ignore *= 0.3
            w_deep *= 1.6
            w_glance *= 1.5

        if scene_age_sec < 6.0:
            w_deep *= 1.08
            w_silence *= 0.90
        elif scene_age_sec > 45.0:
            w_deep *= 0.40
            w_glance *= 1.10
            w_ignore *= 1.40

        eng = vision_engagement
        w_deep *= 0.6 + eng
        w_glance *= 1.1 - 0.4 * eng
        if mood_focus < 0.50:
            w_tangent *= 1.35
        if mood_arousal < 0.35:
            w_silence *= 1.4
            w_ignore *= 1.2
            w_deep *= 0.7
        if mood_talkativity < 0.35:
            w_silence *= 1.6
            w_deep *= 0.8
        if mood_arousal > 0.75:
            w_silence *= 0.4
            w_deep *= 1.1

        if in_monologue_flow:
            w_tangent *= 0.50
            w_deep *= 0.45
            w_glance *= 0.70
            w_ignore *= 1.60
            w_silence *= 1.20

        if not tangent_seeds:
            w_tangent *= 0.4
        else:
            w_tangent *= 0.7 + self._state.tangent_pressure * 1.2

        if has_topic:
            w_tangent *= 1.15

        if screen_activity == "scroll":
            w_ignore *= 1.6
            w_glance *= 1.3
            w_deep *= 0.4
            w_silence *= 1.2
        elif screen_activity == "typing":
            w_ignore *= 2.5
            w_silence *= 1.8
            w_deep *= 0.2
            w_glance *= 0.5
        elif screen_activity == "app_switch":
            w_deep *= 1.6
            w_glance *= 1.4
            w_ignore *= 0.35
            w_silence *= 0.6
        elif screen_activity == "navigation":
            w_glance *= 1.3
            w_deep *= 0.9
            w_tangent *= 1.1
        elif screen_activity == "media":
            w_deep *= 1.10
            w_glance *= 1.20
            w_ignore *= 0.65
            w_silence *= 0.75
            w_tangent *= 1.05

        if rapid_browsing:
            w_ignore *= 2.0
            w_silence *= 1.5
            w_deep *= 0.25
            w_glance *= 0.6

        if user_settled and screen_activity not in ("typing",):
            w_deep *= 1.3
            w_ignore *= 0.6
            w_glance *= 1.1

        if self._organicity < 0.2:
            w_tangent *= 0.3
            w_ignore *= 0.3
            w_silence *= 0.2

        weights = {
            VisionReaction.DEEP:    max(0.01, w_deep),
            VisionReaction.GLANCE:  max(0.01, w_glance),
            VisionReaction.TANGENT: max(0.01, w_tangent),
            VisionReaction.IGNORE:  max(0.01, w_ignore),
            VisionReaction.SILENCE: max(0.01, w_silence),
        }
        total = sum(weights.values())
        r = self._rng.random() * total
        acc = 0.0
        chosen = VisionReaction.DEEP
        for reaction, w in weights.items():
            acc += w
            if r <= acc:
                chosen = reaction
                break

        directive = self._build_directive(
            chosen=chosen,
            mood_arousal=mood_arousal,
            mood_focus=mood_focus,
            change_kind=change_kind,
            tangent_seeds=tangent_seeds,
            weights=weights,
            total=total,
        )
        self._apply_state_update(chosen, change_kind)
        return directive

    def should_hold_silence(
        self,
        *,
        mood_silence_probability: float,
        in_monologue_flow: bool,
    ) -> bool:
        p = mood_silence_probability
        if in_monologue_flow:
            p += 0.05
        p += self._state.silence_pressure * 0.25
        # Cap.
        p = min(0.45, p)
        return self._rng.random() < p

    def on_scene_change(self) -> None:
        now = time.time()
        self._state.last_scene_changed_at = now
        self._state.scene_novelty_deadline = now + 15.0

    def on_segment_spoken(self, *, intent_kind: str) -> None:
        if intent_kind == "vision":
            self._state.vision_reactions_in_a_row += 1
            self._state.ignored_scenes_in_a_row = 0
            self._state.tangent_pressure = max(0.0, self._state.tangent_pressure - 0.2)
            self._state.last_vision_reaction_ts = time.time()
        else:
            self._state.vision_reactions_in_a_row = 0
        self._state.silence_pressure = min(1.0, self._state.silence_pressure + 0.08)
        self._state.tangent_pressure = min(1.0, self._state.tangent_pressure + 0.05)

    def on_silence_beat(self) -> None:
        self._state.silence_pressure = max(0.0, self._state.silence_pressure - 0.3)

    def on_vision_ignored(self) -> None:
        self._state.ignored_scenes_in_a_row += 1

    def snapshot(self) -> dict:
        since = time.time() - self._state.last_vision_reaction_ts if self._state.last_vision_reaction_ts > 0 else None
        return {
            "vision_streak": self._state.vision_reactions_in_a_row,
            "ignored_streak": self._state.ignored_scenes_in_a_row,
            "tangent_pressure": round(self._state.tangent_pressure, 2),
            "silence_pressure": round(self._state.silence_pressure, 2),
            "organicity": self._organicity,
            "vision_cooldown_remaining": round(max(0.0, self._min_vision_react_interval - since), 1) if since is not None else 0,
        }

    def _tick_pressures(self) -> None:
        self._state.tangent_pressure = max(
            0.0, self._state.tangent_pressure - 0.01
        )
        self._state.silence_pressure = max(
            0.0, self._state.silence_pressure - 0.01
        )

    def _apply_state_update(self, chosen: VisionReaction, change_kind: str) -> None:
        now = time.time()
        # Only update cooldown on actual reactions, not IGNORE/SILENCE.
        if chosen == VisionReaction.IGNORE:
            self._state.ignored_scenes_in_a_row += 1
            self._state.vision_reactions_in_a_row = 0
        elif chosen == VisionReaction.SILENCE:
            self._state.silence_pressure = max(0.0, self._state.silence_pressure - 0.4)
        else:
            self._state.last_vision_reaction_ts = now
            self._state.vision_reactions_in_a_row += 1
            self._state.ignored_scenes_in_a_row = 0
            if chosen == VisionReaction.TANGENT:
                self._state.tangent_pressure = max(0.0, self._state.tangent_pressure - 0.3)
            self._state.silence_pressure = min(1.0, self._state.silence_pressure + 0.1)

    def _build_directive(
        self,
        *,
        chosen: VisionReaction,
        mood_arousal: float,
        mood_focus: float,
        change_kind: str,
        tangent_seeds: Optional[list[str]],
        weights: dict,
        total: float,
    ) -> VisionDirective:
        if chosen == VisionReaction.DEEP:
            target = 4 if mood_arousal > 0.5 else 3
        elif chosen == VisionReaction.TANGENT:
            target = 3
        elif chosen == VisionReaction.GLANCE:
            target = 3 if mood_arousal > 0.6 else 2
        else:
            target = 0  # ignore / silence

        glance_style = "neutral"
        if chosen == VisionReaction.GLANCE:
            if mood_arousal > 0.70:
                glance_style = "amused"
            elif mood_arousal < 0.35:
                glance_style = "annoyed"
            elif mood_focus < 0.45:
                glance_style = "curious"

        tangent_seed: Optional[str] = None
        if chosen == VisionReaction.TANGENT and tangent_seeds:
            tangent_seed = self._rng.choice(tangent_seeds)

        probs = {k.value: round(v / total, 2) for k, v in weights.items()}
        rationale = (
            f"{chosen.value} chosen (change={change_kind}, probs={probs}, "
            f"streak={self._state.vision_reactions_in_a_row}, "
            f"tangent_p={self._state.tangent_pressure:.2f})"
        )

        return VisionDirective(
            reaction=chosen,
            target_sentences=target,
            tangent_seed=tangent_seed,
            glance_style=glance_style,
            rationale=rationale,
        )
