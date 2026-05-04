"""AttentionEngine — organic decision-making for vision reactions.

The single biggest thing that makes an AI streamer feel "robotic" is reacting
to every scene change with equal depth, in the same shape, in the same voice.
A real broadcaster:

  * Glances at a scene and says nothing more than "oh nice" (shallow).
  * Dives deep when the content hooks them (deep).
  * Notices the screen but uses it as a springboard for a personal tangent
    (tangent).
  * Tunes the screen out entirely when their monologue is in flow (ignore).
  * Revisits the screen after a long silence (re-check).

AttentionEngine decides which of these paths the next vision turn takes.
It's stateful — the choice depends on:

  * how many recent segments were vision-reactions in a row (streak fatigue)
  * how long the current scene has been onscreen (novelty curve)
  * whether it's a DELTA vs. SCENE_CHANGE
  * the host's current mood (focus / arousal) from MoodEngine
  * the user-configured `organicity` (0=rigid, 1=max drift)

Output is one of :class:`VisionReaction` plus a structured `VisionDirective`
that the persona layer turns into a prompt. The orchestrator uses the
reaction type to choose intent kind (`vision` vs. `monologue` vs. silence).
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class VisionReaction(str, Enum):
    # Full first reaction (2-4 sentences, name specific things).
    DEEP = "deep"
    # Quick one-liner glance.
    GLANCE = "glance"
    # Screen noticed but used as a springboard to a personal tangent/story.
    TANGENT = "tangent"
    # Ignore the vision event entirely this turn — pretend it didn't happen.
    IGNORE = "ignore"
    # Silent beat: don't speak at all this tick.
    SILENCE = "silence"


@dataclass
class VisionDirective:
    """Structured decision passed to persona.vision_turn() / monologue_turn()."""
    reaction: VisionReaction
    # Hint for how many sentences the response should be (soft guide to LLM).
    target_sentences: int = 3
    # If TANGENT, a short seed the prompt can lean on (e.g. "your ex's router").
    tangent_seed: Optional[str] = None
    # If GLANCE, whether this is just a "noticed something" acknowledgement.
    glance_style: str = "neutral"  # "neutral" | "amused" | "annoyed" | "curious"
    # Debug trace — shows why this reaction was picked (surfaced to dashboard).
    rationale: str = ""


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
    """Chooses how the streamer actually reacts to a vision event.

    The orchestrator calls :meth:`decide_on_vision` with the change kind
    (`"scene"` / `"delta"`) and the current mood snapshot. The engine returns
    a :class:`VisionDirective` describing the chosen reaction.
    """

    def __init__(
        self,
        *,
        organicity: float = 0.75,
        deep_base: float = 0.12,
        glance_base: float = 0.16,
        tangent_base: float = 0.08,
        ignore_base: float = 0.40,
        silence_base: float = 0.24,
        min_vision_react_interval: float = 30.0,
        seed: Optional[int] = None,
    ) -> None:
        self._organicity = max(0.0, min(1.0, organicity))
        # Base distribution for "scene" events. Probabilities are normalized
        # after all mood/state biases are applied.
        self._deep_base = deep_base
        self._glance_base = glance_base
        self._tangent_base = tangent_base
        self._ignore_base = ignore_base
        self._silence_base = silence_base
        self._min_vision_react_interval = min_vision_react_interval
        self._rng = random.Random(seed)
        self._state = _AttentionState()

    # ------------------------------------------------------------------
    # Public: vision decision
    # ------------------------------------------------------------------
    def decide_on_vision(
        self,
        *,
        change_kind: str,                  # "scene" | "delta"
        scene_age_sec: float,              # how long this scene has been up
        mood_arousal: float = 0.5,
        mood_focus: float = 0.5,
        mood_talkativity: float = 0.5,
        vision_engagement: float = 0.5,    # MoodEngine.wants_vision_engagement()
        has_topic: bool = False,
        in_monologue_flow: bool = False,   # True if last 2+ segments were monologue
        segments_total: int = 0,
        tangent_seeds: Optional[list[str]] = None,
        # v4: screen activity context
        screen_activity: str = "",         # ScreenActivity.value
        user_pattern: str = "",            # "browsing" | "settled" | "watching" | etc.
        user_settled: bool = False,
        rapid_browsing: bool = False,
    ) -> VisionDirective:
        """Pick a reaction for the next vision event.

        The core idea: treat each reaction type as a weighted slot and bias
        the weights using state + mood. This gives organic variety without
        hand-crafted rules for every branch.
        """
        self._tick_pressures()

        w_deep = self._deep_base
        w_glance = self._glance_base
        w_tangent = self._tangent_base
        w_ignore = self._ignore_base
        w_silence = self._silence_base

        # --- Vision cooldown: suppress reactions if we just reacted -------
        # Media/game content is the show itself — use a shorter cooldown so
        # the streamer stays engaged with what the audience is watching.
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

        # --- Change kind --------------------------------------------------
        if change_kind == "delta":
            # A small change within the same scene: skew HARD to glance/ignore.
            # Most deltas should NOT trigger a full reaction.
            w_deep *= 0.35
            w_glance *= 1.30
            w_tangent *= 0.80
            w_ignore *= 1.60
            w_silence *= 1.20
        else:  # "scene" — brand new
            # Fresh scene gets more engagement.
            w_deep *= 1.10
            w_ignore *= 0.85

        # --- Streak fatigue: real streamers don't react back-to-back ------
        streak = self._state.vision_reactions_in_a_row
        if streak >= 1:
            w_deep *= max(0.10, 1.0 - 0.45 * streak)
            w_glance *= max(0.20, 1.0 - 0.25 * streak)
            w_tangent *= 1.20
            w_ignore *= 1.50
            w_silence *= 1.30
        if streak >= 2:
            w_deep *= 0.3
            w_ignore *= 1.40

        # --- Ignored streak: eventually force re-engagement ---------------
        if self._state.ignored_scenes_in_a_row >= 5:
            w_ignore *= 0.5
            w_deep *= 1.3
            w_glance *= 1.2

        # --- Scene age (novelty) ------------------------------------------
        # Very fresh: slight bump but nothing dramatic — a real streamer
        # doesn't react just because a tab switched.
        if scene_age_sec < 6.0:
            w_deep *= 1.08
            w_silence *= 0.90
        elif scene_age_sec > 45.0:
            w_deep *= 0.40
            w_glance *= 1.10
            w_ignore *= 1.40

        # --- Mood biases --------------------------------------------------
        # Focused + energized → deep. Scattered/tired → glance or tangent.
        eng = vision_engagement
        w_deep *= 0.6 + eng
        w_glance *= 1.1 - 0.4 * eng
        # Low focus = tangent-prone (the classic "wait, this reminds me of...").
        if mood_focus < 0.50:
            w_tangent *= 1.35
        # Low arousal → prefer silence/ignore over deep reaction.
        if mood_arousal < 0.35:
            w_silence *= 1.4
            w_ignore *= 1.2
            w_deep *= 0.7
        # Low talkativity → silence is the organic choice.
        if mood_talkativity < 0.35:
            w_silence *= 1.6
            w_deep *= 0.8
        # High arousal → less silence, more engagement.
        if mood_arousal > 0.75:
            w_silence *= 0.4
            w_deep *= 1.1

        # --- Monologue flow: a real streamer doesn't break mid-thought -----
        if in_monologue_flow:
            w_tangent *= 1.15
            w_deep *= 0.30       # almost never break a good monologue for screen
            w_glance *= 0.50
            w_ignore *= 1.80
            w_silence *= 1.30

        # --- Tangent pressure: enable the rich path when seeds are available
        if not tangent_seeds:
            w_tangent *= 0.4    # no material → don't try to fake a tangent
        else:
            w_tangent *= 0.7 + self._state.tangent_pressure * 1.2

        # --- Topic anchor present: slightly favor tangent (tie screen to topic)
        if has_topic:
            w_tangent *= 1.15

        # --- Screen activity biases (v4) ------------------------------------
        # Adapt reaction based on WHAT the user is doing with the screen.
        if screen_activity == "scroll":
            # Scrolling: user is browsing. Don't interrupt with deep reactions.
            w_ignore *= 1.6
            w_glance *= 1.3
            w_deep *= 0.4
            w_silence *= 1.2
        elif screen_activity == "typing":
            # User is typing: almost always ignore. Don't talk over their input.
            w_ignore *= 2.5
            w_silence *= 1.8
            w_deep *= 0.2
            w_glance *= 0.5
        elif screen_activity == "app_switch":
            # App switch: new context! Higher chance of deep or glance.
            w_deep *= 1.4
            w_glance *= 1.2
            w_ignore *= 0.5
        elif screen_activity == "navigation":
            # Navigation within same app: moderate engagement.
            w_glance *= 1.3
            w_deep *= 0.9
            w_tangent *= 1.1
        elif screen_activity == "media":
            # Watching video / playing game: engage more than browsing, but
            # still not every frame — real streamers go quiet during action
            # and react to key moments, not a play-by-play.
            w_deep *= 1.10
            w_glance *= 1.20
            w_ignore *= 0.65
            w_silence *= 0.75
            w_tangent *= 1.05

        # Rapid browsing pattern: suppress reactions until user settles.
        if rapid_browsing:
            w_ignore *= 2.0
            w_silence *= 1.5
            w_deep *= 0.25
            w_glance *= 0.6

        # User has settled: they're reading/watching. OK to engage deeper.
        if user_settled and screen_activity not in ("typing",):
            w_deep *= 1.3
            w_ignore *= 0.6
            w_glance *= 1.1

        # --- Organicity dial: at 0, behave deterministically (deep/glance only)
        if self._organicity < 0.2:
            w_tangent *= 0.3
            w_ignore *= 0.3
            w_silence *= 0.2

        # --- Sample ------------------------------------------------------
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

        # --- Build the directive ------------------------------------------
        directive = self._build_directive(
            chosen=chosen,
            mood_arousal=mood_arousal,
            mood_focus=mood_focus,
            change_kind=change_kind,
            tangent_seeds=tangent_seeds,
            weights=weights,
            total=total,
        )
        # --- Update state on this decision -------------------------------
        self._apply_state_update(chosen, change_kind)
        return directive

    # ------------------------------------------------------------------
    # Public: monologue silence bias (used when NO vision event pending)
    # ------------------------------------------------------------------
    def should_hold_silence(
        self,
        *,
        mood_silence_probability: float,
        in_monologue_flow: bool,
    ) -> bool:
        """Returns True if the orchestrator should hold a short silent beat
        instead of generating another monologue. Silence pressure grows when
        you've been yapping non-stop."""
        p = mood_silence_probability
        if in_monologue_flow:
            p += 0.05
        p += self._state.silence_pressure * 0.25
        # Cap.
        p = min(0.45, p)
        return self._rng.random() < p

    # ------------------------------------------------------------------
    # Event hooks (keep counters in sync)
    # ------------------------------------------------------------------
    def on_scene_change(self) -> None:
        now = time.time()
        self._state.last_scene_changed_at = now
        # Scene novelty lasts ~15 seconds.
        self._state.scene_novelty_deadline = now + 15.0

    def on_segment_spoken(self, *, intent_kind: str) -> None:
        if intent_kind == "vision":
            self._state.vision_reactions_in_a_row += 1
            self._state.ignored_scenes_in_a_row = 0
            self._state.tangent_pressure = max(0.0, self._state.tangent_pressure - 0.2)
        else:
            self._state.vision_reactions_in_a_row = 0
        # After every segment, nudge counters.
        self._state.silence_pressure = min(1.0, self._state.silence_pressure + 0.08)
        self._state.tangent_pressure = min(1.0, self._state.tangent_pressure + 0.05)

    def on_silence_beat(self) -> None:
        self._state.silence_pressure = max(0.0, self._state.silence_pressure - 0.3)

    def on_vision_ignored(self) -> None:
        self._state.ignored_scenes_in_a_row += 1

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _tick_pressures(self) -> None:
        """Slow growth so the system doesn't stay at 0 forever."""
        # Pressures decay slightly each decision to prevent runaway.
        self._state.tangent_pressure = max(
            0.0, self._state.tangent_pressure - 0.01
        )
        self._state.silence_pressure = max(
            0.0, self._state.silence_pressure - 0.01
        )

    def _apply_state_update(self, chosen: VisionReaction, change_kind: str) -> None:
        now = time.time()
        self._state.last_vision_reaction_ts = now
        if chosen == VisionReaction.IGNORE:
            self._state.ignored_scenes_in_a_row += 1
            self._state.vision_reactions_in_a_row = 0
        elif chosen == VisionReaction.SILENCE:
            self._state.silence_pressure = max(0.0, self._state.silence_pressure - 0.4)
        else:
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
        # Number of sentences per reaction type — soft guide for prompt.
        if chosen == VisionReaction.DEEP:
            target = 1 if mood_arousal < 0.7 else 2
        elif chosen == VisionReaction.TANGENT:
            target = 2
        elif chosen == VisionReaction.GLANCE:
            target = 1
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

        # Rationale string for dashboard/logging — human-readable.
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
