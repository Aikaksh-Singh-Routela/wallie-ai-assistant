"""MoodEngine — slow-evolving energy/mood state for the streamer."""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MoodState:
    arousal: float = 0.55
    valence: float = 0.15
    focus: float = 0.75
    talkativity: float = 0.70
    mood_label: str = "warm"
    # Internal tracking.
    last_update: float = field(default_factory=time.time)
    spoken_in_a_row: int = 0
    silent_in_a_row: int = 0
    session_started_at: float = field(default_factory=time.time)


class MoodEngine:
    # Baselines.
    _BASE_AROUSAL = 0.55
    _BASE_VALENCE = 0.15
    _BASE_FOCUS = 0.75
    _BASE_TALK = 0.70

    _FATIGUE_PER_HOUR = 0.12
    _SCATTER_PER_HOUR = 0.08

    def __init__(self, *, base_energy: str = "warm", seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)
        self._state = MoodState()
        self._configure_from_energy(base_energy)

    def _configure_from_energy(self, energy: str) -> None:
        table = {
            "chill":    (0.40, 0.10, 0.70, 0.55, "chill"),
            "warm":     (0.55, 0.15, 0.75, 0.70, "warm"),
            "hyped":    (0.75, 0.25, 0.70, 0.85, "hyped"),
            "unhinged": (0.85, 0.10, 0.50, 0.90, "wired"),
        }
        a, v, f, t, label = table.get(energy, table["warm"])
        self._BASE_AROUSAL = a
        self._BASE_VALENCE = v
        self._BASE_FOCUS = f
        self._BASE_TALK = t
        self._state = MoodState(
            arousal=a, valence=v, focus=f, talkativity=t, mood_label=label,
        )

    @property
    def state(self) -> MoodState:
        return self._state

    @property
    def arousal(self) -> float:
        return self._state.arousal

    @property
    def valence(self) -> float:
        return self._state.valence

    @property
    def focus(self) -> float:
        return self._state.focus

    @property
    def talkativity(self) -> float:
        return self._state.talkativity

    @property
    def label(self) -> str:
        return self._state.mood_label

    def describe(self) -> str:
        s = self._state
        if s.arousal >= 0.80:
            a = "buzzing with energy"
        elif s.arousal >= 0.60:
            a = "locked in"
        elif s.arousal >= 0.40:
            a = "steady"
        elif s.arousal >= 0.25:
            a = "winding down, low-key"
        else:
            a = "sleepy, wrung out"

        if s.valence >= 0.30:
            v = "in a good mood"
        elif s.valence >= 0.0:
            v = "neutral"
        elif s.valence >= -0.30:
            v = "mildly salty"
        else:
            v = "cranky, over it"

        if s.focus >= 0.75:
            f = "sharp and on-topic"
        elif s.focus >= 0.55:
            f = "mostly on-topic"
        elif s.focus >= 0.35:
            f = "prone to tangents"
        else:
            f = "scatterbrained, jumping around"
        return f"{a}, {v}, {f}"

    def on_highlight_chat(self) -> None:
        self._state.arousal = min(1.0, self._state.arousal + 0.25)
        self._state.valence = min(1.0, self._state.valence + 0.25)
        self._state.talkativity = min(1.0, self._state.talkativity + 0.15)
        self._state.focus = min(1.0, self._state.focus + 0.10)
        self._retag()

    def on_ordinary_chat(self) -> None:
        self._state.arousal = min(1.0, self._state.arousal + 0.05)
        self._state.valence = min(1.0, self._state.valence + 0.03)
        self._state.talkativity = min(1.0, self._state.talkativity + 0.05)
        self._retag()

    def on_scene_change(self) -> None:
        self._state.arousal = min(1.0, self._state.arousal + 0.06)
        self._state.focus = min(1.0, self._state.focus + 0.04)
        self._retag()

    def on_boring_stretch(self) -> None:
        self._state.arousal = max(0.0, self._state.arousal - 0.04)
        self._state.talkativity = max(0.0, self._state.talkativity - 0.03)
        self._retag()

    def on_segment_spoken(self, *, length_chars: int = 0) -> None:
        self._state.spoken_in_a_row += 1
        self._state.silent_in_a_row = 0
        if length_chars > 600:
            self._state.arousal = max(0.0, self._state.arousal - 0.06)
            self._state.talkativity = max(0.0, self._state.talkativity - 0.12)
        else:
            self._state.talkativity = max(0.0, self._state.talkativity - 0.04)
        if self._state.spoken_in_a_row > 3:
            self._state.talkativity = max(0.0, self._state.talkativity - 0.05)
        self._retag()

    def on_silence_beat(self) -> None:
        self._state.silent_in_a_row += 1
        self._state.spoken_in_a_row = 0
        self._state.talkativity = min(1.0, self._state.talkativity + 0.12)
        self._retag()

    def tick(self) -> None:
        now = time.time()
        dt = now - self._state.last_update
        self._state.last_update = now
        if dt <= 0:
            return

        tau = 90.0
        decay = 1.0 - math.exp(-dt / tau)
        self._state.arousal += (self._BASE_AROUSAL - self._state.arousal) * decay * 0.6
        self._state.valence += (self._BASE_VALENCE - self._state.valence) * decay * 0.6
        self._state.focus += (self._BASE_FOCUS - self._state.focus) * decay * 0.4
        self._state.talkativity += (self._BASE_TALK - self._state.talkativity) * decay * 0.5

        hours = (now - self._state.session_started_at) / 3600.0
        fatigue = min(0.35, self._FATIGUE_PER_HOUR * hours)
        scatter = min(0.30, self._SCATTER_PER_HOUR * hours)
        self._state.arousal = max(0.10, self._state.arousal - fatigue * decay)
        self._state.focus = max(0.20, self._state.focus - scatter * decay)

        self._state.arousal += (self._rng.random() - 0.5) * 0.01
        self._state.valence += (self._rng.random() - 0.5) * 0.01
        self._state.focus += (self._rng.random() - 0.5) * 0.01
        self._state.talkativity += (self._rng.random() - 0.5) * 0.01

        self._state.arousal = _clip01(self._state.arousal)
        self._state.focus = _clip01(self._state.focus)
        self._state.talkativity = _clip01(self._state.talkativity)
        self._state.valence = max(-1.0, min(1.0, self._state.valence))

        self._retag()

    def llm_temperature_bias(self) -> float:
        base = 0.0
        base += (self._state.arousal - 0.55) * 0.25
        base += (0.5 - self._state.focus) * 0.15
        return max(-0.20, min(0.25, base))

    def silence_probability(self) -> float:
        p = 0.0
        p += (1.0 - self._state.talkativity) * 0.35
        if self._state.spoken_in_a_row >= 3:
            p += 0.15
        if self._state.arousal < 0.35:
            p += 0.10
        return min(0.55, p)

    def wants_vision_engagement(self) -> float:
        eng = 0.5 * self._state.arousal + 0.5 * self._state.focus
        if self._state.spoken_in_a_row >= 2:
            eng -= 0.10
        return _clip01(eng)

    def _retag(self) -> None:
        a, v, f = self._state.arousal, self._state.valence, self._state.focus
        if a >= 0.78 and v <= -0.12:
            self._state.mood_label = "tilted"
        elif a >= 0.75 and f >= 0.80:
            self._state.mood_label = "locked in"
        elif a >= 0.80 and v >= 0.10:
            self._state.mood_label = "wired"
        elif a >= 0.70 and v >= 0.25:
            self._state.mood_label = "hyped"
        elif a >= 0.55 and v >= 0.10:
            self._state.mood_label = "warm"
        elif a >= 0.55 and v < -0.10:
            self._state.mood_label = "salty"
        elif a < 0.35 and v >= 0.0:
            self._state.mood_label = "mellow"
        elif a < 0.35 and v < 0.0:
            self._state.mood_label = "sleepy"
        elif f < 0.40:
            self._state.mood_label = "scattered"
        else:
            self._state.mood_label = "steady"


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))
