"""Single conversation history for the streamer.

Long streams (1h+) need more than a flat rolling buffer. We keep the last N
assistant turns verbatim — useful for dedupe and tight continuity — and fold
everything older into a "session_notes" string that the orchestrator updates
periodically with an LLM summarizer.

The system prompt then receives:

    <persona block>

    SESSION NOTES (everything you said earlier, compressed):
    <session_notes>

    <recent verbatim history as chat messages>

This bounds prompt size while preserving the show's continuity across hours.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Optional

from utils.sentences import similarity

Role = Literal["system", "user", "assistant"]


@dataclass
class ImageBlock:
    data: bytes
    mime: str = "image/jpeg"


@dataclass
class Message:
    role: Role
    content: str
    images: list[ImageBlock] = field(default_factory=list)
    source: str = ""  # "monologue" | "chat:platform:user" | "vision" | "user" | ...
    ts: float = field(default_factory=time.time)


@dataclass
class Conversation:
    """Sliding-window history with a rolling summary for long sessions."""

    # Hard caps; reachable only if the summarizer is asleep.
    max_messages: int = 200
    max_chars: int = 60000
    # How many assistant turns we keep verbatim before they may be summarized.
    recent_verbatim_turns: int = 24
    # Sliding dedupe window for repeat detection.
    recent_assistant_window: int = 30

    _messages: list[Message] = field(default_factory=list)
    _recent_assistant: list[str] = field(default_factory=list)
    session_notes: str = ""
    session_started_at: float = field(default_factory=time.time)
    total_segments: int = 0

    # ----- mutation -----
    def add_user(
        self,
        content: str,
        source: str = "user",
        images: Optional[list[ImageBlock]] = None,
    ) -> None:
        self._messages.append(
            Message(role="user", content=content, source=source, images=images or [])
        )
        self._enforce_caps()

    def add_assistant(self, content: str, source: str = "monologue") -> None:
        if not content.strip():
            return
        self._messages.append(Message(role="assistant", content=content.strip(), source=source))
        self._recent_assistant.append(content.strip())
        if len(self._recent_assistant) > self.recent_assistant_window:
            self._recent_assistant.pop(0)
        self.total_segments += 1
        self._enforce_caps()

    def clear(self) -> None:
        self._messages.clear()
        self._recent_assistant.clear()
        self.session_notes = ""
        self.total_segments = 0
        self.session_started_at = time.time()

    # ----- queries -----
    def messages(self) -> list[Message]:
        return list(self._messages)

    def recent_assistant_text(self, n: int = 10) -> list[str]:
        return self._recent_assistant[-n:]

    def is_repeat(self, candidate: str, window: int, threshold: float) -> bool:
        if not candidate.strip():
            return True
        for prior in self._recent_assistant[-window:]:
            if similarity(candidate, prior) >= threshold:
                return True
        return False

    def session_seconds(self) -> float:
        return time.time() - self.session_started_at

    # ----- summarizer hooks -----
    def messages_eligible_for_summary(self) -> list[Message]:
        """Return assistant turns older than the verbatim window. The orchestrator
        passes these to the summarizer LLM and then calls compact_history()."""
        # Walk from the start until we have all but the last `recent_verbatim_turns`
        # assistant turns. Keep user turns interleaved with their assistants so the
        # summary has context.
        assistant_idxs = [i for i, m in enumerate(self._messages) if m.role == "assistant"]
        if len(assistant_idxs) <= self.recent_verbatim_turns:
            return []
        cutoff = assistant_idxs[-self.recent_verbatim_turns]
        return [m for m in self._messages[:cutoff] if m.role in ("assistant", "user")]

    def compact_history(self, new_session_notes: str) -> None:
        """After summarizing, drop everything older than the verbatim window."""
        assistant_idxs = [i for i, m in enumerate(self._messages) if m.role == "assistant"]
        if len(assistant_idxs) <= self.recent_verbatim_turns:
            self.session_notes = new_session_notes.strip()
            return
        cutoff = assistant_idxs[-self.recent_verbatim_turns]
        self._messages = self._messages[cutoff:]
        self.session_notes = new_session_notes.strip()

    # ----- internals -----
    def _enforce_caps(self) -> None:
        if len(self._messages) > self.max_messages:
            # Drop oldest non-system messages.
            overflow = len(self._messages) - self.max_messages
            kept: list[Message] = []
            dropped = 0
            for m in self._messages:
                if dropped < overflow and m.role != "system":
                    dropped += 1
                    continue
                kept.append(m)
            self._messages = kept
        total = sum(len(m.content) for m in self._messages)
        i = 0
        while total > self.max_chars and i < len(self._messages):
            if self._messages[i].role == "system":
                i += 1
                continue
            total -= len(self._messages.pop(i).content)

    # ----- provider export -----
    def to_provider_messages(self, system_prompt: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        for m in self._messages:
            if m.role == "system":
                continue
            if m.images:
                blocks: list[dict[str, Any]] = [{"type": "text", "text": m.content}]
                for img in m.images:
                    blocks.append({"type": "image", "data": img.data, "mime": img.mime})
                out.append({"role": m.role, "content": blocks})
            else:
                out.append({"role": m.role, "content": m.content})
        return out


def pick_topic(topics: Iterable[str], exclude_recent: list[str]) -> Optional[str]:
    topics = [t for t in topics if t.strip()]
    if not topics:
        return None
    last = exclude_recent[-1] if exclude_recent else None
    fresh = [t for t in topics if t != last] or topics
    import random
    return random.choice(fresh)
