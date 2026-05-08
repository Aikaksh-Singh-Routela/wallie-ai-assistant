"""Turn a token stream into a sentence stream so TTS can start before the LLM finishes."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import AsyncIterator

_SENTENCE_END = re.compile(r"([\.!\?…]+|\n{2,})")
_MIN_SENTENCE_LEN = 14
_FIRST_SENTENCE_MIN_LEN = 6
_SOFT_BREAK_AFTER_CHARS = 160
_SOFT_BREAK = re.compile(r"(—|;)")

_SHORT_COMPLETE = {
    "ok.", "evet.", "hayır.", "yes.", "no.", "hmm.", "vay.", "aha.",
    "doğru.", "right.", "okay.", "tamam.", "yeah.", "nope.", "nah.",
    "wait.", "hold on.", "oh.",
}


@dataclass
class SentenceStreamer:

    min_len: int = _MIN_SENTENCE_LEN
    first_min_len: int = _FIRST_SENTENCE_MIN_LEN
    soft_break_after: int = _SOFT_BREAK_AFTER_CHARS
    _buffer: str = field(default="", init=False)
    _emitted_any: bool = field(default=False, init=False)

    def feed(self, token: str) -> list[str]:
        if not token:
            return []
        self._buffer += token
        out: list[str] = []
        while True:
            # Try hard terminator first.
            m = _SENTENCE_END.search(self._buffer)
            if m:
                end_idx = m.end()
                candidate = self._buffer[:end_idx].strip()
                min_needed = self.first_min_len if not self._emitted_any else self.min_len
                if len(candidate) < min_needed and not self._looks_like_complete(candidate):
                    break
                out.append(candidate)
                self._buffer = self._buffer[end_idx:].lstrip()
                self._emitted_any = True
                continue

            # No hard terminator. If the buffer is long enough, try a soft break.
            if len(self._buffer) >= self.soft_break_after:
                sb = _SOFT_BREAK.search(self._buffer)
                if sb:
                    end_idx = sb.end()
                    candidate = self._buffer[:end_idx].rstrip(" ,;—").strip()
                    # Force termination so TTS gets clean prosody.
                    if candidate and not candidate.endswith((".", "!", "?", "…")):
                        candidate += "."
                    out.append(candidate)
                    self._buffer = self._buffer[end_idx:].lstrip()
                    self._emitted_any = True
                    continue
            break
        return out

    def flush(self) -> list[str]:
        rest = self._buffer.strip()
        self._buffer = ""
        self._emitted_any = True
        return [rest] if rest else []

    def reset(self) -> None:
        self._buffer = ""
        self._emitted_any = False

    @staticmethod
    def _looks_like_complete(s: str) -> bool:
        return s.strip().lower() in _SHORT_COMPLETE


async def stream_sentences(
    token_iter: AsyncIterator[str],
    min_len: int = _MIN_SENTENCE_LEN,
) -> AsyncIterator[str]:
    streamer = SentenceStreamer(min_len=min_len)
    async for token in token_iter:
        for sent in streamer.feed(token):
            yield sent
    for sent in streamer.flush():
        yield sent


def _word_ngrams(text: str, n: int) -> set:
    words = text.split()
    if len(words) < n:
        return {text} if words else set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def _ngram_jaccard(a_norm: str, b_norm: str, n: int) -> float:
    ag = _word_ngrams(a_norm, n)
    bg = _word_ngrams(b_norm, n)
    if not ag and not bg:
        return 1.0
    if not ag or not bg:
        return 0.0
    inter = len(ag & bg)
    union = len(ag | bg)
    return inter / union if union else 0.0


def similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    a_norm = re.sub(r"\s+", " ", a.lower().strip())
    b_norm = re.sub(r"\s+", " ", b.lower().strip())

    char_ratio = SequenceMatcher(None, a_norm, b_norm).ratio()
    bi = _ngram_jaccard(a_norm, b_norm, 2)

    a_wc = len(a_norm.split())
    b_wc = len(b_norm.split())
    tri = _ngram_jaccard(a_norm, b_norm, 3) if a_wc >= 3 and b_wc >= 3 else bi

    ngram_blend = (bi + tri) / 2.0
    return max(char_ratio, ngram_blend)
