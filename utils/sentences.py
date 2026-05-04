"""Turn a token stream into a sentence stream so TTS can start before the LLM finishes.

v4 stability notes:
  * v3's aggressive ", and / , but / , so" soft-break was the cause of a
    "cuts off then resumes" failure mode: a normal mid-thought sentence
    like "I'm watching Omni-Man demolish this skyscraper, and he's just
    standing there." would be chopped at ", and " into two separate TTS
    calls, which the listener perceived as the streamer truncating itself.
  * v4 keeps the FIRST-sentence aggressive emit (low TTFA) but only
    soft-breaks on em-dashes and semicolons — actual prosodic pauses,
    not grammatical glue.
  * The soft-break threshold is also bumped to 160 chars so it only
    fires on genuine run-ons, not on normal-length sentences.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import AsyncIterator

# Sentence terminators for TR + EN. Minimum length guards against abbreviations.
_SENTENCE_END = re.compile(r"([\.!\?…]+|\n{2,})")
# Minimum length for a NON-FIRST sentence to be emitted. Short enough to
# catch 8-word punchlines; long enough that "Mr." doesn't split us.
_MIN_SENTENCE_LEN = 14
# For the FIRST sentence of a segment we use a lower floor — latency is
# dominated by time-to-first-audio, and a short opener is often exactly
# right ("Okay." "Right." "Hmm.").
_FIRST_SENTENCE_MIN_LEN = 6
# Soft-break fallback: only fires for genuinely runaway sentences (>160 chars
# with no terminator yet). At v3's 80-char threshold this triggered on normal
# mid-thought sentences and broke them mid-clause.
_SOFT_BREAK_AFTER_CHARS = 160
# Soft-break patterns — em-dashes and semicolons are natural prosodic pauses
# that read cleanly when chopped. Conjunction-based breaks (", and / but /
# so / because") are NOT included because they sit inside a continuing
# thought and produce the audible "cut off then resume" effect.
_SOFT_BREAK = re.compile(r"(—|;)")

# Short phrases that should be treated as complete sentences on their own.
_SHORT_COMPLETE = {
    "ok.", "evet.", "hayır.", "yes.", "no.", "hmm.", "vay.", "aha.",
    "doğru.", "right.", "okay.", "tamam.", "yeah.", "nope.", "nah.",
    "wait.", "hold on.", "oh.",
}


@dataclass
class SentenceStreamer:
    """Split an incremental token stream into sentence chunks.

    Motivation: feeding the TTS engine sentence by sentence minimises
    time-to-first-audio while keeping prosody natural. The trailing buffer
    is emitted on flush().

    The first emitted chunk is treated more aggressively (lower min_len,
    soft-break permitted earlier) so the viewer hears audio quickly.
    """

    min_len: int = _MIN_SENTENCE_LEN
    first_min_len: int = _FIRST_SENTENCE_MIN_LEN
    soft_break_after: int = _SOFT_BREAK_AFTER_CHARS
    _buffer: str = field(default="", init=False)
    _emitted_any: bool = field(default=False, init=False)

    def feed(self, token: str) -> list[str]:
        """Append a token, return any sentences that are now complete."""
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
        """Return whatever remains when the upstream stream ends."""
        rest = self._buffer.strip()
        self._buffer = ""
        self._emitted_any = True
        return [rest] if rest else []

    def reset(self) -> None:
        """Prepare the streamer to be reused for a new segment."""
        self._buffer = ""
        self._emitted_any = False

    @staticmethod
    def _looks_like_complete(s: str) -> bool:
        return s.strip().lower() in _SHORT_COMPLETE


async def stream_sentences(
    token_iter: AsyncIterator[str],
    min_len: int = _MIN_SENTENCE_LEN,
) -> AsyncIterator[str]:
    """Adapt an async token iterator to an async sentence iterator."""
    streamer = SentenceStreamer(min_len=min_len)
    async for token in token_iter:
        for sent in streamer.feed(token):
            yield sent
    for sent in streamer.flush():
        yield sent


def _word_ngrams(text: str, n: int) -> set:
    """Return a set of space-joined word n-grams from *text*."""
    words = text.split()
    if len(words) < n:
        return {text} if words else set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def _ngram_jaccard(a_norm: str, b_norm: str, n: int) -> float:
    """Jaccard similarity over word n-grams. Returns 0-1."""
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
    """Paraphrase-aware similarity in [0, 1].

    Blends three signals so both exact rewording and paraphrasing are caught:
      1. Character-level SequenceMatcher ratio  -- exact/near-exact matches
      2. Bigram Jaccard                          -- catches reordering + synonyms
      3. Trigram Jaccard (when text is long enough)
    """
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
