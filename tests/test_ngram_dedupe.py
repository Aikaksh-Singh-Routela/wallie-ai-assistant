"""Behavioural regression tests for the paraphrase-aware n-gram dedupe.

Run with:  python -m pytest tests/ -v
"""
import pytest
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from utils.sentences import similarity, _ngram_jaccard, _word_ngrams


# ─────────────────────────────────────────────────────────────────────
# _word_ngrams
# ─────────────────────────────────────────────────────────────────────

def test_bigrams_basic():
    ng = _word_ngrams("the quick brown fox", 2)
    assert "the quick" in ng
    assert "quick brown" in ng
    assert "brown fox" in ng
    assert len(ng) == 3


def test_unigrams_short():
    # When text shorter than n, returns the whole text as one entry.
    ng = _word_ngrams("hi", 2)
    assert ng == {"hi"}


def test_empty_text():
    assert _word_ngrams("", 2) == set()


# ─────────────────────────────────────────────────────────────────────
# _ngram_jaccard
# ─────────────────────────────────────────────────────────────────────

def test_jaccard_identical():
    a = "the quick brown fox"
    assert _ngram_jaccard(a, a, 2) == 1.0


def test_jaccard_disjoint():
    a = "apple orange mango"
    b = "car truck motorbike"
    assert _ngram_jaccard(a, b, 2) == 0.0


def test_jaccard_partial():
    a = "the quick brown fox"
    b = "the quick lazy dog"
    j = _ngram_jaccard(a, b, 2)
    assert 0.0 < j < 1.0


# ─────────────────────────────────────────────────────────────────────
# similarity — exact/near-exact matches
# ─────────────────────────────────────────────────────────────────────

def test_exact_match():
    s = "That was absolutely insane."
    assert similarity(s, s) == pytest.approx(1.0)


def test_near_exact_high():
    a = "That was absolutely insane."
    b = "That was absolutely insane!"
    assert similarity(a, b) > 0.85


def test_unrelated_low():
    a = "I love playing video games on the weekend."
    b = "The weather forecast shows rain tomorrow morning."
    assert similarity(a, b) < 0.35


# ─────────────────────────────────────────────────────────────────────
# similarity — paraphrase-awareness (the NEW behaviour)
# ─────────────────────────────────────────────────────────────────────

def test_paraphrase_synonym_caught():
    """Synonym swap should still register as high similarity."""
    a = "That is absolutely incredible and I love it."
    b = "That is absolutely amazing and I love it."
    assert similarity(a, b) > 0.65, (
        "Synonym paraphrase ('incredible' → 'amazing') should score > 0.65"
    )


def test_paraphrase_reorder_caught():
    """Word reordering should score high when most n-grams overlap."""
    a = "honestly this is the best thing I've seen all stream"
    b = "this honestly is the best thing all stream I've seen"
    assert similarity(a, b) > 0.55


def test_paraphrase_different_enough():
    """A genuine follow-up sentence sharing only common words should score low."""
    a = "Let me tell you something wild."
    b = "So the reason this is interesting is that nobody expected it."
    assert similarity(a, b) < 0.45


def test_short_vs_long_no_false_positive():
    """Short filler vs long sentence should not trigger dedupe."""
    a = "Right."
    b = "Right, so the thing about Elden Ring is that every boss is secretly a tutorial for the next one."
    assert similarity(a, b) < 0.40


# ─────────────────────────────────────────────────────────────────────
# Conversation.is_repeat regression
# ─────────────────────────────────────────────────────────────────────

# Direct import to avoid core/__init__ → orchestrator → sounddevice chain
import importlib as _il, sys as _sys
_ctx_spec = _il.util.spec_from_file_location(
    "wallie_context",
    pathlib.Path(__file__).parent.parent / "core" / "context.py"
)
_ctx = _il.util.module_from_spec(_ctx_spec)
_sys.modules["wallie_context"] = _ctx
_ctx_spec.loader.exec_module(_ctx)
Conversation = _ctx.Conversation


def test_is_repeat_exact():
    conv = Conversation()
    conv.add_assistant("I've never seen anything like this before.")
    assert conv.is_repeat("I've never seen anything like this before.", window=8, threshold=0.78)


def test_is_repeat_paraphrase():
    conv = Conversation()
    conv.add_assistant("This is genuinely the most impressive thing I have seen today.")
    candidate = "This is genuinely the most impressive thing I have seen all day."
    assert conv.is_repeat(candidate, window=8, threshold=0.78), (
        "Near-paraphrase with only 'today' → 'all day' should be caught as repeat"
    )


def test_is_not_repeat_different_topic():
    conv = Conversation()
    conv.add_assistant("Elden Ring bosses are all secretly tutorials for the next area.")
    candidate = "The economy in Baldur's Gate 3 is completely broken in the best way."
    assert not conv.is_repeat(candidate, window=8, threshold=0.78)


def test_is_repeat_window_respected():
    """A sentence outside the dedupe window should NOT be flagged."""
    conv = Conversation()
    repeated = "Chat I genuinely cannot believe this happened."
    conv.add_assistant(repeated)
    for i in range(10):
        conv.add_assistant(f"Filler segment number {i} about something completely different.")
    # Window is 4, but the repeated sentence is 11 turns back.
    assert not conv.is_repeat(repeated, window=4, threshold=0.78)
