"""Unit tests for MemoryStore."""
import json
import time
import pytest
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from pathlib import Path
import importlib, sys
# Import directly to avoid core/__init__ pulling in sounddevice/openai
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
spec = importlib.util.spec_from_file_location("memory_store",
    pathlib.Path(__file__).parent.parent / "core" / "memory_store.py")
_mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(_mod)
MemoryStore = _mod.MemoryStore


# ─────────────────────────────────────────────────────────────────────
# Load / Save round-trip
# ─────────────────────────────────────────────────────────────────────

def test_save_and_load(tmp_path):
    p = tmp_path / "test.memory.json"
    store = MemoryStore(p)
    store.notes = "- covered Elden Ring boss strats\n- made fun of the tutorial"
    store.log_viewer(username="ChatGuy", platform="twitch", text="lol")
    store.save()

    store2 = MemoryStore(p)
    store2.load()
    assert "Elden Ring" in store2.notes
    assert len(store2.viewer_log) == 1
    assert store2.viewer_log[0]["username"] == "ChatGuy"


def test_load_missing_file_is_noop(tmp_path):
    p = tmp_path / "nonexistent.memory.json"
    store = MemoryStore(p)
    store.load()  # should not raise
    assert store.notes == ""
    assert store.viewer_log == []


def test_load_corrupted_json_is_noop(tmp_path):
    p = tmp_path / "bad.memory.json"
    p.write_text("this is not valid json {{{{", encoding="utf-8")
    store = MemoryStore(p)
    store.load()  # should not raise
    assert store.notes == ""


# ─────────────────────────────────────────────────────────────────────
# update_notes
# ─────────────────────────────────────────────────────────────────────

def test_update_notes():
    store = MemoryStore(Path("/tmp/dummy.json"))
    store.update_notes("  - bullet one\n- bullet two  ")
    assert store.notes == "- bullet one\n- bullet two"


def test_update_notes_empty_clears():
    store = MemoryStore(Path("/tmp/dummy.json"))
    store.notes = "some old notes"
    store.update_notes("")
    assert store.notes == ""


# ─────────────────────────────────────────────────────────────────────
# log_viewer
# ─────────────────────────────────────────────────────────────────────

def test_log_viewer_appends():
    store = MemoryStore(Path("/tmp/dummy.json"))
    store.log_viewer(username="Alice", platform="youtube", text="hi there")
    store.log_viewer(username="Bob", platform="twitch", text="lol")
    assert len(store.viewer_log) == 2
    assert store.viewer_log[0]["username"] == "Alice"
    assert store.viewer_log[1]["platform"] == "twitch"


def test_log_viewer_truncates_long_text():
    store = MemoryStore(Path("/tmp/dummy.json"))
    long_text = "x" * 500
    store.log_viewer(username="Spammer", platform="twitch", text=long_text)
    assert len(store.viewer_log[0]["text"]) <= 300


def test_log_viewer_rolling_trim():
    store = MemoryStore(Path("/tmp/dummy.json"))
    for i in range(1100):
        store.log_viewer(username=f"user{i}", platform="twitch", text="msg")
    # Rolling trim fires at 2*MAX_LOG = 1000 entries, trims to 500.
    # After that, 100 more entries are appended before the loop ends → max 600.
    assert len(store.viewer_log) <= 600


# ─────────────────────────────────────────────────────────────────────
# recent_viewers
# ─────────────────────────────────────────────────────────────────────

def test_recent_viewers_limits():
    store = MemoryStore(Path("/tmp/dummy.json"))
    for i in range(200):
        store.log_viewer(username=f"u{i}", platform="twitch", text="x")
    recent = store.recent_viewers(50)
    assert len(recent) == 50
    # Should be newest-last: last logged = u199 at tail
    assert recent[-1]["username"] == "u199"


# ─────────────────────────────────────────────────────────────────────
# summary_for_prompt
# ─────────────────────────────────────────────────────────────────────

def test_summary_for_prompt_empty():
    store = MemoryStore(Path("/tmp/dummy.json"))
    assert store.summary_for_prompt() == ""


def test_summary_for_prompt_truncated():
    store = MemoryStore(Path("/tmp/dummy.json"))
    store.notes = "x" * 1000
    assert len(store.summary_for_prompt(max_chars=200)) == 200


def test_summary_for_prompt_full():
    store = MemoryStore(Path("/tmp/dummy.json"))
    store.notes = "short note"
    assert store.summary_for_prompt() == "short note"


# ─────────────────────────────────────────────────────────────────────
# Save respects max_log cap
# ─────────────────────────────────────────────────────────────────────

def test_save_caps_log_at_500(tmp_path):
    p = tmp_path / "cap.memory.json"
    store = MemoryStore(p)
    for i in range(600):
        store.log_viewer(username=f"u{i}", platform="twitch", text="x")
    store.save()
    data = json.loads(p.read_text())
    assert len(data["viewer_log"]) <= 500
