"""Scene change classification.

Distinguishes between three meaningful change levels so the orchestrator can
react proportionally — a tiny cursor move doesn't deserve the same energy as
switching to a completely different application.

v4: ScreenActivity enum — classifies *what the user did* (scroll, navigate,
    switch apps, etc.) so the AI can adapt organically.
"""
from __future__ import annotations

from enum import Enum


class ChangeType(Enum):
    NONE = "none"
    # min_change_threshold <= hamming distance < scene_change_threshold
    DELTA = "delta"
    # hamming distance >= scene_change_threshold → entirely new scene
    SCENE_CHANGE = "scene"
    # image variance below idle_variance_threshold → blank / desktop / static
    IDLE = "idle"


class ScreenActivity(Enum):
    """What the user is doing with the screen right now.

    Detected from frame-to-frame comparison patterns. The AI uses this to
    shape its response so it sounds like IT is the one controlling the screen.
    """
    STATIC = "static"           # Nothing moved
    SCROLL = "scroll"           # Content scrolled vertically/horizontally
    NAVIGATION = "navigation"   # Same app, new page/content loaded
    APP_SWITCH = "app_switch"   # Different application / window entirely
    MEDIA_PLAYING = "media"     # Video/animation running (continuous flux)
    TYPING = "typing"           # Small localized text-area changes
    MICRO = "micro"             # Cursor blink, hover highlight, minor UI tick
