"""Scene change classification and screen activity types."""
from __future__ import annotations

from enum import Enum


class ChangeType(Enum):
    NONE = "none"
    DELTA = "delta"
    SCENE_CHANGE = "scene"
    IDLE = "idle"


class ScreenActivity(Enum):
    STATIC = "static"
    SCROLL = "scroll"
    NAVIGATION = "navigation"
    APP_SWITCH = "app_switch"
    MEDIA_PLAYING = "media"
    TYPING = "typing"
    MICRO = "micro"
