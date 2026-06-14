from __future__ import annotations

import threading

_lock = threading.Lock()
_value = ""


def set_activity(text: str) -> None:
    global _value
    with _lock:
        _value = (text or "").strip()


def get_activity() -> str:
    with _lock:
        return _value
