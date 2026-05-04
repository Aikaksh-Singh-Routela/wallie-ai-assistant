"""Screen capture using mss. Fast on Windows (GDI-backed)."""
from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Optional

import mss
from PIL import Image


@dataclass
class Frame:
    jpeg: bytes  # Already encoded; cheaper to keep and re-send.
    width: int
    height: int
    mime: str = "image/jpeg"


class ScreenCapture:
    def __init__(self, *, monitor_index: int = 1, max_edge_px: int = 768, jpeg_quality: int = 80) -> None:
        self._monitor_index = monitor_index
        self._max_edge = max_edge_px
        self._quality = jpeg_quality
        # mss instances are not threadsafe; one per thread.
        self._sct: Optional[mss.mss] = None

    def _ensure(self) -> mss.mss:
        if self._sct is None:
            self._sct = mss.mss()
        return self._sct

    def grab(self) -> Frame:
        sct = self._ensure()
        mon = sct.monitors[self._monitor_index]
        raw = sct.grab(mon)
        img = Image.frombytes("RGB", raw.size, raw.rgb)

        # Downscale so the long edge is at most max_edge_px. This is the single
        # biggest latency and cost win for vision models.
        w, h = img.size
        scale = min(1.0, self._max_edge / max(w, h))
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=self._quality)
        return Frame(jpeg=buf.getvalue(), width=img.size[0], height=img.size[1])

    def close(self) -> None:
        if self._sct is not None:
            self._sct.close()
            self._sct = None
