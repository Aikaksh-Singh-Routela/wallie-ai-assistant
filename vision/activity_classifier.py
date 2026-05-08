"""Screen activity classifier — detects scroll, navigation, app switch, etc. from frame diffs."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional, Tuple

import numpy as np
from .capture import Frame
from .scene_classifier import ScreenActivity

GRID_ROWS = 8
GRID_COLS = 6
HISTORY_LEN = 12
PHASE_CORR_THRESHOLD = 0.15


@dataclass
class ActivityResult:
    activity: ScreenActivity
    confidence: float = 0.0
    scroll_direction: str = ""
    scroll_speed: float = 0.0
    changed_ratio: float = 0.0
    chrome_stable: bool = True
    content_region_focus: float = 0.0


def _frame_to_gray(frame: Frame) -> np.ndarray:
    return np.array(frame.to_pil().convert("L"), dtype=np.float32)


def _cell_hashes(gray: np.ndarray, rows: int = GRID_ROWS, cols: int = GRID_COLS) -> np.ndarray:
    h, w = gray.shape
    ch, cw = h // rows, w // cols
    trimmed = gray[: rows * ch, : cols * cw]
    return trimmed.reshape(rows, ch, cols, cw).mean(axis=(1, 3))


def _phase_correlate(prev_gray: np.ndarray, curr_gray: np.ndarray) -> Tuple[float, float, float]:
    """Estimate translation between two grayscale frames via phase correlation.

    Returns (dy, dx, peak_value). dy>0 means content shifted down (user scrolled up).
    peak_value indicates confidence — higher is better.
    """
    # Ensure same size.
    h = min(prev_gray.shape[0], curr_gray.shape[0])
    w = min(prev_gray.shape[1], curr_gray.shape[1])
    a = prev_gray[:h, :w]
    b = curr_gray[:h, :w]

    # Windowing to reduce edge artifacts.
    wy = np.hanning(h).reshape(-1, 1)
    wx = np.hanning(w).reshape(1, -1)
    window = wy * wx
    a = a * window
    b = b * window

    # Cross-power spectrum.
    fa = np.fft.fft2(a)
    fb = np.fft.fft2(b)
    cross = fa * np.conj(fb)
    denom = np.abs(cross)
    denom[denom < 1e-10] = 1e-10
    cross_norm = cross / denom
    cc = np.fft.ifft2(cross_norm).real

    peak_idx = np.unravel_index(np.argmax(cc), cc.shape)
    peak_val = cc[peak_idx]

    dy = float(peak_idx[0])
    dx = float(peak_idx[1])
    # Unwrap: if > half the dimension, it's a negative shift.
    if dy > h / 2:
        dy -= h
    if dx > w / 2:
        dx -= w

    return dy, dx, float(peak_val)


class ScreenActivityClassifier:
    """Classifies user screen activity from consecutive frame comparisons."""

    def __init__(
        self,
        *,
        cell_change_threshold: float = 8.0,
        chrome_rows: int = 1,
    ) -> None:
        self._cell_threshold = cell_change_threshold
        self._chrome_rows = chrome_rows  # how many top/bottom rows count as "chrome"
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_cells: Optional[np.ndarray] = None
        self._history: Deque[ScreenActivity] = deque(maxlen=HISTORY_LEN)
        self._consecutive_media: int = 0

    def classify(self, frame: Frame) -> ActivityResult:
        gray = _frame_to_gray(frame)
        cells = _cell_hashes(gray)

        if self._prev_gray is None or self._prev_cells is None:
            self._prev_gray = gray
            self._prev_cells = cells
            result = ActivityResult(activity=ScreenActivity.STATIC, confidence=1.0)
            self._history.append(result.activity)
            return result

        # --- Build the change map ---
        cell_diff = np.abs(cells - self._prev_cells)
        changed_mask = cell_diff > self._cell_threshold
        changed_ratio = float(changed_mask.sum()) / changed_mask.size

        # Chrome stability: check top and bottom rows.
        top_stable = not changed_mask[:self._chrome_rows, :].any()
        bottom_stable = not changed_mask[-self._chrome_rows:, :].any()
        chrome_stable = top_stable or bottom_stable

        # Content region: everything except chrome.
        cr = self._chrome_rows
        content_mask = changed_mask[cr:-cr, :] if cr > 0 else changed_mask
        content_changed_ratio = float(content_mask.sum()) / max(1, content_mask.size)

        # How concentrated are changes? (low = spread out, high = focused in one area)
        if changed_mask.any():
            changed_coords = np.argwhere(changed_mask)
            coord_std = float(changed_coords.std(axis=0).mean())
            focus = 1.0 - min(1.0, coord_std / max(GRID_ROWS, GRID_COLS))
        else:
            focus = 0.0

        # Phase correlation only when cells actually changed.
        if changed_ratio >= 0.05:
            dy, dx, peak_val = _phase_correlate(self._prev_gray, gray)
            has_translation = peak_val > PHASE_CORR_THRESHOLD and (abs(dy) > 2 or abs(dx) > 2)
        else:
            dy, dx, peak_val = 0.0, 0.0, 0.0
            has_translation = False

        # --- Classify ---
        result = self._decide(
            changed_ratio=changed_ratio,
            content_changed_ratio=content_changed_ratio,
            chrome_stable=chrome_stable,
            focus=focus,
            has_translation=has_translation,
            dy=dy, dx=dx, peak_val=peak_val,
            changed_mask=changed_mask,
            cell_diff=cell_diff,
        )

        # Update state.
        self._prev_gray = gray
        self._prev_cells = cells
        self._history.append(result.activity)

        # Track consecutive media frames.
        if result.activity == ScreenActivity.MEDIA_PLAYING:
            self._consecutive_media += 1
        else:
            self._consecutive_media = 0

        return result

    def recent_pattern(self) -> str:
        if len(self._history) < 3:
            return "starting"

        recent = list(self._history)[-6:]
        counts: dict[ScreenActivity, int] = {}
        for a in recent:
            counts[a] = counts.get(a, 0) + 1

        static_count = counts.get(ScreenActivity.STATIC, 0)
        scroll_count = counts.get(ScreenActivity.SCROLL, 0)
        nav_count = counts.get(ScreenActivity.NAVIGATION, 0)
        app_count = counts.get(ScreenActivity.APP_SWITCH, 0)
        media_count = counts.get(ScreenActivity.MEDIA_PLAYING, 0)

        # Dominant pattern.
        if media_count >= 3:
            return "watching"
        if static_count >= 4:
            return "settled"
        if scroll_count >= 3:
            return "browsing"
        if nav_count + app_count >= 3:
            return "rapid_switching"
        if scroll_count + nav_count >= 3:
            return "exploring"
        return "mixed"

    def is_user_settled(self) -> bool:
        if len(self._history) < 4:
            return False
        recent = list(self._history)[-4:]
        return all(a in (ScreenActivity.STATIC, ScreenActivity.MICRO) for a in recent)

    def is_rapid_browsing(self) -> bool:
        if len(self._history) < 4:
            return False
        recent = list(self._history)[-4:]
        active = sum(1 for a in recent if a in (
            ScreenActivity.SCROLL, ScreenActivity.NAVIGATION, ScreenActivity.APP_SWITCH
        ))
        return active >= 3

    def reset(self) -> None:
        self._prev_gray = None
        self._prev_cells = None
        self._history.clear()
        self._consecutive_media = 0

    # --- internal ---
    def _decide(
        self,
        *,
        changed_ratio: float,
        content_changed_ratio: float,
        chrome_stable: bool,
        focus: float,
        has_translation: bool,
        dy: float, dx: float, peak_val: float,
        changed_mask: np.ndarray,
        cell_diff: np.ndarray,
    ) -> ActivityResult:
        # Nothing changed.
        if changed_ratio < 0.05:
            return ActivityResult(
                activity=ScreenActivity.STATIC,
                confidence=0.95,
                changed_ratio=changed_ratio,
                chrome_stable=chrome_stable,
            )

        if changed_ratio < 0.15 and focus > 0.6:
            changed_rows = changed_mask.any(axis=1)
            bottom_half_active = changed_rows[len(changed_rows) // 2:].sum()
            if bottom_half_active >= 1 and changed_ratio < 0.10:
                return ActivityResult(
                    activity=ScreenActivity.TYPING,
                    confidence=0.6,
                    changed_ratio=changed_ratio,
                    chrome_stable=chrome_stable,
                    content_region_focus=focus,
                )
            return ActivityResult(
                activity=ScreenActivity.MICRO,
                confidence=0.7,
                changed_ratio=changed_ratio,
                chrome_stable=chrome_stable,
                content_region_focus=focus,
            )

        # Translation detected + chrome stable → SCROLL.
        if has_translation and chrome_stable and changed_ratio < 0.85:
            direction = ""
            if abs(dy) > abs(dx):
                direction = "down" if dy > 0 else "up"
            else:
                direction = "right" if dx > 0 else "left"
            speed = (abs(dy) ** 2 + abs(dx) ** 2) ** 0.5
            return ActivityResult(
                activity=ScreenActivity.SCROLL,
                confidence=min(0.95, peak_val * 2),
                scroll_direction=direction,
                scroll_speed=speed,
                changed_ratio=changed_ratio,
                chrome_stable=chrome_stable,
            )

        # Nearly everything changed → APP_SWITCH.
        if changed_ratio > 0.80 and not chrome_stable:
            return ActivityResult(
                activity=ScreenActivity.APP_SWITCH,
                confidence=0.85,
                changed_ratio=changed_ratio,
                chrome_stable=False,
            )

        # Content changed but chrome stable → NAVIGATION.
        if content_changed_ratio > 0.45 and chrome_stable:
            return ActivityResult(
                activity=ScreenActivity.NAVIGATION,
                confidence=0.75,
                changed_ratio=changed_ratio,
                chrome_stable=True,
                content_region_focus=focus,
            )

        # Moderate widespread changes + not translating → MEDIA_PLAYING.
        if 0.15 <= changed_ratio <= 0.70 and not has_translation:
            if focus < 0.5 or self._consecutive_media >= 2:
                return ActivityResult(
                    activity=ScreenActivity.MEDIA_PLAYING,
                    confidence=0.6 + 0.1 * min(3, self._consecutive_media),
                    changed_ratio=changed_ratio,
                    chrome_stable=chrome_stable,
                    content_region_focus=focus,
                )

        # Chrome changed + moderate content change → APP_SWITCH (softer).
        if not chrome_stable and changed_ratio > 0.40:
            return ActivityResult(
                activity=ScreenActivity.APP_SWITCH,
                confidence=0.65,
                changed_ratio=changed_ratio,
                chrome_stable=False,
            )

        # Moderate content change, chrome stable → NAVIGATION (fallback).
        if chrome_stable and content_changed_ratio > 0.25:
            return ActivityResult(
                activity=ScreenActivity.NAVIGATION,
                confidence=0.55,
                changed_ratio=changed_ratio,
                chrome_stable=True,
                content_region_focus=focus,
            )

        # Fallback.
        if has_translation:
            direction = "down" if dy > 0 else "up"
            return ActivityResult(
                activity=ScreenActivity.SCROLL,
                confidence=0.4,
                scroll_direction=direction,
                scroll_speed=(abs(dy) ** 2 + abs(dx) ** 2) ** 0.5,
                changed_ratio=changed_ratio,
                chrome_stable=chrome_stable,
            )

        return ActivityResult(
            activity=ScreenActivity.MICRO,
            confidence=0.4,
            changed_ratio=changed_ratio,
            chrome_stable=chrome_stable,
            content_region_focus=focus,
        )
