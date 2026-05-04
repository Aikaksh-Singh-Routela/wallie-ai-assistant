"""Screen activity classifier — detects *what the user is doing* from frame diffs.

The core insight: pHash tells us *how much* changed, but not *how* it changed.
A scroll, a tab switch, and a video frame all produce similar Hamming distances
but require completely different AI responses.

Algorithm (cheap, no OpenCV required):
  1. Divide each frame into a grid of cells (e.g. 6 rows x 4 cols).
  2. Hash each cell independently (pHash or average hash).
  3. Compare cell hashes between consecutive frames to build a "change map".
  4. Pattern-match the change map to classify the activity:
     - SCROLL: vertical band of cells changed, edges (chrome) stable
     - NAVIGATION: content cells changed, chrome partially stable
     - APP_SWITCH: nearly all cells changed including chrome
     - MEDIA_PLAYING: continuous moderate changes concentrated in one region
     - TYPING: 1-2 cells changed in a text-area-shaped region
     - MICRO: only 1 cell changed very slightly

  5. Phase correlation (numpy FFT) detects translation → confirms scroll
     direction and speed.

All operations run on already-downscaled JPEG frames (768px max edge),
so the cost is negligible compared to a single LLM call.
"""
from __future__ import annotations

import io
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional, Tuple

import numpy as np
from PIL import Image

from .capture import Frame
from .scene_classifier import ScreenActivity

# Grid dimensions for region-based comparison.
GRID_ROWS = 6
GRID_COLS = 4
# How many recent activity classifications to keep for pattern detection.
HISTORY_LEN = 12
# Phase correlation peak threshold — above this, we're confident about translation.
PHASE_CORR_THRESHOLD = 0.15


@dataclass
class ActivityResult:
    """Output of a single classify() call."""
    activity: ScreenActivity
    confidence: float           # 0..1 — how certain we are about this classification
    scroll_direction: str = ""  # "up" | "down" | "left" | "right" | ""
    scroll_speed: float = 0.0   # estimated pixels of translation (on the downscaled frame)
    changed_ratio: float = 0.0  # fraction of grid cells that changed
    chrome_stable: bool = True  # whether the app chrome (top/bottom bars) is stable
    content_region_focus: float = 0.0  # how concentrated changes are in one area


def _frame_to_gray(frame: Frame) -> np.ndarray:
    """Convert a JPEG Frame to a grayscale numpy array."""
    img = Image.open(io.BytesIO(frame.jpeg)).convert("L")
    return np.array(img, dtype=np.float32)


def _cell_hashes(gray: np.ndarray, rows: int = GRID_ROWS, cols: int = GRID_COLS) -> np.ndarray:
    """Compute a cheap average-intensity fingerprint per grid cell.

    Returns a (rows, cols) array of mean pixel values. Comparing these between
    frames is much faster than per-cell pHash and good enough for activity
    classification — we don't need perceptual quality, just "did this region
    change significantly?".
    """
    h, w = gray.shape
    cell_h, cell_w = h // rows, w // cols
    result = np.zeros((rows, cols), dtype=np.float32)
    for r in range(rows):
        for c in range(cols):
            y0, y1 = r * cell_h, (r + 1) * cell_h
            x0, x1 = c * cell_w, (c + 1) * cell_w
            result[r, c] = gray[y0:y1, x0:x1].mean()
    return result


def _cell_stds(gray: np.ndarray, rows: int = GRID_ROWS, cols: int = GRID_COLS) -> np.ndarray:
    """Compute per-cell standard deviation — helps distinguish typing from noise."""
    h, w = gray.shape
    cell_h, cell_w = h // rows, w // cols
    result = np.zeros((rows, cols), dtype=np.float32)
    for r in range(rows):
        for c in range(cols):
            y0, y1 = r * cell_h, (r + 1) * cell_h
            x0, x1 = c * cell_w, (c + 1) * cell_w
            result[r, c] = gray[y0:y1, x0:x1].std()
    return result


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
    """Classifies user screen activity from consecutive frame comparisons.

    Call :meth:`classify` with each new frame. The classifier maintains state
    (previous frame data, activity history) to detect patterns over time.
    """

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
        """Classify the screen activity for a new frame.

        Must be called for every captured frame (including ones that didn't
        pass the change detector threshold) to maintain accurate state.
        """
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

        # --- Phase correlation for scroll detection ---
        dy, dx, peak_val = _phase_correlate(self._prev_gray, gray)
        has_translation = peak_val > PHASE_CORR_THRESHOLD and (abs(dy) > 2 or abs(dx) > 2)

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
        """Summarize the recent activity pattern for the AI.

        Returns a short label like "browsing", "settled", "watching", "rapid_switching".
        """
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
        """True if the user has been on the same content for a while."""
        if len(self._history) < 4:
            return False
        recent = list(self._history)[-4:]
        return all(a in (ScreenActivity.STATIC, ScreenActivity.MICRO) for a in recent)

    def is_rapid_browsing(self) -> bool:
        """True if the user is quickly flipping through content."""
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

    # ------------------------------------------------------------------
    # Internal decision logic
    # ------------------------------------------------------------------
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
        # 1) Nothing changed.
        if changed_ratio < 0.05:
            return ActivityResult(
                activity=ScreenActivity.STATIC,
                confidence=0.95,
                changed_ratio=changed_ratio,
                chrome_stable=chrome_stable,
            )

        # 2) Very small, focused change → MICRO or TYPING.
        if changed_ratio < 0.15 and focus > 0.6:
            # Typing heuristic: changes concentrated in middle-bottom rows
            # (where text input areas typically are).
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

        # 3) Translation detected + chrome stable → SCROLL.
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

        # 4) Nearly everything changed → APP_SWITCH.
        if changed_ratio > 0.80 and not chrome_stable:
            return ActivityResult(
                activity=ScreenActivity.APP_SWITCH,
                confidence=0.85,
                changed_ratio=changed_ratio,
                chrome_stable=False,
            )

        # 5) Content changed but chrome stable → NAVIGATION.
        if content_changed_ratio > 0.45 and chrome_stable:
            return ActivityResult(
                activity=ScreenActivity.NAVIGATION,
                confidence=0.75,
                changed_ratio=changed_ratio,
                chrome_stable=True,
                content_region_focus=focus,
            )

        # 6) Moderate widespread changes + not translating → MEDIA_PLAYING.
        if 0.15 <= changed_ratio <= 0.70 and not has_translation:
            # Media heuristic: changes are spread across a region (not focused
            # like typing) but not everywhere (not app switch).
            if focus < 0.5 or self._consecutive_media >= 2:
                return ActivityResult(
                    activity=ScreenActivity.MEDIA_PLAYING,
                    confidence=0.6 + 0.1 * min(3, self._consecutive_media),
                    changed_ratio=changed_ratio,
                    chrome_stable=chrome_stable,
                    content_region_focus=focus,
                )

        # 7) Chrome changed + moderate content change → APP_SWITCH (softer).
        if not chrome_stable and changed_ratio > 0.40:
            return ActivityResult(
                activity=ScreenActivity.APP_SWITCH,
                confidence=0.65,
                changed_ratio=changed_ratio,
                chrome_stable=False,
            )

        # 8) Moderate content change, chrome stable, no clear translation →
        #    NAVIGATION (fallback).
        if chrome_stable and content_changed_ratio > 0.25:
            return ActivityResult(
                activity=ScreenActivity.NAVIGATION,
                confidence=0.55,
                changed_ratio=changed_ratio,
                chrome_stable=True,
                content_region_focus=focus,
            )

        # 9) Fallback: some change happened but can't classify clearly.
        #    Default to SCROLL if there's any translation signal, else MICRO.
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
