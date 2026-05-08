"""Periodic vision loop — grabs frames, runs change detection, emits VisionEvents."""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

from config import VisionConfig

from .capture import Frame, ScreenCapture
from .change_detector import FrameChangeDetector
from .scene_classifier import ChangeType, ScreenActivity
from .activity_classifier import ActivityResult


@dataclass
class VisionEvent:
    frame: Frame
    changed: bool
    captured_at: float = field(default_factory=time.time)
    change_type: ChangeType = ChangeType.SCENE_CHANGE
    activity: ScreenActivity = ScreenActivity.STATIC
    activity_detail: Optional[ActivityResult] = None
    user_pattern: str = ""


class VisionLoop:
    def __init__(self, cfg: VisionConfig, out_queue: "asyncio.Queue[VisionEvent]") -> None:
        self._cfg = cfg
        self._queue = out_queue
        self._capture = ScreenCapture(
            monitor_index=cfg.monitor_index,
            max_edge_px=cfg.max_edge_px,
        )
        self._detector = FrameChangeDetector(
            threshold=cfg.min_change_threshold,
            scene_change_threshold=cfg.scene_change_threshold,
            idle_variance_threshold=cfg.idle_variance_threshold,
            micro_change_threshold=getattr(cfg, "micro_change_threshold", 4),
        )
        self._task: Optional[asyncio.Task] = None
        self._last_emit_ts: float = 0.0

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name="vision-loop")
            logger.info(
                "vision: loop started, interval={}s, threshold={}, "
                "scene_threshold={}, min_emit={}s, monitor={}".format(
                    self._cfg.interval_sec,
                    self._cfg.min_change_threshold,
                    self._cfg.scene_change_threshold,
                    self._cfg.min_emit_interval_sec,
                    self._cfg.monitor_index,
                )
            )

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._capture.close()

    async def grab_now(self) -> Optional[VisionEvent]:
        loop = asyncio.get_event_loop()
        try:
            frame = await loop.run_in_executor(None, self._capture.grab)
            return VisionEvent(
                frame=frame,
                changed=False,
                captured_at=time.time(),
                change_type=ChangeType.DELTA,
            )
        except Exception as e:
            logger.warning("vision: on-demand grab failed: {}".format(e))
            return None

    # --- internal ---

    async def _run(self) -> None:
        loop = asyncio.get_event_loop()
        try:
            while True:
                captured_at = time.time()
                frame = await loop.run_in_executor(None, self._capture.grab)
                change_type, _ = await loop.run_in_executor(
                    None, self._detector.classify, frame
                )

                activity_result = self._detector.last_activity
                user_pattern = self._detector.activity_pattern

                if change_type not in (ChangeType.NONE, ChangeType.IDLE):
                    now = time.time()
                    if now - self._last_emit_ts >= self._cfg.min_emit_interval_sec:
                        logger.info(
                            "vision: {} detected (activity={}), emitting frame {}x{} ({} bytes)".format(
                                change_type.value,
                                activity_result.activity.value,
                                frame.width,
                                frame.height,
                                len(frame.jpeg),
                            )
                        )
                        event = VisionEvent(
                            frame=frame,
                            changed=True,
                            captured_at=captured_at,
                            change_type=change_type,
                            activity=activity_result.activity,
                            activity_detail=activity_result,
                            user_pattern=user_pattern,
                        )
                        self._enqueue(event)
                        self._last_emit_ts = now
                    else:
                        logger.debug(
                            "vision: {} suppressed (emit interval not elapsed)".format(
                                change_type.value
                            )
                        )
                elif change_type == ChangeType.IDLE:
                    logger.debug("vision: idle frame skipped")

                await asyncio.sleep(self._adaptive_interval(change_type))
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("vision: loop crashed: {}".format(e))

    def _adaptive_interval(self, change_type: ChangeType) -> float:
        base = self._cfg.interval_sec
        if change_type == ChangeType.SCENE_CHANGE:
            return base * 0.4
        if change_type == ChangeType.DELTA:
            return base * 0.7
        if change_type == ChangeType.IDLE:
            return base * 2.0
        # NONE — nothing happening, sample slower.
        return base * 1.3

    def _enqueue(self, event: VisionEvent) -> None:
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                self._queue.put_nowait(event)
            except asyncio.QueueFull:
                logger.debug("vision: queue still full after eviction, dropping frame")
