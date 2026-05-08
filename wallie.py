"""Entrypoint — builds all subsystems from config and starts the pipeline."""
from __future__ import annotations

import asyncio
import sys
from typing import Optional

from loguru import logger

from audio import AudioPlayer
from chat import ChatManager
from config import Runtime, get_runtime
from core import Orchestrator, Persona, MemoryStore
from config import PROFILES_DIR
from llm import build_provider
from tts import build_tts


def _configure_logger(level: str) -> None:
    logger.remove()
    logger.add(sys.stderr, level=level, enqueue=True, backtrace=False, diagnose=False)


def build_orchestrator(runtime: Optional[Runtime] = None) -> Orchestrator:
    runtime = runtime or get_runtime()
    cfg = runtime.config
    import os
    _configure_logger(os.getenv("LOG_LEVEL", "INFO"))

    persona = Persona.from_config(cfg.persona)
    llm = build_provider(cfg.llm, runtime.secrets)
    tts = build_tts(cfg.tts, runtime.secrets)
    player = AudioPlayer(sample_rate=tts.sample_rate, channels=tts.channels)

    chat_manager: Optional[ChatManager] = None
    
    if cfg.chat.youtube_enabled or cfg.chat.twitch_enabled or cfg.chat.kick_enabled:
        chat_manager = ChatManager(cfg.chat, runtime.secrets)

    vision_queue = None
    vision_loop = None
    if cfg.vision.enabled:
        if not cfg.llm.vision_capable:
            logger.warning("vision enabled but llm.vision_capable is False; disabling vision")
        else:
            try:
                from vision import VisionEvent, VisionLoop
            except ModuleNotFoundError as e:
                logger.error(
                    f"vision enabled but a dep is missing: {e}. "
                    "Install: pip install mss pillow imagehash"
                )
            else:
                vision_queue = asyncio.Queue(maxsize=4)
                vision_loop = VisionLoop(cfg.vision, vision_queue)

    avatar = None
    if cfg.avatar.enabled:
        try:
            from avatar import VTubeStudioAvatar
            avatar = VTubeStudioAvatar(cfg.avatar)
            asyncio.create_task(avatar.connect(), name="vts-avatar")
            logger.info(f"avatar: VTube Studio enabled ({cfg.avatar.vts_host}:{cfg.avatar.vts_port})")
        except Exception as e:
            logger.error(f"avatar: failed to start VTS client: {e}")

    profile_name = cfg.profile_name or "default"
    memory_path = PROFILES_DIR / f"{profile_name}.memory.json"
    memory_store = MemoryStore(memory_path)

    return Orchestrator(
        runtime=runtime,
        persona=persona,
        llm=llm,
        tts=tts,
        player=player,
        chat_manager=chat_manager,
        vision_loop=vision_loop,
        vision_queue=vision_queue,
        avatar=avatar,
        memory_store=memory_store,
    )


async def run_cli() -> None:
    orch = build_orchestrator()
    await orch.start()
    try:
        # Block forever; the orchestrator's main task handles everything.
        await asyncio.Event().wait()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        await orch.stop()


async def run_with_dashboard() -> None:
    from dashboard.server import serve

    orch = build_orchestrator()
    await serve(orch)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(prog="wallie", description="AI streamer runtime")
    ap.add_argument(
        "--dashboard",
        action="store_true",
        help="Start the web dashboard instead of the headless loop",
    )
    args = ap.parse_args()

    try:
        if args.dashboard:
            asyncio.run(run_with_dashboard())
        else:
            asyncio.run(run_cli())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
