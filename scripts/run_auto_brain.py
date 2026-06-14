"""Run Wallie as a fully autonomous Minecraft run: wood -> ... -> Ender Dragon.

  python scripts/run_auto_brain.py                # start from the beginning (wood)
  python scripts/run_auto_brain.py 5              # resume from phase index 5 (0-based)
  python scripts/run_auto_brain.py 0 gremlin      # phase 0 with a given profile

Launch Minecraft (Fabric 1.21.11, with the smoothcam + baritone + meteor mods), join a
survival world, then run this. Focus Minecraft during the 5s countdown. F8 = STOP.
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import config
from config import Secrets
from llm import build_provider
from vision.capture import ScreenCapture
from control.auto_brain import AutoBrain, TECH_TREE, load_progress

STATE_FILE = str(Path(__file__).resolve().parent.parent / "wallie_progress.json")


def main() -> None:
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    profile = sys.argv[2] if len(sys.argv) > 2 else "gremlin"
    if arg == "reset":
        try:
            os.remove(STATE_FILE)
        except OSError:
            pass
        print("progress reset — starting from phase 0"); arg = None
    if arg is not None:
        start = int(arg)                      # explicit phase index
    else:
        start = load_progress(STATE_FILE)     # resume where we left off
    start = max(0, min(start, len(TECH_TREE) - 1))

    cfg = config.load_profile(profile)
    if not cfg.llm.vision_capable:
        print(f"WARNING: profile '{profile}' LLM is not vision_capable.")
    provider = build_provider(cfg.llm, Secrets())
    cap = ScreenCapture(monitor_index=cfg.vision.monitor_index, max_edge_px=512, jpeg_quality=60)
    brain = AutoBrain(cap, provider, start_phase=start, state_file=STATE_FILE)

    print(f"\nFULL AUTONOMOUS RUN — {len(TECH_TREE)} phases, starting at #{start} ({TECH_TREE[start].name})")
    print(f"Model: {cfg.llm.provider}/{cfg.llm.model}")
    print("Phases:", " -> ".join(p.name for p in TECH_TREE))
    print("Starts in 5s. Focus Minecraft!  F8 = STOP.")
    for i in range(5, 0, -1):
        print(f"  {i}...", flush=True)
        time.sleep(1)
    try:
        import pygetwindow as gw
        wins = [w for w in gw.getAllWindows() if "Minecraft" in w.title]
        if wins:
            wins[0].activate(); time.sleep(0.4)
            print(f"Focused: {wins[0].title}")
    except Exception:
        print("Click Minecraft yourself!")

    async def _run():
        try:
            await brain.run()
        finally:
            await provider.aclose()
    try:
        asyncio.run(_run())
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\nstopped.")


if __name__ == "__main__":
    main()
