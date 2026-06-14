"""Run Wallie as a Baritone-integrated Minecraft agent.

This requires you to be running Minecraft Java Edition with the Baritone mod installed.
Wallie will watch your screen and type Baritone commands into chat to play the game smoothly!

  python scripts/run_baritone_agent.py
  python scripts/run_baritone_agent.py "Mine 3 diamonds and build a house"

Focus Minecraft during the 5s countdown. F8 aborts and releases all inputs.
"""
from __future__ import annotations

import asyncio
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
from control.baritone_agent import BaritoneAgent


async def _run(agent: BaritoneAgent, provider) -> None:
    try:
        await agent.run()
    finally:
        await provider.aclose()


def main() -> None:
    goal = sys.argv[1] if len(sys.argv) > 1 else \
        "Collect wood with '#mine oak_log', then mine stone and craft basic tools."
    profile = sys.argv[2] if len(sys.argv) > 2 else "gremlin"

    cfg = config.load_profile(profile)
    if not cfg.llm.vision_capable:
        print(f"WARNING: profile '{profile}' LLM is not vision_capable — agent needs vision.")
    provider = build_provider(cfg.llm, Secrets())
    
    # Capture the screen to see what Baritone is doing
    cap = ScreenCapture(monitor_index=cfg.vision.monitor_index, max_edge_px=512, jpeg_quality=60)
    agent = BaritoneAgent(cap, provider, goal=goal)

    print(f"\nGOAL: {goal}\nModel: {cfg.llm.provider}/{cfg.llm.model} (monitor {cfg.vision.monitor_index})")
    print("IMPORTANT: Make sure Baritone is installed in your Minecraft Client!")
    print("Agent starts in 5s. Attempting to auto-focus Minecraft...  F8 = STOP.")
    for i in range(5, 0, -1):
        print(f"  {i}...", flush=True)
        time.sleep(1)

    try:
        import pygetwindow as gw
        mc_windows = [w for w in gw.getAllWindows() if "Minecraft" in w.title or "Lunar Client" in w.title]
        if mc_windows:
            mc_windows[0].activate()
            time.sleep(0.5)
            print(f"Auto-focused: {mc_windows[0].title}")
    except Exception as e:
        print("Could not auto-focus. Please click Minecraft manually!")

    try:
        asyncio.run(_run(agent, provider))
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\nstopped.")


if __name__ == "__main__":
    main()
