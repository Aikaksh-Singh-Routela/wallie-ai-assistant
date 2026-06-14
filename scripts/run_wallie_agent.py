"""Run Wallie as an autonomous LangGraph agent (Observe->Plan->Act->Reflect).

It decides its own moves from the REAL game state (wallie_state.json from the mod) — no
fixed script. Baritone + the crafting mod execute; Meteor handles combat/eating.

  python scripts/run_wallie_agent.py
  python scripts/run_wallie_agent.py "Get full iron gear and a diamond pickaxe"
  python scripts/run_wallie_agent.py "Build a base and stock food" <profile>

Launch Minecraft (Fabric 1.21.11, smoothcam+baritone+meteor mods), join a survival world,
then run this. Focus Minecraft during the countdown. F8 = STOP.
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
from control.agent_graph import WallieAgent


def main() -> None:
    profile = sys.argv[2] if len(sys.argv) > 2 else None
    cfg = config.load_profile(profile)
    goal = sys.argv[1] if len(sys.argv) > 1 else cfg.play.goal
    provider = build_provider(cfg.llm, Secrets())
    cap = ScreenCapture(monitor_index=cfg.vision.monitor_index, max_edge_px=512, jpeg_quality=55)
    agent = WallieAgent(provider, goal=goal, capture=cap)

    print(f"\nWALLIE AGENT (LangGraph) — GOAL: {goal}")
    print(f"Model: {cfg.llm.provider}/{cfg.llm.model}")
    print("Decides its own actions from live game state. Starts in 5s. Focus Minecraft!  F8 = STOP.")
    for i in range(5, 0, -1):
        print(f"  {i}...", flush=True)
        time.sleep(1)
    try:
        import ctypes
        import pygetwindow as gw
        wins = [w for w in gw.getAllWindows() if "Minecraft" in w.title]
        if wins:
            w = wins[0]
            try:
                ctypes.windll.user32.ShowWindow(w._hWnd, 9)            # SW_RESTORE
                ctypes.windll.user32.SetForegroundWindow(w._hWnd)     # force foreground
            except Exception:
                w.activate()
            time.sleep(0.6)
            print(f"Focused: {w.title}  (if it didn't focus, CLICK the Minecraft window now!)")
        else:
            print(">>> No Minecraft window found — CLICK Minecraft yourself! <<<")
    except Exception:
        print(">>> CLICK the Minecraft window now! <<<")

    async def _run():
        try:
            await agent.run()
        finally:
            await provider.aclose()
    try:
        asyncio.run(_run())
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\nstopped.")


if __name__ == "__main__":
    main()
