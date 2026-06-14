"""Wallie LIVE — the active persona plays Minecraft and reacts in character.

Runs the streamer core (vision/persona/TTS) and the playing agent together; the agent feeds its
current intent to the core so the commentary stays grounded in what's actually happening.

  python scripts/run_wallie_live.py
  python scripts/run_wallie_live.py "Get full iron gear and a diamond pickaxe"
  python scripts/run_wallie_live.py "Build a base and stock food" <profile>

Launch Minecraft (Fabric 1.21.11, mods installed), join a survival world, focus the game during
the countdown. F8 = STOP.
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
from config import Runtime, Secrets, load_profile
from llm import build_provider
from vision.capture import ScreenCapture
from control.agent_graph import WallieAgent
from wallie import build_orchestrator


def _focus_minecraft() -> None:
    try:
        import ctypes
        import pygetwindow as gw
        wins = [w for w in gw.getAllWindows() if "Minecraft" in w.title]
        if wins:
            w = wins[0]
            try:
                ctypes.windll.user32.ShowWindow(w._hWnd, 9)
                ctypes.windll.user32.SetForegroundWindow(w._hWnd)
            except Exception:
                w.activate()
            time.sleep(0.6)
            print(f"Focused: {w.title}  (if it didn't focus, CLICK the Minecraft window now!)")
        else:
            print(">>> No Minecraft window found — CLICK Minecraft yourself! <<<")
    except Exception:
        print(">>> CLICK the Minecraft window now! <<<")


def main() -> None:
    profile = sys.argv[2] if len(sys.argv) > 2 else None   # None -> active profile (any persona)
    cfg = load_profile(profile)
    goal = sys.argv[1] if len(sys.argv) > 1 else cfg.play.goal

    runtime = Runtime(config=cfg, secrets=Secrets())

    orchestrator = build_orchestrator(runtime)
    agent_provider = build_provider(cfg.llm, Secrets())
    cap = ScreenCapture(monitor_index=cfg.vision.monitor_index, max_edge_px=512, jpeg_quality=55)
    agent = WallieAgent(agent_provider, goal=goal, capture=cap)

    print(f"\nWALLIE LIVE — {cfg.persona.name} PLAYS Minecraft (persona commentary + TTS)")
    print(f"GOAL: {goal}")
    print(f"Decision model: {cfg.llm.provider}/{cfg.llm.model}  |  Voice: {cfg.tts.provider}")
    print("Starts in 5s. Focus Minecraft!  F8 = STOP.")
    for i in range(5, 0, -1):
        print(f"  {i}...", flush=True)
        time.sleep(1)
    _focus_minecraft()

    async def _run() -> None:
        await orchestrator.start()
        try:
            await agent.run()
        finally:
            await orchestrator.stop()
            await agent_provider.aclose()

    try:
        asyncio.run(_run())
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\nstopped.")


if __name__ == "__main__":
    main()
