"""Prove synthetic input reaches Minecraft Bedrock — the foundation for the agent.

Run it, then ALT-TAB into Minecraft and put the cursor in the game (so it's captured).
Wallie will run a short scripted demo: look around, walk, jump, mine. Watch whether
the character actually moves.

  python scripts/test_minecraft_control.py

PANIC: press F8 any time to abort and release everything.
Tip: if nothing moves in fullscreen, try Minecraft in *windowed* mode and re-run.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from control import InputController


def countdown(sec: int) -> None:
    print(f"\nFocus Minecraft NOW (click into the game). Starting in {sec}s...", flush=True)
    for i in range(sec, 0, -1):
        print(f"  {i}...", flush=True)
        time.sleep(1)


def step(ic: InputController, label: str) -> bool:
    """Print a step label; return False if aborted."""
    if ic.abort_requested():
        print("ABORT (F8).", flush=True)
        return False
    print(f"  > {label}", flush=True)
    return True


def main() -> None:
    ic = InputController()
    print("=== Wallie input test === (F8 to abort)")
    countdown(5)
    try:
        if not step(ic, "look RIGHT"): return
        ic.smooth_look(260, 0)
        time.sleep(0.4)
        if not step(ic, "look LEFT"): return
        ic.smooth_look(-260, 0)
        time.sleep(0.4)
        if not step(ic, "look UP then DOWN"): return
        ic.smooth_look(0, -160); time.sleep(0.25); ic.smooth_look(0, 160)
        time.sleep(0.4)

        if not step(ic, "walk FORWARD (W) 1.5s"): return
        ic.key_down("w"); time.sleep(1.5); ic.key_up("w")
        time.sleep(0.3)

        if not step(ic, "turn right ~90 then JUMP"): return
        ic.smooth_look(420, 0, steps=30)
        ic.tap("space", 0.06)
        time.sleep(0.4)

        if not step(ic, "MINE (hold left-click 2.5s + natural look)"): return
        ic.mouse_down("left")
        t_end = time.time() + 2.5
        while time.time() < t_end:
            if ic.abort_requested():
                break
            ic.look_jitter(2)
            time.sleep(0.05)
        ic.mouse_up("left")

        print("\nDONE. Did the character look / walk / jump / mine?", flush=True)
    finally:
        ic.release_all()
        print("(all keys/buttons released)", flush=True)


if __name__ == "__main__":
    main()
