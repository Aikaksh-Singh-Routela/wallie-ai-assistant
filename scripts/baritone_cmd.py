"""Send ONE Baritone command into Minecraft chat — manual test, no AI.

Use this FIRST to confirm Baritone itself works before running the AI agent:

  python scripts/baritone_cmd.py "#mine oak_log"
  python scripts/baritone_cmd.py "#goto 100 70 100"
  python scripts/baritone_cmd.py "#stop"

It focuses Minecraft, counts down 4s, then pastes the command into chat (T) and
presses Enter. Pasting via clipboard means '#' is correct on ANY keyboard layout.
If Baritone walks/mines after this → Baritone is installed correctly.
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

from control.input_controller import InputController
from control.baritone_agent import _set_clipboard


def send_chat(ic: InputController, text: str) -> None:
    ic.tap("t", 0.05)          # open chat
    time.sleep(0.35)
    if not _set_clipboard(text):
        print("clipboard failed — aborting")
        return
    ic.key_down("ctrl")
    ic.tap("v", 0.04)
    ic.key_up("ctrl")
    time.sleep(0.15)
    ic.tap("enter", 0.05)


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "#mine oak_log"
    print(f"\nWill send to Minecraft chat:  {cmd}")
    print("Focus Minecraft! Sending in 4s...  (Baritone prefix is '#')")

    try:
        import pygetwindow as gw
        wins = [w for w in gw.getAllWindows() if "Minecraft" in w.title]
        if wins:
            wins[0].activate()
            time.sleep(0.4)
            print(f"Auto-focused: {wins[0].title}")
    except Exception:
        print("Could not auto-focus — click the Minecraft window yourself.")

    for i in range(4, 0, -1):
        print(f"  {i}...", flush=True)
        time.sleep(1)

    ic = InputController()
    send_chat(ic, cmd)
    print("Sent. Watch the game — Baritone should start. (#stop to cancel.)")


if __name__ == "__main__":
    main()
