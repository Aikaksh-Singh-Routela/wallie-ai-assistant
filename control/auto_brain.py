"""Autonomous progression brain: plays a full Minecraft run as a tech-tree of phases,
from gathering wood toward the Ender Dragon.

Design: each PHASE has Baritone command(s) to drive it and a completion CUE the VLM
checks on screen. The brain issues the command, lets Baritone work, and every few
seconds asks the VLM "is this objective done? if stuck, what command?" — then advances.

Reliable today: the MINING / TRAVEL / EXPLORE / DIG phases (pure Baritone).
Scaffolded (need crafting/combat automation): the CRAFT / SMELT / NETHER / END phases —
the VLM attempts them and the brain advances on a timeout so the run never hard-locks.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass, field

from loguru import logger


class _BaritoneLog:
    """Tails .minecraft/logs/latest.log to know when Baritone is working vs idle.
    A mining/dig phase is 'done' when Baritone has been active and then goes quiet
    (e.g. it reached '#mine 20', printed 'have 21 items' and stopped)."""
    def __init__(self) -> None:
        self.path = os.path.join(os.environ.get("APPDATA", ""), ".minecraft", "logs", "latest.log")
        self._pos = 0
        try:
            self._pos = os.path.getsize(self.path)
        except OSError:
            pass

    def new_baritone_lines(self) -> int:
        try:
            with open(self.path, "r", encoding="utf-8", errors="ignore") as f:
                f.seek(self._pos)
                data = f.read()
                self._pos = f.tell()
        except OSError:
            return 0
        return sum(1 for ln in data.splitlines() if "Baritone" in ln)

from .baritone_agent import _set_clipboard, _LOG_TYPES
from .input_controller import InputController


@dataclass
class Phase:
    name: str
    desc: str                       # human objective, shown to the VLM
    commands: list[str] = field(default_factory=list)   # Baritone cmds to (re)issue on entry
    cue: str = ""                   # what the VLM should see/own to mark this done
    max_sec: float = 240.0          # safety: advance anyway after this long
    craft: bool = False             # needs inventory/crafting (not pure Baritone)
    craft_cmds: list[str] = field(default_factory=list)  # /wcraft + /wtable steps (auto, no F9)


# The run, start to finish. Commands use Baritone; craft phases are VLM/assisted.
TECH_TREE: list[Phase] = [
    Phase("wood", "Gather logs from nearby trees (enough for tools AND a wooden shelter)",
          [f"#mine 32 {_LOG_TYPES}"], "a couple stacks of logs", 260),
    Phase("wood_tools", "Craft LOTS of planks (for the shelter), sticks, table, wooden pickaxe + axe",
          [], "a wooden pickaxe + plenty of planks", 150, craft=True,
          craft_cmds=["/wcraft planks 24", "/wcraft sticks 2", "/wcraft crafting_table",
                      "/wtable", "/wcraft wooden_pickaxe", "/wcraft wooden_axe"]),
    # WOODEN SHELTER FIRST — built from the planks above, no mining needed, for night safety.
    Phase("shelter", "Build a small WOODEN shelter to survive the first night",
          ["#build wallie_house.schem ~ ~ ~"], "a wooden shelter built around you", 400),
    Phase("cobble", "Mine cobblestone for stone tools",
          ["#mine 48 cobblestone stone"], "about 40+ cobblestone", 300),
    Phase("stone_tools", "Craft a stone pickaxe, sword, axe and a furnace",
          [], "a stone pickaxe and furnace", 150, craft=True,
          craft_cmds=["/wtable", "/wcraft sticks 2", "/wcraft stone_pickaxe", "/wcraft stone_sword",
                      "/wcraft stone_axe", "/wcraft furnace"]),
    Phase("coal", "Mine coal ore and make torches",
          ["#mine 16 coal_ore"], "coal and some torches", 200),
    Phase("iron", "Mine iron ore",
          ["#mine 24 iron_ore"], "around 24 raw iron", 320),
    Phase("iron_gear", "Smelt iron; craft iron pickaxe, sword, full armor and a shield",
          [], "iron armor worn and an iron pickaxe", 300, craft=True),
    Phase("food", "Hunt animals or find food so hunger is safe",
          [], "cooked food in inventory", 180, craft=True),
    Phase("diamond", "Dig down to y -59 and mine diamonds",
          ["#mine 12 diamond_ore"], "at least 3 diamonds", 600),
    Phase("diamond_gear", "Craft a diamond pickaxe and sword",
          [], "a diamond pickaxe", 180, craft=True,
          craft_cmds=["/wtable", "/wcraft sticks 1", "/wcraft diamond_pickaxe", "/wcraft diamond_sword"]),
    Phase("obsidian", "Mine obsidian for a nether portal (need water+lava or a diamond pick)",
          ["#mine 10 obsidian"], "about 10 obsidian", 400),
    Phase("nether", "Build a nether portal, light it and enter the Nether",
          [], "the red Nether sky / you are in the Nether", 300, craft=True),
    Phase("blaze", "Find a nether fortress and get blaze rods",
          ["#mine 6 blaze_rod"], "blaze rods (-> blaze powder)", 600),
    Phase("pearls", "Trade with piglins or kill endermen for ender pearls",
          [], "at least 12 ender pearls", 600, craft=True),
    Phase("eyes", "Craft eyes of ender from blaze powder + pearls",
          [], "12+ eyes of ender", 120, craft=True),
    Phase("stronghold", "Throw eyes of ender and travel to the stronghold",
          [], "the stronghold / end portal frame", 600),
    Phase("end", "Fill the end portal frame and jump in",
          [], "the black void of the End dimension", 150, craft=True),
    Phase("dragon", "Destroy the end crystals, then kill the Ender Dragon",
          ["#mine 10 end_crystal"], "the Ender Dragon is dead / the egg is spawned", 900, craft=True),
]


_SYSTEM = (
    "You are autonomously playing a full Minecraft survival run, start to finish, toward "
    "killing the Ender Dragon. You have the Baritone mod (it pathfinds/mines/digs for you via "
    "chat commands like '#mine', '#goto', '#explore', '#stop'). You see the screen each step.\n"
    "You are given your CURRENT OBJECTIVE. Look at the screen and your hotbar/inventory and "
    "decide ONE of:\n"
    "- done: true  -> the objective is clearly complete (you have the item / are in the place).\n"
    "- a 'command' -> a Baritone chat command to make progress (e.g. '#mine 16 iron_ore', "
    "'#goto 100 12 100', '#explore', '#stop'). Use this if Baritone seems idle or off-track.\n"
    "- nothing (done:false, no command) -> Baritone is actively working; just wait.\n"
    "Reply ONLY strict JSON: {\"done\":false,\"command\":\"\",\"note\":\"<short>\"}"
)


def _sanitize(cmd: str) -> str:
    """Fix the most common bad block names the VLM emits ('log'/'logs'/'wood' aren't real
    1.21 blocks -> expand to the actual log types)."""
    cmd = re.sub(r"\b(logs?|wood)\b", _LOG_TYPES, cmd)
    return cmd


def _parse(text: str) -> dict:
    out = {"done": False, "command": "", "note": ""}
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            d = json.loads(m.group(0))
            out["done"] = bool(d.get("done", False))
            out["command"] = str(d.get("command", "")).strip()
            out["note"] = str(d.get("note", ""))[:160]
        except Exception:
            pass
    return out


def load_progress(path: str) -> int:
    try:
        with open(path) as f:
            return int(json.load(f).get("phase", 0))
    except (OSError, ValueError, json.JSONDecodeError):
        return 0


class AutoBrain:
    def __init__(self, capture, provider, *, start_phase: int = 0, state_file: str | None = None) -> None:
        self.ic = InputController()
        self.capture = capture
        self.provider = provider
        self.idx = start_phase
        self.state_file = state_file
        self._running = False
        self._chatting = False
        self._blog = _BaritoneLog()

    def _advance(self) -> None:
        """Move to the next phase and persist progress so reruns resume here."""
        self.idx += 1
        if self.state_file:
            try:
                with open(self.state_file, "w") as f:
                    json.dump({"phase": self.idx}, f)
            except OSError:
                pass

    # --- camera naturalness is handled by the smoothcam mod + Baritone freeLook ---
    def _chat(self, text: str) -> None:
        self._chatting = True
        try:
            self.ic.tap("t", 0.05)
            time.sleep(0.35)
            if _set_clipboard(text):
                self.ic.key_down("ctrl"); self.ic.tap("v", 0.04); self.ic.key_up("ctrl")
            time.sleep(0.15)
            self.ic.tap("enter", 0.05)
            time.sleep(0.2)
        finally:
            self._chatting = False

    def _setup(self) -> None:
        for c in ["#set freeLook false", "#set smoothLook false", "#set renderPath false",
                  "#set renderGoal false", "#set renderGoalXZBeacon false",
                  "#set echoCommands false", "#set chatDebug false"]:
            if InputController.abort_requested():
                return
            self._chat(c); time.sleep(0.1)

    async def _ask(self, phase: Phase) -> dict:
        frame = self.capture.grab()
        user = (f"CURRENT OBJECTIVE ({self.idx+1}/{len(TECH_TREE)}): {phase.desc}\n"
                f"DONE WHEN: {phase.cue}\n"
                "Look at the screen + inventory. Reply ONLY JSON.")
        msgs = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": [
                {"type": "text", "text": user},
                {"type": "image", "data": frame.jpeg, "mime": "image/jpeg"},
            ]},
        ]
        text = ""
        for _ in range(2):
            text = ""
            async for ch in self.provider.stream(msgs, temperature=0.2, top_p=0.9, max_tokens=160):
                text += ch
            if "{" in text:
                break
        return _parse(text)

    async def _idle(self, seconds: float) -> str:
        """Wait, but poll F8 (abort) / F9 (advance) every ~0.12s so they feel responsive."""
        end = time.time() + seconds
        while time.time() < end:
            if InputController.abort_requested():
                return "abort"
            if InputController.continue_requested():
                return "continue"
            await asyncio.sleep(0.12)
        return "ok"

    async def run(self) -> None:
        self._running = True
        logger.info(f"AutoBrain: full run, starting at phase '{TECH_TREE[self.idx].name}'  (F8=stop)")
        self._setup()
        try:
            while self._running and self.idx < len(TECH_TREE):
                phase = TECH_TREE[self.idx]
                logger.info(f"=== PHASE {self.idx+1}/{len(TECH_TREE)}: {phase.name} — {phase.desc}")
                if phase.craft_cmds:
                    # AUTO-CRAFT via our mod: stand still, then run /wcraft + /wtable steps.
                    self._chat("#stop"); time.sleep(0.3)
                    logger.info(f"  🔧 auto-crafting ({len(phase.craft_cmds)} steps)")
                    for cc in phase.craft_cmds:
                        if InputController.abort_requested():
                            self._running = False; break
                        self._chat(cc)
                        time.sleep(2.2 if "wtable" in cc else 1.0)  # let the table menu open
                    self.ic.tap("esc", 0.05)      # CLOSE the crafting GUI so the next phase can move/mine
                    time.sleep(0.3)
                    sig = await self._idle(5.0)   # settle; stay responsive to F8/F9
                    if sig == "abort":
                        self._running = False
                    self._advance()
                    continue
                if phase.craft:
                    # the AI can't auto-do this one (smelt/portal/etc.) — hand off to the human.
                    self._chat("#stop")
                    logger.info(f"  🔨 YOUR TURN — {phase.desc}. Do it in-game, then HOLD F9 to continue.")
                else:
                    for c in phase.commands:
                        if InputController.abort_requested():
                            break
                        self._chat(c)
                t0 = last_kick = time.time()
                self._blog.new_baritone_lines()        # reset the tail position for this phase
                last_activity = time.time()
                seen_activity = False
                while self._running:
                    if InputController.abort_requested():
                        logger.warning("AutoBrain: F8 abort"); self._running = False; break
                    if InputController.continue_requested():
                        logger.info("  ▶ F9 — advancing to next phase"); break
                    # watch Baritone's log: active -> then quiet = task finished -> advance
                    if phase.commands and self._blog.new_baritone_lines() > 0:
                        seen_activity = True
                        last_activity = time.time()
                    if (phase.commands and seen_activity
                            and time.time() - last_activity > 12
                            and time.time() - t0 > 15):
                        logger.info(f"  -> '{phase.name}' done (Baritone went idle)"); break
                    try:
                        d = await self._ask(phase)
                        logger.info(f"  [{phase.name}] done={d['done']} {d['note']}")
                        if d["done"]:
                            logger.info(f"  -> '{phase.name}' looks complete"); break
                    except Exception as e:
                        logger.error(f"  brain error: {e}")
                    # re-kick the phase's OWN command if Baritone has likely gone idle/finished
                    if not phase.craft and phase.commands and time.time() - last_kick > 50:
                        logger.info(f"  re-issuing (stop+restart): {phase.commands[0]}")
                        self._chat("#stop")          # force Baritone to re-plan if it was stuck
                        for c in phase.commands:
                            self._chat(c)
                        last_kick = time.time()
                    if time.time() - t0 > phase.max_sec:
                        logger.warning(f"  -> '{phase.name}' timed out, advancing"); break
                    # idle between checks, but stay responsive to F8/F9
                    sig = await self._idle(5.0)
                    if sig == "abort":
                        logger.warning("AutoBrain: F8 abort"); self._running = False; break
                    if sig == "continue":
                        logger.info("  ▶ F9 — advancing to next phase"); break
                self._advance()
            if self.idx >= len(TECH_TREE):
                logger.info("AutoBrain: tech tree complete — GG.")
                if self.state_file:   # finished: reset so the next run starts fresh
                    try:
                        os.remove(self.state_file)
                    except OSError:
                        pass
        finally:
            self._running = False
            self.ic.release_all()
            logger.info("AutoBrain: stopped")
