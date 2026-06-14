"""Baritone-integrated vision agent.
Uses a vision LLM to see the screen and make high-level decisions, but outsources the
actual pathfinding and mining to Baritone via chat commands.
"""
from __future__ import annotations

import asyncio
import ctypes
import threading
import time
from loguru import logger

from .input_controller import InputController
from .memory import AgentMemory
import json
import re


def _set_clipboard(text: str) -> bool:
    """Put text on the Windows clipboard (unicode) via pure ctypes — no deps.
    Used so we can PASTE Baritone commands into chat instead of typing them,
    which dodges keyboard-layout issues (e.g. '#' is AltGr+3 on a Turkish layout)."""
    try:
        CF_UNICODETEXT = 13
        GMEM_MOVEABLE = 0x0002
        k32 = ctypes.windll.kernel32
        u32 = ctypes.windll.user32
        # 64-bit: declare pointer-returning funcs as void* so handles aren't truncated
        k32.GlobalAlloc.restype = ctypes.c_void_p
        k32.GlobalAlloc.argtypes = [ctypes.c_uint, ctypes.c_size_t]
        k32.GlobalLock.restype = ctypes.c_void_p
        k32.GlobalLock.argtypes = [ctypes.c_void_p]
        k32.GlobalUnlock.argtypes = [ctypes.c_void_p]
        u32.SetClipboardData.restype = ctypes.c_void_p
        u32.SetClipboardData.argtypes = [ctypes.c_uint, ctypes.c_void_p]
        data = text.encode("utf-16-le") + b"\x00\x00"
        h = k32.GlobalAlloc(GMEM_MOVEABLE, len(data))
        ptr = k32.GlobalLock(h)
        ctypes.memmove(ptr, data, len(data))
        k32.GlobalUnlock(h)
        if not u32.OpenClipboard(0):
            return False
        u32.EmptyClipboard()
        u32.SetClipboardData(CF_UNICODETEXT, h)
        u32.CloseClipboard()
        return True
    except Exception as e:
        logger.warning(f"clipboard set failed: {e}")
        return False

def _parse_baritone(text: str) -> dict:
    out = {"action": "wait", "text": "", "note": "", "subgoal": "", "reason": ""}
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            d = json.loads(m.group(0))
            a = str(d.get("action", "wait")).strip().lower()
            out["action"] = a
            out["text"] = str(d.get("text", ""))
            out["note"] = str(d.get("note", ""))[:220]
            out["subgoal"] = str(d.get("subgoal", ""))[:170]
            out["reason"] = str(d.get("reason", ""))[:170]
            return out
        except Exception:
            pass
    # fallback
    low = text.lower()
    if "chat" in low:
        out["action"] = "chat"
    elif "inventory" in low:
        out["action"] = "inventory"
    return out

_SYSTEM = (
    "You are a Minecraft Java Edition player. You have the 'Baritone' mod installed, "
    "which allows you to do complex tasks effortlessly by typing commands into the chat.\n"
    "You will receive screenshots of the game. Look at your health, inventory, and surroundings.\n"
    "You must decide the single next high-level action to take.\n\n"
    "HOW TO PLAY:\n"
    "1. You do not walk manually. You send Baritone chat commands and it pathfinds + mines.\n"
    "2. PICK WHAT'S NEARBY — look at the screenshot and target the CLOSEST resource you can "
    "actually SEE. Baritone always goes to the NEAREST matching block, so to avoid trekking "
    "across the map for one specific type, mine ANY nearby log at once:\n"
    "   #mine oak_log birch_log spruce_log jungle_log acacia_log dark_oak_log\n"
    "   For stone/ore just: '#mine stone' / '#mine iron_ore' / '#mine diamond_ore'.\n"
    "3. CAP IT so you don't wander forever: add a count, e.g. '#mine 16 oak_log birch_log ...'.\n"
    "4. If, after issuing #mine, Baritone starts walking FAR (you see it crossing open terrain "
    "with no tree close), send '#stop' and pick a closer resource that's visible on screen.\n"
    "5. If Baritone is actively working (moving/mining in the screenshot), choose 'wait'.\n"
    "6. To craft or check items, use 'inventory'. If stuck or Baritone stopped, issue a new command.\n\n"
    "Available Actions (JSON):\n"
    "- chat: Type a message or command into the game chat. (e.g. '#mine oak_log', '#stop')\n"
    "- wait: Do nothing. Use this while Baritone is executing your command.\n"
    "- inventory: Press 'E' to open/close inventory.\n"
    "- look_around: Look around to survey the area.\n"
    "- stop_baritone: Sends '#stop' to cancel the current Baritone task.\n"
    "Reply ONLY in STRICT JSON format:\n"
    '{"action":"chat", "text":"#mine 16 oak_log birch_log spruce_log", "subgoal":"Gather wood", "note":"birch right in front", "reason":"closest tree is birch"}'
)

_LOG_TYPES = "oak_log birch_log spruce_log jungle_log acacia_log dark_oak_log"


def _goal_to_command(goal: str) -> str | None:
    """A sensible Baritone command to KICK OFF the goal without waiting on the LLM,
    so the bot always starts doing something even if the free model returns empty."""
    g = goal.lower()
    if any(k in g for k in ("wood", "tree", "log", "plank", "axe")):
        return f"#mine 24 {_LOG_TYPES}"
    if "diamond" in g:
        return "#mine diamond_ore"
    if "iron" in g:
        return "#mine iron_ore"
    if any(k in g for k in ("stone", "cobble", "pickaxe")):
        return "#mine 32 stone"
    return None


def _build_user(goal: str, mem: AgentMemory) -> str:
    return (
        f"GOAL: {goal}\n"
        f"CURRENT PLAN: {mem.subgoal or '(decide one)'}\n"
        f"MEMORY NOTES: {mem.note or '(empty)'}\n"
        f"RECENT HISTORY: {mem.recent_text()}\n"
        "Look at the screen. Is Baritone currently doing something? Are you in a menu?\n"
        "Decide the single next action. Reply ONLY JSON."
    )

class BaritoneAgent:
    def __init__(self, capture, provider, *, goal: str, tick_hz: int = 30) -> None:
        self.ic = InputController()
        self.capture = capture
        self.provider = provider
        self.goal = goal
        self.mem = AgentMemory()
        self._running = False
        self._lock = threading.Lock()
        self._action = "wait"
        self._chat_text = ""
        self._chatting = False           # gate the organic camera while typing
        self._cam_on = False
        self._cam_thread: threading.Thread | None = None
        # OFF by default: with freeLook=false Baritone drives the camera toward the
        # travel/mining direction, so adding our own moves just fights it (jitter).
        self._organic_cam = False
        self._boot_cmd = _goal_to_command(goal)
        self._last_cmd_ts = 0.0          # when we last issued a real Baritone command
        self._idle_decisions = 0         # consecutive decisions with no command

    # Baritone settings that make it STREAM-CLEAN: no floating path lines, no goal
    # beacon, no command echo spam. freeLook=FALSE so Baritone turns the CLIENT camera
    # to face where it walks/mines — natural "look where you're going" — instead of
    # leaving the view adrift (which made the character walk forward while facing back).
    _STREAM_CLEAN = [
        # freeLook false → Baritone points the camera at what it walks to / mines; smoothcam
        # captures that target per tick and eases the displayed rotation per frame (zero snaps).
        "#set freeLook false",
        "#set smoothLook false",
        "#set renderPath false",
        "#set renderGoal false",
        "#set renderGoalAnimated false",
        "#set renderGoalXZBeacon false",
        "#set renderSelectionBoxes false",
        "#set renderCachedChunks false",
        "#set echoCommands false",
        "#set chatDebug false",
    ]

    def _setup_stream_clean(self) -> None:
        """Once, on start: hide Baritone's on-screen path/goal render + chat spam."""
        logger.info("baritone: applying stream-clean render settings")
        for cmd in self._STREAM_CLEAN:
            if InputController.abort_requested():
                return
            self._execute_chat(cmd)
            time.sleep(0.15)

    def _execute_chat(self, text: str) -> None:
        """Open chat (T), PASTE the command via clipboard (Ctrl+V), press Enter.
        Pasting avoids keyboard-layout problems with '#' on a Turkish layout."""
        logger.info(f"chat> {text}")
        self._chatting = True
        try:
            self.ic.tap("t", 0.05)          # open chat
            time.sleep(0.35)                 # let the text field focus
            if _set_clipboard(text):
                self.ic.key_down("ctrl")
                self.ic.tap("v", 0.04)
                self.ic.key_up("ctrl")
            else:
                logger.warning("clipboard failed — chat command not sent")
            time.sleep(0.15)
            self.ic.tap("enter", 0.05)       # submit
            time.sleep(0.2)
        finally:
            self._chatting = False

    # ---------- organic camera (naturalness) ----------
    def _start_camera(self) -> None:
        self._cam_on = True
        self._cam_thread = threading.Thread(target=self._camera_loop, name="organic-cam", daemon=True)
        self._cam_thread.start()

    def _camera_loop(self) -> None:
        """Gentle human-like glances while Baritone walks/mines. freeLook=true means
        these are purely cosmetic and never fight Baritone's pathing."""
        import random
        while self._cam_on:
            # idle for a human-ish beat, but stay responsive to stop/chat
            wait = random.uniform(2.5, 6.5)
            t0 = time.time()
            while time.time() - t0 < wait:
                if not self._cam_on:
                    return
                time.sleep(0.1)
            if self._chatting or InputController.abort_requested():
                continue
            # a small look-around: mostly horizontal, occasional slight tilt
            dx = random.choice([-1, 1]) * random.randint(55, 150)
            dy = random.randint(-35, 45)
            self.ic.smooth_look(dx, dy, steps=random.randint(16, 26), dt=0.012)
            # often drift partway back so it doesn't slowly spin off
            if random.random() < 0.6 and self._cam_on and not self._chatting:
                time.sleep(random.uniform(0.3, 1.0))
                self.ic.smooth_look(int(-dx * 0.6), int(-dy * 0.6),
                                    steps=random.randint(14, 20), dt=0.012)

    async def _decide(self) -> dict:
        frame = self.capture.grab()
        self.mem.on_frame(frame.to_pil())
        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": [
                {"type": "text", "text": _build_user(self.goal, self.mem)},
                {"type": "image", "data": frame.jpeg, "mime": "image/jpeg"},
            ]},
        ]
        
        # free models occasionally return empty/partial — retry once if no JSON
        text = ""
        for attempt in range(2):
            text = ""
            async for chunk in self.provider.stream(
                messages, temperature=0.2, top_p=0.9, max_tokens=220,
                presence_penalty=0.0, frequency_penalty=0.0,
            ):
                text += chunk
            if "{" in text:
                break
            logger.warning(f"empty/garbage VLM output (attempt {attempt+1}) — retrying")

        try:
            d = _parse_baritone(text)
        except Exception:
            logger.warning(f"Failed to parse VLM output: {text!r}")
            return {"action": "wait", "text": "", "subgoal": "error", "note": "", "reason": ""}

        if "action" not in d:
            d["action"] = "wait"

        self.mem.on_action(d["action"], 0, d.get("note", ""), d.get("subgoal", ""))

        # Execute the action immediately
        act = d["action"]
        if act == "chat":
            txt = d.get("text", "").strip()
            if txt:
                self._execute_chat(txt)
                self._last_cmd_ts = time.time()
                self._idle_decisions = 0
            else:
                self._idle_decisions += 1   # said 'chat' but gave no command
        elif act == "stop_baritone":
            self._execute_chat("#stop")
            self._last_cmd_ts = time.time()
        elif act == "inventory":
            self.ic.tap("e", 0.1)
        elif act == "look_around":
            self.ic.smooth_look(300, 0, steps=20)
        else:  # wait
            self._idle_decisions += 1

        return d

    async def run(self) -> None:
        self._running = True
        logger.info(f"BaritoneAgent: running — goal={self.goal!r}  (F8 = stop)")
        # one-time: hide path/goal render + chat spam, enable freeLook for free camera
        self._setup_stream_clean()
        # KICK OFF immediately so the bot always starts working (don't wait on the LLM)
        if self._boot_cmd:
            logger.info(f"baritone: bootstrap → {self._boot_cmd}")
            self._execute_chat(self._boot_cmd)
            self._last_cmd_ts = time.time()
        # optional gentle organic camera (off by default; Baritone now drives the camera)
        if self._organic_cam:
            self._start_camera()
        try:
            while self._running:
                if InputController.abort_requested():
                    logger.warning("agent: F8 abort")
                    break

                try:
                    d = await self._decide()
                    tag = d["action"]
                    if tag == "chat":
                        tag += f" [{d.get('text', '')}]"
                    logger.info(f"act> {tag:20} | plan: {str(d.get('subgoal', ''))[:40]:40} | {str(d.get('reason', ''))[:60]}")
                except Exception as e:
                    logger.error(f"agent: decide error: {e}")

                # idle watchdog: if the model keeps stalling (no command) for a long
                # stretch, re-kick the bootstrap so the bot never just stands forever.
                if (self._boot_cmd and self._idle_decisions >= 4
                        and time.time() - self._last_cmd_ts > 90.0):
                    logger.info("baritone: idle too long — re-kicking bootstrap command")
                    self._execute_chat(self._boot_cmd)
                    self._last_cmd_ts = time.time()
                    self._idle_decisions = 0

                # Sleep a bit between VLM decisions so we don't spam the API
                # while Baritone is doing the hard work.
                await asyncio.sleep(2.0)
        finally:
            self._running = False
            self._cam_on = False
            self.ic.release_all()
            logger.info("agent: stopped")
