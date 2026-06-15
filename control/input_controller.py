"""Low-level synthetic input for game control — pure ctypes SendInput (no deps).

Games read DirectInput SCANCODES (not virtual keys) and RAW relative mouse motion,
so we send hardware-scancode key events and relative MOUSEEVENTF_MOVE deltas. This
is what makes camera-look and WASD actually register inside Minecraft Bedrock.

Safety: every key/button pressed is tracked and released by release_all(); the agent
loop should call it on stop and on abort.
"""
from __future__ import annotations

import ctypes
import random
import threading
import time
from ctypes import wintypes

_user32 = ctypes.windll.user32
_SendInput = _user32.SendInput

# --- input type / event flags ---
_INPUT_MOUSE = 0
_INPUT_KEYBOARD = 1
_KEYEVENTF_KEYUP = 0x0002
_KEYEVENTF_SCANCODE = 0x0008
_MOUSEEVENTF_MOVE = 0x0001
_MOUSEEVENTF_LEFTDOWN = 0x0002
_MOUSEEVENTF_LEFTUP = 0x0004
_MOUSEEVENTF_RIGHTDOWN = 0x0008
_MOUSEEVENTF_RIGHTUP = 0x0010

# DirectInput scancodes (US layout) for the keys a player actually uses.
SCAN = {
    "w": 0x11, "a": 0x1E, "s": 0x1F, "d": 0x20,
    "space": 0x39, "shift": 0x2A, "ctrl": 0x1D, "e": 0x12, "q": 0x10,
    "esc": 0x01, "f": 0x21, "t": 0x14, "v": 0x2F, "enter": 0x1C, "slash": 0x35,
    "1": 0x02, "2": 0x03, "3": 0x04, "4": 0x05, "5": 0x06,
    "6": 0x07, "7": 0x08, "8": 0x09, "9": 0x0A,
}

_ABORT_VK = 0x77  # F8 — panic key to bail out of any run
_CONTINUE_VK = 0x78  # F9 — "advance to next phase" (hybrid: press after you craft)

# F8 stop must be reliable: a background thread polls the global key state every ~15ms and LATCHES
# the abort the instant F8 is pressed — from ANY window. A single tap is enough; we no longer rely
# on the agent happening to sample the key while it's held.
_abort_latched = False
_watcher_started = False
_watcher_lock = threading.Lock()


def _abort_watcher() -> None:
    global _abort_latched
    while not _abort_latched:
        if _user32.GetAsyncKeyState(_ABORT_VK) & 0x8000:
            _abort_latched = True
            return
        time.sleep(0.015)


def _ensure_abort_watcher() -> None:
    global _watcher_started
    with _watcher_lock:
        if _watcher_started:
            return
        _watcher_started = True
        threading.Thread(target=_abort_watcher, name="wallie-f8-watch", daemon=True).start()


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG),
                ("mouseData", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.POINTER(wintypes.ULONG))]


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD),
                ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD),
                ("dwExtraInfo", ctypes.POINTER(wintypes.ULONG))]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("mi", _MOUSEINPUT), ("ki", _KEYBDINPUT)]


class _INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("u", _INPUTUNION)]


def _send(inp: _INPUT) -> None:
    _SendInput(1, ctypes.byref(inp), ctypes.sizeof(_INPUT))


class InputController:
    def __init__(self) -> None:
        self._keys_down: set[str] = set()
        self._mouse_down: set[str] = set()
        # smooth mouse driver: a high-rate thread emits tiny moves toward a target
        # look-velocity (px/sec). Decouples buttery camera motion from the slower
        # control loop, and accumulates fractional pixels so slow pans aren't jerky.
        self._look_vel = (0.0, 0.0)
        self._acc = [0.0, 0.0]
        self._driver_on = False
        self._driver_thread: threading.Thread | None = None

    # ---------- smooth mouse driver ----------
    def start_mouse_driver(self, hz: int = 120) -> None:
        if self._driver_on:
            return
        self._driver_on = True
        self._driver_thread = threading.Thread(
            target=self._driver_loop, args=(hz,), name="mouse-driver", daemon=True)
        self._driver_thread.start()

    def stop_mouse_driver(self) -> None:
        self._driver_on = False
        self._look_vel = (0.0, 0.0)

    def set_look_velocity(self, vx: float, vy: float) -> None:
        """Target camera pan speed in px/sec; the driver eases toward it smoothly."""
        self._look_vel = (vx, vy)

    def _driver_loop(self, hz: int) -> None:
        dt = 1.0 / hz
        last = time.time()
        cvx = cvy = 0.0  # current (eased) velocity → no abrupt starts/stops
        while self._driver_on:
            now = time.time()
            e = min(0.05, now - last)
            last = now
            tvx, tvy = self._look_vel
            # exponential ease toward target velocity (smooth accel/decel)
            a = 1.0 - pow(0.0015, e)  # ~time-constant ease
            cvx += (tvx - cvx) * a
            cvy += (tvy - cvy) * a
            self._acc[0] += cvx * e
            self._acc[1] += cvy * e
            mx, my = int(self._acc[0]), int(self._acc[1])
            if mx or my:
                self.move_rel(mx, my)
                self._acc[0] -= mx
                self._acc[1] -= my
            time.sleep(dt)

    # ---------- keyboard ----------
    def key_down(self, name: str) -> None:
        sc = SCAN[name]
        _send(_INPUT(type=_INPUT_KEYBOARD,
                     u=_INPUTUNION(ki=_KEYBDINPUT(0, sc, _KEYEVENTF_SCANCODE, 0, None))))
        self._keys_down.add(name)

    def key_up(self, name: str) -> None:
        sc = SCAN[name]
        _send(_INPUT(type=_INPUT_KEYBOARD,
                     u=_INPUTUNION(ki=_KEYBDINPUT(0, sc, _KEYEVENTF_SCANCODE | _KEYEVENTF_KEYUP, 0, None))))
        self._keys_down.discard(name)

    def tap(self, name: str, hold: float = 0.05) -> None:
        self.key_down(name)
        time.sleep(hold)
        self.key_up(name)

    def set_key(self, name: str, pressed: bool) -> None:
        """Idempotent hold/release — the fast loop calls this every tick."""
        if pressed and name not in self._keys_down:
            self.key_down(name)
        elif not pressed and name in self._keys_down:
            self.key_up(name)

    # ---------- mouse buttons ----------
    def mouse_down(self, button: str = "left") -> None:
        flag = _MOUSEEVENTF_LEFTDOWN if button == "left" else _MOUSEEVENTF_RIGHTDOWN
        _send(_INPUT(type=_INPUT_MOUSE, u=_INPUTUNION(mi=_MOUSEINPUT(0, 0, 0, flag, 0, None))))
        self._mouse_down.add(button)

    def mouse_up(self, button: str = "left") -> None:
        flag = _MOUSEEVENTF_LEFTUP if button == "left" else _MOUSEEVENTF_RIGHTUP
        _send(_INPUT(type=_INPUT_MOUSE, u=_INPUTUNION(mi=_MOUSEINPUT(0, 0, 0, flag, 0, None))))
        self._mouse_down.discard(button)

    def click(self, button: str = "left", hold: float = 0.04) -> None:
        self.mouse_down(button)
        time.sleep(hold)
        self.mouse_up(button)

    def set_mouse(self, button: str, pressed: bool) -> None:
        if pressed and button not in self._mouse_down:
            self.mouse_down(button)
        elif not pressed and button in self._mouse_down:
            self.mouse_up(button)

    # ---------- mouse cursor (absolute, for menus/UI like the respawn button) ----------
    def move_abs(self, fx: float, fy: float) -> None:
        """Place the cursor at a fractional screen position (0..1). Menus release the
        raw-input capture, so SetCursorPos works there."""
        sw = _user32.GetSystemMetrics(0)
        sh = _user32.GetSystemMetrics(1)
        _user32.SetCursorPos(int(fx * sw), int(fy * sh))

    # ---------- mouse look (relative) ----------
    def move_rel(self, dx: int, dy: int) -> None:
        _send(_INPUT(type=_INPUT_MOUSE,
                     u=_INPUTUNION(mi=_MOUSEINPUT(int(dx), int(dy), 0, _MOUSEEVENTF_MOVE, 0, None))))

    def smooth_look(self, dx: int, dy: int, steps: int = 24, dt: float = 0.008) -> None:
        """Turn the camera by (dx,dy) total, split into small eased steps so it looks
        like a human flick rather than a teleport. Aborts if F8 is pressed."""
        if steps < 1:
            steps = 1
        moved_x = moved_y = 0.0
        for i in range(1, steps + 1):
            if self.abort_requested():
                return
            # ease-in-out so the flick accelerates then settles
            p0 = _ease((i - 1) / steps)
            p1 = _ease(i / steps)
            tx, ty = dx * p1, dy * p1
            self.move_rel(round(tx - moved_x), round(ty - moved_y))
            moved_x, moved_y = tx, ty
            time.sleep(dt)

    def look_jitter(self, amount: int = 3) -> None:
        """Tiny natural camera drift — used while mining/idling so it isn't a statue."""
        self.move_rel(random.randint(-amount, amount), random.randint(-amount, amount))

    # ---------- safety ----------
    @staticmethod
    def abort_requested() -> bool:
        _ensure_abort_watcher()
        return _abort_latched

    @staticmethod
    def continue_requested() -> bool:
        return bool(_user32.GetAsyncKeyState(_CONTINUE_VK) & 0x8000)

    def release_all(self) -> None:
        self._look_vel = (0.0, 0.0)
        self._acc = [0.0, 0.0]
        for name in list(self._keys_down):
            self.key_up(name)
        for btn in list(self._mouse_down):
            self.mouse_up(btn)


def _ease(t: float) -> float:
    """Smoothstep ease-in-out."""
    return t * t * (3 - 2 * t)
