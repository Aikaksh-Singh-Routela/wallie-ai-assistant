"""VTube Studio WebSocket client for Wallie.

Five layers of animation, all running over a single WS:

    * lipsync          — speaker drives MouthOpen + a low-floor MouthSmile
    * idle motion      — quiet drift on FaceAngleX/Y plus occasional eye darts
    * blink            — periodic natural eye blinks with double-blink variation
    * body motion      — slow torso/body sway independent of head movement
    * expressions      — VTS hotkeys triggered by sentence content / events

Mood-reactive: the orchestrator pushes arousal/valence/focus each turn, which
modulates idle amplitude, eye-dart frequency, blink rate, brow position, and
resting smile — so the avatar feels alive and emotionally congruent.

The class auto-reconnects every 5s if VTS isn't running and never blocks the
audio pipeline. All public coroutines are safe to call before/while/without a
live connection: they short-circuit when not authenticated.

Discovery helpers (`query_hotkeys`, `query_model_info`) let the dashboard offer
real dropdowns instead of asking the user to type hotkey names by hand.

Expression auto-mapping: on connect, unset expression slots are matched against
discovered hotkey names so users don't need to configure every slot manually.
"""
from __future__ import annotations

import asyncio
import json
import math
import random
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from loguru import logger

if TYPE_CHECKING:
    from config import AvatarConfig

_TOKEN_FILE = Path(__file__).resolve().parent.parent / ".wallie_vts_token"

_PLUGIN_NAME = "Wallie"
_PLUGIN_DEV = "WallieStreamer"

_RECONNECT_DELAY = 5.0
_SEND_TIMEOUT = 2.0
_QUERY_TIMEOUT = 5.0


def _msg(message_type: str, data: dict) -> str:
    return json.dumps(
        {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": str(uuid.uuid4()),
            "messageType": message_type,
            "data": data,
        }
    )


class VTubeStudioAvatar:
    """Async VTS client. Construct once, call ``connect()`` as a long-lived task."""

    def __init__(self, cfg: "AvatarConfig") -> None:
        self._cfg = cfg
        self._ws: Any = None  # websockets.WebSocketClientProtocol | None
        self._send_lock = asyncio.Lock()
        self._ready = False
        self._running = True
        self._speaking = False

        # Lipsync envelope state.
        self._mouth_current: float = 0.0
        self._smile_current: float = 0.0
        self._form_current: float = 0.5

        # Animation task handles.
        self._idle_task: Optional[asyncio.Task] = None
        self._eye_dart_task: Optional[asyncio.Task] = None
        self._blink_task: Optional[asyncio.Task] = None
        self._body_task: Optional[asyncio.Task] = None
        self._connect_started_at: float = 0.0

        # Mood-reactive state (updated by orchestrator each turn).
        self._mood_arousal: float = 0.55
        self._mood_valence: float = 0.15
        self._mood_focus: float = 0.75
        self._brow_current: float = 0.0
        self._resting_smile: float = 0.0

        # Pending discovery responses, keyed by requestID.
        self._pending: dict[str, asyncio.Future] = {}

        # Last known model info (populated after auth).
        self._model_info: dict[str, Any] = {}
        self._hotkeys: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Top-level keep-alive. Reconnects on failure, never raises."""
        while self._running:
            try:
                await self._connect_once()
            except Exception as exc:
                logger.warning(f"avatar: VTS connection lost: {exc!r}  — retry in {_RECONNECT_DELAY}s")
            finally:
                self._ready = False
                self._ws = None
                self._cancel_animation_tasks()
            if self._running:
                await asyncio.sleep(_RECONNECT_DELAY)

    async def close(self) -> None:
        self._running = False
        self._cancel_animation_tasks()
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass

    @property
    def is_connected(self) -> bool:
        return self._ready

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self._cfg.enabled,
            "connected": self._ready,
            "host": f"{self._cfg.vts_host}:{self._cfg.vts_port}",
            "speaking": self._speaking,
            "model": self._model_info.get("modelName") or self._model_info.get("name"),
            "model_id": self._model_info.get("modelID") or self._model_info.get("id"),
            "hotkey_count": len(self._hotkeys),
            "uptime_sec": round(time.time() - self._connect_started_at, 1) if self._ready else 0.0,
            "mood_arousal": round(self._mood_arousal, 2),
            "mood_valence": round(self._mood_valence, 2),
            "mood_focus": round(self._mood_focus, 2),
        }

    # ------------------------------------------------------------------
    # Public control surface
    # ------------------------------------------------------------------

    async def set_speaking(self, speaking: bool) -> None:
        """Hard cue at sentence start/end. Drives a small permanent smile while speaking."""
        self._speaking = speaking
        if not speaking:
            self._mouth_current = 0.0
            self._form_current = 0.5
            params = {
                self._cfg.param_mouth_open: 0.0,
                self._cfg.param_mouth_smile: max(0.0, self._smile_current),
            }
            if self._cfg.enable_viseme_lipsync:
                params[self._cfg.param_mouth_form] = 0.5
            await self._inject(params)

    async def set_volume(self, rms: float) -> None:
        """PCM RMS → MouthOpen with attack/release envelope and a noise floor."""
        if rms < self._cfg.lipsync_floor:
            target = 0.0
        else:
            target = min(1.0, rms * self._cfg.lipsync_gain) * self._cfg.lipsync_ceiling

        # Asymmetric envelope: snappier opening, smoother closing.
        if target > self._mouth_current:
            a = self._cfg.lipsync_attack
            self._mouth_current = self._mouth_current + (target - self._mouth_current) * a
        else:
            r = self._cfg.lipsync_release
            self._mouth_current = self._mouth_current + (target - self._mouth_current) * r

        # Smile: speaking_smile while talking, mood resting_smile when idle.
        smile_target = self._cfg.speaking_smile if self._speaking else self._resting_smile
        self._smile_current = self._smile_current + (smile_target - self._smile_current) * 0.15

        await self._inject(
            {
                self._cfg.param_mouth_open: round(self._mouth_current, 3),
                self._cfg.param_mouth_smile: round(self._smile_current, 3),
            }
        )

    async def feed_audio(self, pcm: bytes, sample_rate: int = 24000) -> None:
        """Drive lip sync from raw PCM: RMS for mouth open + spectral analysis for mouth shape."""
        if not pcm or len(pcm) < 4:
            return
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        rms = float(np.sqrt(np.mean(samples ** 2))) / 32768.0
        rms = min(1.0, rms)

        # Mouth open envelope (same logic as set_volume).
        if rms < self._cfg.lipsync_floor:
            target = 0.0
        else:
            target = min(1.0, rms * self._cfg.lipsync_gain) * self._cfg.lipsync_ceiling

        if target > self._mouth_current:
            self._mouth_current += (target - self._mouth_current) * self._cfg.lipsync_attack
        else:
            self._mouth_current += (target - self._mouth_current) * self._cfg.lipsync_release

        # Smile envelope.
        smile_target = self._cfg.speaking_smile if self._speaking else self._resting_smile
        self._smile_current += (smile_target - self._smile_current) * 0.15

        params: dict[str, float] = {
            self._cfg.param_mouth_open: round(self._mouth_current, 3),
            self._cfg.param_mouth_smile: round(self._smile_current, 3),
        }

        # Viseme: spectral shape → mouth form (wide vs round).
        if self._cfg.enable_viseme_lipsync and rms > self._cfg.lipsync_floor:
            form = self._estimate_mouth_form(samples / 32768.0, sample_rate)
            smooth = self._cfg.viseme_smoothing
            self._form_current += (form - self._form_current) * smooth
            params[self._cfg.param_mouth_form] = round(self._form_current, 3)
        elif self._cfg.enable_viseme_lipsync:
            # Silence — relax mouth form toward neutral.
            self._form_current += (0.5 - self._form_current) * 0.15
            params[self._cfg.param_mouth_form] = round(self._form_current, 3)

        await self._inject(params)

    def _estimate_mouth_form(self, samples: np.ndarray, sr: int) -> float:
        """Spectral band ratio → mouth form.
        High-frequency dominance (front vowels A/E/I) → wide (1.0).
        Low-frequency dominance (back vowels O/U) → round (0.0).
        """
        n = len(samples)
        if n < 256:
            return self._form_current

        window = np.hanning(n)
        spectrum = np.abs(np.fft.rfft(samples * window))
        freqs = np.fft.rfftfreq(n, 1.0 / sr)

        # Band energy: low (300-1200 Hz) vs high (1500-3500 Hz).
        low_mask = (freqs >= 300) & (freqs <= 1200)
        high_mask = (freqs >= 1500) & (freqs <= 3500)
        low_energy = float(spectrum[low_mask].sum()) if low_mask.any() else 0.0
        high_energy = float(spectrum[high_mask].sum()) if high_mask.any() else 0.0

        total = low_energy + high_energy
        if total < 1e-6:
            return self._form_current

        # Ratio: 0 = all low (round), 1 = all high (wide).
        return high_energy / total

    async def set_smile(self, amount: float) -> None:
        """Manual smile control (for chat reactions etc.)."""
        amount = max(0.0, min(1.0, amount))
        self._smile_current = amount
        await self._inject({self._cfg.param_mouth_smile: round(amount, 3)})

    async def look_at(self, x: float, y: float, *, hold_sec: float = 0.6) -> None:
        """Glance briefly. x, y in [-30, 30] degrees roughly."""
        x = max(-30.0, min(30.0, x))
        y = max(-30.0, min(30.0, y))
        await self._inject({self._cfg.param_face_x: x, self._cfg.param_face_y: y})
        if hold_sec > 0:
            await asyncio.sleep(hold_sec)

    async def trigger_expression(self, name: str) -> None:
        """Fire a VTS hotkey by name OR id. Resolves via the cached hotkey list."""
        if not name or not self._ready:
            return
        hotkey_id = self._resolve_hotkey_id(name)
        if not hotkey_id:
            logger.debug(f"avatar: no matching hotkey for '{name}'")
            return
        await self._send(_msg("HotkeyTriggerRequest", {"hotkeyID": hotkey_id}))

    async def trigger_emotion(self, slot: str) -> None:
        """High-level: ``slot`` is e.g. 'happy', 'hype', 'thinking'.

        Resolves to ``cfg.expr_<slot>`` and fires it. Centralised so callers
        don't need to know the hotkey name strings.
        """
        if not slot:
            return
        name = getattr(self._cfg, f"expr_{slot}", "") or ""
        if name:
            await self.trigger_expression(name)

    async def update_mood(self, arousal: float, valence: float, focus: float) -> None:
        """Sync mood state from the orchestrator. Adjusts brow, resting smile,
        and modulates idle/body/blink behaviour via stored mood values.

        Called once per orchestrator turn — cheap, never blocks.
        """
        self._mood_arousal = arousal
        self._mood_valence = valence
        self._mood_focus = focus

        if not self._ready or not self._cfg.enable_mood_link:
            return

        # Brow: valence (-1..1) → brow offset range.
        t = (valence + 1.0) / 2.0  # normalise to 0..1
        brow_target = self._cfg.mood_brow_min + (self._cfg.mood_brow_max - self._cfg.mood_brow_min) * t
        self._brow_current += (brow_target - self._brow_current) * 0.12

        # Resting smile: positive valence → subtle smile when not speaking.
        smile_target = max(0.0, valence) * self._cfg.mood_smile_max
        self._resting_smile += (smile_target - self._resting_smile) * 0.12

        params: dict[str, float] = {self._cfg.param_brows: round(self._brow_current, 3)}
        if not self._speaking:
            params[self._cfg.param_mouth_smile] = round(max(self._resting_smile, self._smile_current), 3)
        await self._inject(params)

    # ------------------------------------------------------------------
    # Discovery — used by the dashboard for nicer pickers
    # ------------------------------------------------------------------

    async def query_hotkeys(self) -> list[dict[str, Any]]:
        """Fetch the model's available hotkeys from VTS."""
        if not self._ready:
            return self._hotkeys
        resp = await self._request("HotkeysInCurrentModelRequest", {})
        if resp:
            self._hotkeys = resp.get("availableHotkeys", []) or []
        return self._hotkeys

    async def query_model_info(self) -> dict[str, Any]:
        if not self._ready:
            return self._model_info
        resp = await self._request("CurrentModelRequest", {})
        if resp:
            self._model_info = resp
        return self._model_info

    # ------------------------------------------------------------------
    # Mood helpers
    # ------------------------------------------------------------------

    def _mood_amplitude_scale(self) -> float:
        """Map arousal → idle/body sway amplitude multiplier."""
        if not self._cfg.enable_mood_link:
            return 1.0
        lo, hi = self._cfg.mood_idle_min_scale, self._cfg.mood_idle_max_scale
        return lo + (hi - lo) * self._mood_arousal

    def _mood_dart_interval_scale(self) -> float:
        """Low focus → more frequent darts (shorter interval)."""
        if not self._cfg.enable_mood_link:
            return 1.0
        # focus 1.0 → scale 1.3 (less darts); focus 0.0 → scale 0.5 (more darts)
        return 0.5 + 0.8 * self._mood_focus

    # ------------------------------------------------------------------
    # Idle motion
    # ------------------------------------------------------------------

    async def _idle_loop(self) -> None:
        """Slow head sway while not speaking. Amplitude scales with mood arousal."""
        period = max(2.0, self._cfg.idle_sway_period_sec)
        # Two independent phases so X and Y don't move in lockstep.
        phase_x = random.random() * math.tau
        phase_y = random.random() * math.tau
        try:
            while self._ready and self._cfg.enable_idle_motion:
                if not self._speaking:
                    t = time.time()
                    amp = self._cfg.idle_sway_amplitude * self._mood_amplitude_scale()
                    angle_x = amp * math.sin(t * math.tau / period + phase_x)
                    angle_y = amp * 0.6 * math.sin(t * math.tau / (period * 1.3) + phase_y)
                    await self._inject(
                        {self._cfg.param_face_x: round(angle_x, 2), self._cfg.param_face_y: round(angle_y, 2)}
                    )
                await asyncio.sleep(0.08)  # ~12 Hz update is plenty
        except asyncio.CancelledError:
            return

    async def _eye_dart_loop(self) -> None:
        """Random small saccades every few seconds while idle.
        Frequency increases when mood focus is low (scattered attention).
        """
        try:
            while self._ready and self._cfg.enable_eye_darts:
                base = self._cfg.eye_dart_interval_sec * self._mood_dart_interval_scale()
                interval = base * (0.6 + random.random() * 0.8)
                await asyncio.sleep(interval)
                if self._speaking:
                    continue
                # Small dart, then quick return to centre.
                dx = random.uniform(-0.6, 0.6)
                dy = random.uniform(-0.3, 0.3)
                await self._inject(
                    {self._cfg.param_eye_x: round(dx, 2), self._cfg.param_eye_y: round(dy, 2)}
                )
                await asyncio.sleep(random.uniform(0.15, 0.4))
                await self._inject({self._cfg.param_eye_x: 0.0, self._cfg.param_eye_y: 0.0})
        except asyncio.CancelledError:
            return

    # ------------------------------------------------------------------
    # Blink
    # ------------------------------------------------------------------

    async def _blink_loop(self) -> None:
        """Periodic natural eye blinks with occasional double-blinks.
        Blink rate adapts to mood: low arousal (sleepy) → more frequent blinks.
        """
        try:
            while self._ready and self._cfg.enable_blink:
                interval = self._cfg.blink_interval_sec * (0.6 + random.random() * 0.8)
                if self._cfg.enable_mood_link:
                    # Alert (high arousal) → slower blinks; sleepy → faster.
                    interval *= 0.7 + 0.6 * self._mood_arousal
                await asyncio.sleep(interval)
                await self._do_blink()
                # Occasional double-blink for naturalism.
                if random.random() < self._cfg.double_blink_chance:
                    await asyncio.sleep(0.12)
                    await self._do_blink()
        except asyncio.CancelledError:
            return

    async def _do_blink(self) -> None:
        """Execute a single blink: quick close → hold → slower open."""
        L = self._cfg.param_eye_open_left
        R = self._cfg.param_eye_open_right
        # Close (fast, 2 steps)
        await self._inject({L: 0.15, R: 0.15})
        await asyncio.sleep(0.03)
        await self._inject({L: 0.0, R: 0.0})
        # Hold closed
        await asyncio.sleep(self._cfg.blink_hold_sec)
        # Open (slower, 3 steps for smooth reveal)
        await self._inject({L: 0.35, R: 0.35})
        await asyncio.sleep(0.035)
        await self._inject({L: 0.75, R: 0.75})
        await asyncio.sleep(0.035)
        await self._inject({L: 1.0, R: 1.0})

    # ------------------------------------------------------------------
    # Body motion
    # ------------------------------------------------------------------

    async def _body_loop(self) -> None:
        """Slow torso sway. Lower amplitude and longer period than head sway.
        Amplitude scales with mood arousal, same as head idle.
        """
        period = max(4.0, self._cfg.body_sway_period_sec)
        phase_x = random.random() * math.tau
        phase_y = random.random() * math.tau
        phase_z = random.random() * math.tau
        try:
            while self._ready and self._cfg.enable_body_motion:
                if not self._speaking:
                    t = time.time()
                    amp = self._cfg.body_sway_amplitude * self._mood_amplitude_scale()
                    bx = amp * math.sin(t * math.tau / period + phase_x)
                    by = amp * 0.4 * math.sin(t * math.tau / (period * 1.4) + phase_y)
                    bz = amp * 0.3 * math.sin(t * math.tau / (period * 0.7) + phase_z)
                    await self._inject({
                        self._cfg.param_body_x: round(bx, 2),
                        self._cfg.param_body_y: round(by, 2),
                        self._cfg.param_body_z: round(bz, 2),
                    })
                await asyncio.sleep(0.08)
        except asyncio.CancelledError:
            return

    # ------------------------------------------------------------------
    # Animation task management
    # ------------------------------------------------------------------

    def _cancel_animation_tasks(self) -> None:
        for t in (self._idle_task, self._eye_dart_task, self._blink_task, self._body_task):
            if t and not t.done():
                t.cancel()
        self._idle_task = None
        self._eye_dart_task = None
        self._blink_task = None
        self._body_task = None

    # ------------------------------------------------------------------
    # Connection internals
    # ------------------------------------------------------------------

    async def _connect_once(self) -> None:
        import websockets

        url = f"ws://{self._cfg.vts_host}:{self._cfg.vts_port}"
        logger.info(f"avatar: connecting to VTS at {url}")
        async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
            self._ws = ws
            await self._authenticate(ws)
            self._ready = True
            self._connect_started_at = time.time()
            logger.info("avatar: VTS connected and authenticated")

            # Discover model + hotkeys so trigger_expression() can resolve names.
            try:
                await self.query_model_info()
                await self.query_hotkeys()
                logger.info(
                    f"avatar: model '{self._model_info.get('modelName', '?')}' loaded "
                    f"with {len(self._hotkeys)} hotkeys"
                )
            except Exception as e:
                logger.debug(f"avatar: discovery failed: {e}")

            # Auto-map empty expression slots from discovered hotkeys.
            await self._auto_map_expressions()

            # Start all animation loops alongside the recv loop.
            if self._cfg.enable_idle_motion:
                self._idle_task = asyncio.create_task(self._idle_loop(), name="vts-idle")
            if self._cfg.enable_eye_darts:
                self._eye_dart_task = asyncio.create_task(self._eye_dart_loop(), name="vts-eyes")
            if self._cfg.enable_blink:
                self._blink_task = asyncio.create_task(self._blink_loop(), name="vts-blink")
            if self._cfg.enable_body_motion:
                self._body_task = asyncio.create_task(self._body_loop(), name="vts-body")

            # Read incoming messages — server pushes responses to our requests
            # via this channel; pending futures get resolved here.
            async for raw in ws:
                self._dispatch(raw)

    async def _authenticate(self, ws) -> None:
        token = self._load_token()
        if not token:
            await ws.send(
                _msg(
                    "AuthenticationTokenRequest",
                    {"pluginName": _PLUGIN_NAME, "pluginDeveloper": _PLUGIN_DEV},
                )
            )
            raw = await asyncio.wait_for(ws.recv(), timeout=30.0)
            data = json.loads(raw).get("data", {})
            token = data.get("authenticationToken", "")
            if not token:
                raise RuntimeError(f"avatar: token request failed: {data}")
            self._save_token(token)
            logger.info("avatar: new VTS plugin token saved (user approved in app)")

        await ws.send(
            _msg(
                "AuthenticationRequest",
                {
                    "pluginName": _PLUGIN_NAME,
                    "pluginDeveloper": _PLUGIN_DEV,
                    "authenticationToken": token,
                },
            )
        )
        raw = await asyncio.wait_for(ws.recv(), timeout=10.0)
        data = json.loads(raw).get("data", {})
        if not data.get("authenticated"):
            _TOKEN_FILE.unlink(missing_ok=True)
            raise RuntimeError(f"avatar: authentication rejected: {data}")

    def _dispatch(self, raw: Any) -> None:
        """Resolve any pending request futures using the requestID echoed back."""
        try:
            obj = json.loads(raw)
        except Exception:
            return
        req_id = obj.get("requestID")
        fut = self._pending.pop(req_id, None) if req_id else None
        if fut and not fut.done():
            fut.set_result(obj.get("data", {}))

    # ------------------------------------------------------------------
    # Sends
    # ------------------------------------------------------------------

    async def _inject(self, params: dict[str, float]) -> None:
        if not params:
            return
        await self._send(
            _msg(
                "InjectParameterDataRequest",
                {
                    "faceFound": False,
                    "mode": "set",
                    "parameterValues": [{"id": pid, "value": v} for pid, v in params.items()],
                },
            )
        )

    async def _send(self, payload: str) -> None:
        if not self._ready or self._ws is None:
            return
        try:
            async with self._send_lock:
                await asyncio.wait_for(self._ws.send(payload), timeout=_SEND_TIMEOUT)
        except Exception as exc:
            logger.debug(f"avatar: send failed ({exc!r}); will reconnect")
            self._ready = False

    async def _request(self, message_type: str, data: dict) -> dict[str, Any]:
        """Send a request and await its matching response by requestID."""
        if not self._ready or self._ws is None:
            return {}
        req_id = str(uuid.uuid4())
        payload = json.dumps(
            {
                "apiName": "VTubeStudioPublicAPI",
                "apiVersion": "1.0",
                "requestID": req_id,
                "messageType": message_type,
                "data": data,
            }
        )
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending[req_id] = fut
        try:
            async with self._send_lock:
                await asyncio.wait_for(self._ws.send(payload), timeout=_SEND_TIMEOUT)
            return await asyncio.wait_for(fut, timeout=_QUERY_TIMEOUT)
        except Exception as e:
            self._pending.pop(req_id, None)
            logger.debug(f"avatar: request {message_type} failed: {e}")
            return {}

    # ------------------------------------------------------------------
    # Hotkey resolution
    # ------------------------------------------------------------------

    def _resolve_hotkey_id(self, name_or_id: str) -> str:
        """Match the supplied string against the cached hotkey list by id, name, or file."""
        n = name_or_id.strip().lower()
        for hk in self._hotkeys:
            if hk.get("hotkeyID") == name_or_id:
                return name_or_id
        for hk in self._hotkeys:
            for key in ("hotkeyID", "name", "file", "type"):
                v = (hk.get(key) or "").lower()
                if v and v == n:
                    return hk.get("hotkeyID", name_or_id)
        for hk in self._hotkeys:
            v = (hk.get("name") or "").lower()
            if v and n in v:
                return hk.get("hotkeyID", name_or_id)
        # Unknown — let VTS try anyway (might still match by raw id).
        return name_or_id

    # ------------------------------------------------------------------
    # Expression auto-mapping
    # ------------------------------------------------------------------

    _SLOT_KEYWORDS: dict[str, list[str]] = {
        "happy":     ["happy", "smile", "joy", "cheerful"],
        "surprised": ["surprise", "shock", "gasp", "amazed"],
        "laughing":  ["laugh", "giggle", "lol", "chuckle"],
        "angry":     ["angry", "anger", "mad", "rage", "furious"],
        "sad":       ["sad", "cry", "tear", "sorrow"],
        "thinking":  ["think", "ponder", "hmm", "wonder"],
        "smug":      ["smug", "confident", "cool", "cocky"],
        "eyeroll":   ["eyeroll", "roll", "sigh", "annoyed"],
        "confused":  ["confus", "puzzle", "huh", "question"],
        "hype":      ["hype", "excit", "cheer", "yay", "celebrate"],
        "deadpan":   ["deadpan", "flat", "blank", "neutral", "bored"],
    }

    async def _auto_map_expressions(self) -> None:
        """Best-effort: match discovered hotkeys to empty expression slots."""
        if not self._cfg.auto_map_expressions or not self._hotkeys:
            return

        mapped: list[str] = []
        for slot, keywords in self._SLOT_KEYWORDS.items():
            if getattr(self._cfg, f"expr_{slot}", ""):
                continue  # already configured by user
            for hk in self._hotkeys:
                label = f"{hk.get('name', '')} {hk.get('file', '')}".lower()
                if any(kw in label for kw in keywords):
                    hid = hk.get("hotkeyID", "")
                    if hid:
                        setattr(self._cfg, f"expr_{slot}", hid)
                        mapped.append(f"{slot} -> {hk.get('name', hid)}")
                        break

        if mapped:
            logger.info(f"avatar: auto-mapped expressions: {', '.join(mapped)}")

    # ------------------------------------------------------------------
    # Token persistence
    # ------------------------------------------------------------------

    @staticmethod
    def _load_token() -> str:
        try:
            return _TOKEN_FILE.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return ""

    @staticmethod
    def _save_token(token: str) -> None:
        _TOKEN_FILE.write_text(token, encoding="utf-8")
