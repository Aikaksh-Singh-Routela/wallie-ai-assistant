"""The single pipeline.

Everything the streamer says goes through one place:

    intent  ->  LLM stream (tokens)
              ->  SentenceStreamer (sentences)
              ->  TTS.synthesize (PCM chunks)
              ->  AudioPlayer.write

A single turn looks like "decide intent, speak one segment, update history, loop".
Intents are ranked by urgency:

    1. Highlight chat (super chat / bits / donation)  -> barge in, reply
    2. Vision change event                            -> react on next turn
    3. Ordinary chat                                  -> reply if under rate limit
    4. Monologue                                      -> default fallback

Long sessions:
  * If session_duration_min > 0, the loop schedules a single OUTRO segment
    (signed off in character) when time-remaining drops under outro_seconds,
    then stops itself.
  * Every summarize_every_n delivered segments, a background task asks the LLM
    to compact the older portion of the history into rolling session_notes.
    That compressed memory is injected into the system prompt so the streamer
    keeps continuity across an hour-plus stream without bloating the context.
"""
from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Optional

from loguru import logger

from audio import AudioPlayer
from chat import ChatManager, ChatMessage
from config import AppConfig, Runtime
from core.context import Conversation, ImageBlock, pick_topic
from core.persona import Persona
from llm import LLMProvider
from tts import TTSProvider
from utils.sentences import SentenceStreamer

if TYPE_CHECKING:
    from avatar import VTubeStudioAvatar
    from vision import VisionEvent, VisionLoop

from vision.scene_classifier import ChangeType, ScreenActivity
from vision.vision_memory import SceneMemory, UserBehaviorTracker
from core.attention import AttentionEngine, VisionDirective, VisionReaction
from core.mood import MoodEngine
from core.memory_store import MemoryStore

IntentKind = Literal["chat", "vision", "monologue", "outro"]


@dataclass
class Intent:
    kind: IntentKind
    chat: Optional[ChatMessage] = None
    vision: Optional[VisionEvent] = None
    urgent: bool = False  # True -> interrupt current segment
    # When AttentionEngine routes a vision event, the directive flows through
    # so persona.vision_turn / monologue_turn can shape the prompt accordingly.
    vision_directive: Optional[VisionDirective] = None


class Orchestrator:
    def __init__(
        self,
        runtime: Runtime,
        persona: Persona,
        llm: LLMProvider,
        tts: TTSProvider,
        player: AudioPlayer,
        chat_manager: Optional[ChatManager] = None,
        vision_loop: Optional[VisionLoop] = None,
        vision_queue: Optional[asyncio.Queue[VisionEvent]] = None,
        avatar: Optional["VTubeStudioAvatar"] = None,
        memory_store: Optional[MemoryStore] = None,
    ) -> None:
        self._runtime = runtime
        self._cfg: AppConfig = runtime.config
        self._persona = persona
        self._llm = llm
        self._tts = tts
        self._player = player
        self._chat = chat_manager
        self._vision_loop = vision_loop
        self._vision_queue = vision_queue
        self._avatar = avatar
        self._memory = memory_store

        oc = self._cfg.orchestrator
        self._conv = Conversation(
            max_messages=oc.max_messages,
            max_chars=oc.max_chars,
            recent_verbatim_turns=oc.recent_verbatim_turns,
        )
        self._running = False
        self._main_task: Optional[asyncio.Task] = None
        self._segment_task: Optional[asyncio.Task] = None
        self._summarizer_task: Optional[asyncio.Task] = None
        self._recent_topics: list[str] = []
        self._last_chat_reply_ts = 0.0
        self._current_topic: Optional[str] = None
        self._last_spoken: str = ""
        self._last_monologue_spoken: str = ""   # only monologue/chat, never vision
        self._last_intent_kind: str = ""
        self._session_start_ts: float = 0.0
        self._outro_done: bool = False
        self._segments_since_summary: int = 0
        # Continuity trackers — fed back to the LLM each monologue turn.
        # Open threads: questions or teases the streamer left dangling. Each
        # next segment must pay them off before pivoting.
        self._open_threads: list[str] = []
        # Theme tags: short labels of what each recent segment covered.
        # Used to remind the model not to circle back to the same angle.
        self._recent_themes: list[str] = []
        # Phrase cooldown: catchphrases/gags used in recent segments, with the
        # segment index they were used in. Keeps signature lines from getting
        # repetitive.
        self._phrase_uses: dict[str, int] = {}
        # Track segments that ended with a question — used to throttle the
        # "every segment ends in a question" failure mode.
        self._segments_ended_with_question: int = 0
        # Latest screen frame. Updated whenever a vision event arrives. We
        # consume it ONLY for vision-intent segments. Monologue turns never
        # get the raw frame attached — vision-capable models ignore "IGNORE"
        # instructions and narrate the screen regardless.
        self._latest_frame: Optional["VisionEvent"] = None
        self._latest_frame_ts: float = 0.0
        # Scene memory: tracks what the AI last described so it can build on it.
        self._scene_memory: SceneMemory = SceneMemory()
        # User behavior tracker: tracks what the user is doing with the screen
        # (scrolling, navigating, settled, etc.) for organic adaptation.
        self._behavior: UserBehaviorTracker = UserBehaviorTracker()
        # Organic decision layer. AttentionEngine picks the next vision reaction
        # type (deep / glance / tangent / ignore / silence). MoodEngine feeds it
        # a slow energy/focus signal that drifts naturally over the stream.
        self._mood = MoodEngine(base_energy=self._cfg.persona.energy)
        self._attention = AttentionEngine(
            organicity=self._cfg.vision.organicity,
            min_vision_react_interval=self._cfg.vision.min_vision_react_interval_sec,
        )
        # Track when we last actually delivered a vision-anchored segment so the
        # policy can enforce a minimum re-react interval.
        self._last_vision_turn_ts: float = 0.0
        # Last directive picked by the attention engine, kept around so that
        # _build_user_turn can read it for the monologue/vision prompts.
        self._pending_directive: Optional[VisionDirective] = None
        # Break system: periodic pauses like a real streamer.
        self._on_break = False
        self._next_break_at = float("inf")
        self._enrich_this_turn = False
        self._break_event: asyncio.Event = asyncio.Event()

    # ----- lifecycle -----
    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._session_start_ts = time.time()
        self._outro_done = False
        self._segments_since_summary = 0
        self._on_break = False
        self._break_event = asyncio.Event()
        self._schedule_next_break()
        if self._memory:
            self._memory.load()
        self._player.start()
        if self._chat:
            await self._chat.start()
        if self._vision_loop:
            self._vision_loop.start()
        self._main_task = asyncio.create_task(self._run(), name="orchestrator")
        d = self._cfg.orchestrator.session_duration_min
        if d > 0:
            logger.info(f"orchestrator: started, session length {d:.1f} min")
        else:
            logger.info("orchestrator: started, unlimited session")

    async def stop(self) -> None:
        self._running = False
        for t in (self._segment_task, self._summarizer_task):
            if t and not t.done():
                t.cancel()
        if self._main_task:
            try:
                await asyncio.wait_for(self._main_task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._main_task.cancel()
        if self._vision_loop:
            await self._vision_loop.stop()
        if self._chat:
            await self._chat.stop()
        self._player.close()
        await self._llm.aclose()
        await self._tts.aclose()
        if self._memory:
            self._memory.save()
        logger.info("orchestrator: stopped")

    # ----- public status (for dashboard) -----
    def status(self) -> dict[str, Any]:
        elapsed = time.time() - self._session_start_ts if self._session_start_ts else 0.0
        d = self._cfg.orchestrator.session_duration_min
        remaining_sec = max(0.0, d * 60.0 - elapsed) if d > 0 else None
        return {
            "running": self._running,
            "current_topic": self._current_topic,
            "last_spoken": self._last_spoken[-300:] if self._last_spoken else "",
            "audio_queue_sec": round(self._player.seconds_queued(), 2),
            "llm": f"{self._llm.name}:{self._llm.model}",
            "tts": self._tts.name,
            "elapsed_sec": round(elapsed, 1),
            "remaining_sec": round(remaining_sec, 1) if remaining_sec is not None else None,
            "segments_spoken": self._conv.total_segments,
            "session_notes_chars": len(self._conv.session_notes),
            "session_notes_preview": (self._conv.session_notes[:300] + "…")
                if len(self._conv.session_notes) > 300
                else self._conv.session_notes,
            "verbatim_msgs": len(self._conv.messages()),
            "open_threads": list(self._open_threads),
            "recent_themes": list(self._recent_themes),
            "user_behavior": self._behavior.current_pattern,
            "user_settled": self._behavior.is_settled,
            "browsing_pace": round(self._behavior.browsing_pace, 2),
            "on_break": self._on_break,
            "next_break_in_sec": (
                round(max(0.0, self._next_break_at - time.time()), 1)
                if self._next_break_at != float("inf") else None
            ),
        }

    # ----- main loop -----
    async def _run(self) -> None:
        try:
            self._current_topic = self._pick_next_topic()
            while self._running:
                # Time check FIRST: if we've crossed the outro line, do that and stop.
                if self._should_outro() and not self._outro_done:
                    intent: Intent = Intent(kind="outro")
                    self._outro_done = True
                elif self._should_stop_immediately():
                    logger.info("orchestrator: session time up, stopping")
                    break
                else:
                    intent = await self._choose_intent()

                if intent.urgent and self._segment_task and not self._segment_task.done():
                    self._player.interrupt()
                    self._segment_task.cancel()
                    try:
                        await self._segment_task
                    except asyncio.CancelledError:
                        pass

                # Pre-segment avatar cue: visible "loading my next thought" beat.
                await self._cue_intent_expression(intent)

                self._segment_task = asyncio.create_task(self._run_segment(intent), name="segment")
                try:
                    await self._segment_task
                except asyncio.CancelledError:
                    pass

                if intent.kind == "outro":
                    # Wait for the outro audio to finish and then exit.
                    await self._player.wait_drained()
                    logger.info("orchestrator: outro played, stopping")
                    break

                # Maybe trigger a background summarize cycle.
                self._maybe_kick_summarizer()

                # Slow mood drift + avatar mood sync.
                self._mood.tick()
                await self._sync_mood_to_avatar()

                # Pipeline overlap: instead of waiting for ALL audio to drain,
                # start preparing the next segment while audio is still playing.
                # This eliminates the 3-5 second gap between segments.
                lookahead = self._cfg.orchestrator.max_audio_lookahead_sec
                if not self._has_urgent_pending():
                    if lookahead > 0:
                        # Wait until audio queue drops below the lookahead threshold.
                        while (self._player.seconds_queued() > lookahead
                               and self._running
                               and not self._has_urgent_pending()):
                            await asyncio.sleep(0.1)
                        # Mood-driven breathing gap when audio is nearly drained.
                        if self._player.seconds_queued() < 0.5:
                            gap = self._compute_breathing_gap()
                            if gap > 0:
                                await asyncio.sleep(gap)
                    else:
                        # lookahead=0: old behavior, full drain.
                        await self._player.wait_drained()

                # Periodic break: drain audio first, then pause.
                if self._should_take_break():
                    await self._player.wait_drained()
                    await self._take_break()
            self._running = False
        except Exception as e:
            logger.exception(f"orchestrator: fatal error: {e}")

    # ----- session timing -----
    def _should_outro(self) -> bool:
        d = self._cfg.orchestrator.session_duration_min
        if d <= 0 or not self._session_start_ts:
            return False
        remaining = d * 60.0 - (time.time() - self._session_start_ts)
        return remaining <= max(1.0, self._cfg.orchestrator.outro_seconds)

    def _should_stop_immediately(self) -> bool:
        d = self._cfg.orchestrator.session_duration_min
        if d <= 0 or not self._session_start_ts:
            return False
        return (time.time() - self._session_start_ts) >= d * 60.0 and self._outro_done

    # ----- intent selection -----
    async def _choose_intent(self) -> Intent:
        # 1. Highlight chat always preempts.
        highlight = self._pop_highlight_chat()
        if highlight:
            self._mood.on_highlight_chat()
            return Intent(kind="chat", chat=highlight, urgent=True)

        # 2. Vision event present → consult AttentionEngine for the organic call.
        vision = self._pop_latest_vision()
        if vision is not None:
            # Hard cooldown: if the last vision-anchored segment was too recent,
            # skip the attention engine entirely and fall through to monologue.
            # Media/game content gets a much shorter hard floor — the streamer
            # needs to keep reacting to what the audience is watching.
            vcfg = self._cfg.vision
            is_active_content = vision.activity.value in ("media", "app_switch")
            hard_floor = (
                vcfg.min_vision_react_interval_sec * 0.45 if is_active_content
                else vcfg.min_vision_react_interval_sec * 0.65
            )
            since_last_vision = time.time() - self._last_vision_turn_ts
            if self._last_vision_turn_ts > 0 and since_last_vision < hard_floor:
                logger.debug(
                    f"vision: hard cooldown ({since_last_vision:.1f}s < "
                    f"{hard_floor:.1f}s, active_content={is_active_content}), skipping"
                )
                vision = None  # fall through to chat/monologue

            if vision is not None:
                directive = self._decide_vision_reaction(vision)
                self._pending_directive = directive
                logger.debug(f"attention: {directive.rationale}")
            else:
                directive = None

            if directive is not None and directive.reaction == VisionReaction.SILENCE:
                # Take a beat. Don't generate audio this turn — but still mark
                # the silent slot so the speech-loop sleeps briefly.
                self._mood.on_silence_beat()
                self._attention.on_silence_beat()
                return Intent(kind="monologue", vision=None,
                              vision_directive=VisionDirective(
                                  reaction=VisionReaction.SILENCE,
                                  target_sentences=0,
                                  rationale=directive.rationale,
                              ))

            if directive is not None and directive.reaction == VisionReaction.IGNORE:
                # Pretend the vision event didn't happen.
                self._attention.on_vision_ignored()
                # Fall through to chat / monologue selection below.
            elif directive is not None and directive.reaction in (
                VisionReaction.DEEP, VisionReaction.GLANCE, VisionReaction.TANGENT,
            ):
                # Vision-anchored segment.
                self._mood.on_scene_change()
                self._attention.on_scene_change()
                return Intent(kind="vision", vision=vision, urgent=False,
                              vision_directive=directive)

        # 3. Ordinary chat under rate limit.
        ordinary = self._pop_ordinary_chat()
        if ordinary:
            self._mood.on_ordinary_chat()
            return Intent(kind="chat", chat=ordinary, urgent=False)

        # 4. Should we hold a silence beat instead of monologuing again?
        if self._cfg.vision.organic_vision and self._attention.should_hold_silence(
            mood_silence_probability=self._mood.silence_probability(),
            in_monologue_flow=self._segments_in_monologue_flow() >= 2,
        ):
            self._mood.on_silence_beat()
            self._attention.on_silence_beat()
            return Intent(kind="monologue", vision_directive=VisionDirective(
                reaction=VisionReaction.SILENCE, target_sentences=0,
                rationale="silence-beat: monologue chain too long",
            ))

        # 5. Enrich monologue: probabilistically snap a live frame and attach it so
        # the AI has the screen as additional context even without a change event.
        # CRITICAL: this flag controls whether _build_user_turn attaches a screen
        # frame. Without it, every cached frame leaks into every monologue and the
        # LLM talks about the screen for 10 minutes straight.
        self._enrich_this_turn = False
        vcfg = self._cfg.vision
        if (
            vcfg.enabled
            and vcfg.enrich_monologue
            and self._vision_loop is not None
            and random.random() < vcfg.enrich_probability
        ):
            evt = await self._vision_loop.grab_now()
            if evt is not None:
                self._latest_frame = evt
                self._latest_frame_ts = time.time()
                self._enrich_this_turn = True
                logger.debug("vision: enrich_monologue — fresh frame grabbed for monologue turn")

        return Intent(kind="monologue")

    # ----- vision policy bridge -----
    def _decide_vision_reaction(self, vision: "VisionEvent") -> VisionDirective:
        """Run the AttentionEngine on a fresh vision event."""
        change_kind = "scene" if vision.change_type == ChangeType.SCENE_CHANGE else "delta"
        # Mid-thread protection: if the streamer just posed a question or teased
        # a story, downgrade DELTA reactions so the thread can finish.
        force_glance = bool(self._open_threads) and change_kind == "delta"

        if not self._cfg.vision.organic_vision:
            # Backward-compatible "always deep" fallback.
            return VisionDirective(
                reaction=VisionReaction.DEEP, target_sentences=3,
                rationale="organic_vision off → DEEP",
            )

        directive = self._attention.decide_on_vision(
            change_kind=change_kind,
            scene_age_sec=self._scene_memory.scene_age_sec(),
            mood_arousal=self._mood.arousal,
            mood_focus=self._mood.focus,
            mood_talkativity=self._mood.talkativity,
            vision_engagement=self._mood.wants_vision_engagement(),
            has_topic=bool(self._current_topic),
            in_monologue_flow=self._segments_in_monologue_flow() >= 2,
            segments_total=self._conv.total_segments,
            tangent_seeds=list(self._cfg.persona.running_gags or []),
            # v4: screen activity context for organic adaptation
            screen_activity=vision.activity.value,
            user_pattern=vision.user_pattern or self._behavior.current_pattern,
            user_settled=self._behavior.is_settled,
            rapid_browsing=self._behavior.is_rapid_browsing,
        )
        if force_glance and directive.reaction == VisionReaction.DEEP:
            directive = VisionDirective(
                reaction=VisionReaction.GLANCE,
                target_sentences=1,
                glance_style=directive.glance_style,
                rationale="mid-thread defer: DEEP→GLANCE so open thread can close",
            )
        return directive

    def _segments_in_monologue_flow(self) -> int:
        """Count how many of the last assistant turns were monologue-kind."""
        count = 0
        for m in reversed(self._conv.messages()):
            if m.role != "assistant":
                continue
            if m.source.startswith("monologue"):
                count += 1
            else:
                break
        return count

    def _has_urgent_pending(self) -> bool:
        return self._pop_highlight_chat_peek() is not None

    # ----- breathing gap & break system -----
    def _compute_breathing_gap(self) -> float:
        """Mood-driven variable pause between segments."""
        oc = self._cfg.orchestrator
        lo = oc.min_inter_segment_gap_sec
        hi = oc.breathing_gap_max_sec
        if hi <= lo:
            return lo
        # Low talkativity → longer pause, high → shorter.
        t = 1.0 - self._mood.talkativity
        gap = lo + (hi - lo) * t
        # Small random jitter ±20%.
        gap *= 0.8 + random.random() * 0.4
        return max(lo, min(hi, gap))

    def _schedule_next_break(self) -> None:
        oc = self._cfg.orchestrator
        if not oc.enable_breaks:
            self._next_break_at = float("inf")
            return
        base = oc.break_every_min * 60.0
        jitter = base * oc.break_every_jitter
        self._next_break_at = time.time() + base + random.uniform(-jitter, jitter)
        logger.debug(f"orchestrator: next break in {self._next_break_at - time.time():.0f}s")

    def _should_take_break(self) -> bool:
        if not self._cfg.orchestrator.enable_breaks or self._on_break:
            return False
        return time.time() >= self._next_break_at

    async def _take_break(self) -> None:
        oc = self._cfg.orchestrator
        duration = random.uniform(oc.break_min_sec, oc.break_max_sec)
        # Tired host takes longer breaks.
        duration *= 0.7 + 0.6 * (1.0 - self._mood.arousal)
        self._on_break = True
        self._break_event.clear()
        logger.info(f"orchestrator: taking a break ({duration:.1f}s)")

        deadline = time.time() + duration
        while time.time() < deadline and self._running:
            if self._has_urgent_pending():
                logger.info("orchestrator: break interrupted by urgent event")
                break
            try:
                await asyncio.wait_for(self._break_event.wait(), timeout=0.5)
                logger.info("orchestrator: break ended (manual resume)")
                break
            except asyncio.TimeoutError:
                pass

        self._on_break = False
        self._mood.on_silence_beat()
        self._schedule_next_break()
        logger.info("orchestrator: break over, resuming")

    def trigger_break(self) -> None:
        """Force an immediate break (called from dashboard)."""
        self._next_break_at = 0.0

    def resume_from_break(self) -> None:
        """End the current break early (called from dashboard)."""
        if self._on_break:
            self._break_event.set()

    def _pop_highlight_chat_peek(self) -> Optional[ChatMessage]:
        if not self._chat:
            return None
        drained: list[ChatMessage] = []
        while True:
            msg = self._chat.next_nowait()
            if msg is None:
                break
            drained.append(msg)
        highlight = next((m for m in drained if m.is_highlight), None)
        for m in drained:
            try:
                self._chat.queue.put_nowait(m)
            except asyncio.QueueFull:
                break
        return highlight

    def _pop_highlight_chat(self) -> Optional[ChatMessage]:
        if not self._chat:
            return None
        keep: list[ChatMessage] = []
        found: Optional[ChatMessage] = None
        while True:
            msg = self._chat.next_nowait()
            if msg is None:
                break
            if found is None and msg.is_highlight:
                found = msg
            else:
                keep.append(msg)
        for m in keep:
            try:
                self._chat.queue.put_nowait(m)
            except asyncio.QueueFull:
                break
        return found

    def _pop_ordinary_chat(self) -> Optional[ChatMessage]:
        if not self._chat:
            return None
        msg = self._chat.next_nowait()
        if msg is None:
            return None
        now = time.time()
        if now - self._last_chat_reply_ts < self._cfg.chat.min_reply_interval_sec:
            return None
        if random.random() >= self._cfg.chat.reply_probability:
            return None
        return msg

    def _pop_latest_vision(self) -> Optional["VisionEvent"]:
        """Drain the vision queue. Returns the most recent change event (if any)
        and updates the always-on `_latest_frame` cache so monologue turns can
        still see the screen even when there's no change to react to.

        Also updates SceneMemory so the AI knows whether this is a brand-new scene
        or a small delta within the same scene.
        """
        if self._vision_queue is None:
            return None
        latest: Optional["VisionEvent"] = None
        while True:
            try:
                latest = self._vision_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        if latest is not None:
            self._latest_frame = latest
            self._latest_frame_ts = time.time()
            # Update scene memory based on change type.
            if latest.change_type == ChangeType.SCENE_CHANGE:
                self._scene_memory.record_scene_change(None)  # hash not accessible here
            elif latest.change_type == ChangeType.DELTA:
                self._scene_memory.record_delta()
            # Feed activity info into behavior tracker for organic adaptation.
            scroll_dir = ""
            if latest.activity_detail is not None:
                scroll_dir = latest.activity_detail.scroll_direction
            self._behavior.record_activity(
                activity=latest.activity,
                pattern=latest.user_pattern,
                scroll_direction=scroll_dir,
            )
        return latest

    # ----- segment execution -----
    async def _ensure_fresh_screen_frame(self) -> None:
        """Make sure ``_latest_frame`` is no older than ``max_frame_age_sec`` before a
        monologue or vision segment runs. If it's stale, grab a fresh one synchronously.
        This guarantees that vision-mode segments always have something visual to react to
        even when the screen is static."""
        if not self._cfg.vision.enabled or self._vision_loop is None:
            return
        max_age = self._cfg.vision.max_frame_age_sec
        age = time.time() - self._latest_frame_ts if self._latest_frame_ts else 9999.0
        if self._latest_frame is not None and age <= max_age:
            return
        # Use the loop's on-demand capture path.
        evt = await self._vision_loop.grab_now()
        if evt is not None:
            self._latest_frame = evt
            self._latest_frame_ts = time.time()
            logger.info(f"vision: fresh on-demand frame grabbed ({len(evt.frame.jpeg)} bytes)")

    async def _run_segment(self, intent: Intent) -> None:
        """Run one segment as a producer/consumer pipeline.

        Producer: pulls LLM tokens, splits into sentences, applies dedupe + scrub,
        and queues final spoken sentences.
        Consumer: plays the first sentence chunk-by-chunk for low TTFA, then
        pre-fires the next sentence's TTS while the previous one is still on
        the player. This eliminates the inter-sentence silence gaps that the
        previous serial implementation produced.
        """
        # Silent beat: don't synthesise anything, just take a small breath.
        if intent.vision_directive and intent.vision_directive.reaction == VisionReaction.SILENCE:
            logger.info("orchestrator: SILENCE beat — skipping segment")
            await asyncio.sleep(0.6)
            return

        # Only vision-intent turns need a guaranteed fresh frame. Monologue
        # turns never have a frame attached — see _build_user_turn for why.
        if intent.kind == "vision":
            await self._ensure_fresh_screen_frame()

        user_msg, images, source_tag = self._build_user_turn(intent)
        if user_msg:
            self._conv.add_user(user_msg, source=source_tag, images=images)
        if intent.kind == "chat" and intent.chat is not None:
            self._last_chat_reply_ts = time.time()

        system_prompt = self._persona.system_prompt(
            topic=self._current_topic if intent.kind == "monologue" else None,
            vision_enabled=self._cfg.vision.enabled,
            session_notes=self._conv.session_notes or None,
            persistent_notes=self._memory.summary_for_prompt() if self._memory else None,
            topic_drift_style=self._cfg.topics.drift_style,
        )
        provider_msgs = self._conv.to_provider_messages(system_prompt)

        max_tok = self._cfg.llm.max_tokens
        if intent.kind == "outro":
            max_tok = min(max_tok, 220)
        elif intent.kind == "vision" and intent.vision_directive:
            # Hard token cap per reaction type — the LLM CANNOT physically
            # produce a 10-sentence rant about a notes app if we cut it off.
            _vision_tok_caps = {
                VisionReaction.GLANCE: 35,
                VisionReaction.DEEP: 70,
                VisionReaction.TANGENT: 100,
            }
            max_tok = min(max_tok, _vision_tok_caps.get(
                intent.vision_directive.reaction, 120))
        allow_repeat = intent.kind == "outro"

        streamer = SentenceStreamer()
        spoken_parts: list[str] = []
        # Sentence queue: producer fills, consumer drains. None == EOF.
        sentence_q: asyncio.Queue[Optional[str]] = asyncio.Queue(maxsize=8)

        # vision-only escape hatch: if the model decides the frame is boring
        # it outputs the literal token SKIP. Producer must detect this on the
        # first sentence and silently drop the whole segment.
        skip_eligible = intent.kind == "vision"
        skipped = False

        # Hard sentence cap for vision turns — the LLM's "soft guide" is
        # routinely ignored, so we enforce it in the pipeline. Once the cap
        # is hit the producer stops feeding sentences and signals EOF.
        vision_sentence_cap = 0
        if intent.kind == "vision" and intent.vision_directive:
            vision_sentence_cap = min(2, max(1, intent.vision_directive.target_sentences))
        produced_sentence_count = 0

        async def producer() -> None:
            nonlocal skipped, produced_sentence_count
            first_seen = False
            capped = False
            try:
                async for token in self._llm.stream(
                    provider_msgs,
                    temperature=self._cfg.llm.temperature,
                    top_p=self._cfg.llm.top_p,
                    max_tokens=max_tok,
                    presence_penalty=self._cfg.llm.presence_penalty,
                    frequency_penalty=self._cfg.llm.frequency_penalty,
                ):
                    if skipped or capped:
                        continue
                    for sent in streamer.feed(token):
                        if not first_seen and skip_eligible and _is_skip_signal(sent):
                            skipped = True
                            logger.info("vision: SKIP — frame too boring, no audio for this turn")
                            break
                        first_seen = True
                        pieces = self._prepare_sentence(sent, allow_repeat=allow_repeat)
                        for piece in pieces:
                            await sentence_q.put(piece)
                        if pieces:
                            produced_sentence_count += 1
                        if vision_sentence_cap > 0 and produced_sentence_count >= vision_sentence_cap:
                            capped = True
                            logger.info(f"vision: sentence cap reached ({produced_sentence_count}/{vision_sentence_cap})")
                            break
                    if skipped or capped:
                        continue
                if not skipped and not capped:
                    for sent in streamer.flush():
                        if not first_seen and skip_eligible and _is_skip_signal(sent):
                            skipped = True
                            logger.info("vision: SKIP — frame too boring, no audio for this turn")
                            break
                        first_seen = True
                        pieces = self._prepare_sentence(sent, allow_repeat=allow_repeat)
                        for piece in pieces:
                            await sentence_q.put(piece)
                        if pieces:
                            produced_sentence_count += 1
                        if vision_sentence_cap > 0 and produced_sentence_count >= vision_sentence_cap:
                            logger.info(f"vision: sentence cap reached on flush ({produced_sentence_count}/{vision_sentence_cap})")
                            break
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(f"segment: generation error: {e}")
            finally:
                await sentence_q.put(None)

        async def consumer() -> None:
            in_flight: list[asyncio.Task[bytes]] = []  # Pre-fired buffered TTS.
            first = True
            while True:
                sent = await sentence_q.get()
                if sent is None:
                    break
                if first:
                    first = False
                    await self._stream_sentence_direct(sent, spoken_parts)
                else:
                    # Drain one buffered task before queueing more — this preserves
                    # play order and keeps memory bounded.
                    if in_flight:
                        audio = await in_flight.pop(0)
                        await self._play_buffered(spoken_parts[-1] if spoken_parts else sent, audio)
                    in_flight.append(asyncio.create_task(self._buffer_tts(sent)))
                    spoken_parts.append(sent)
                    logger.info(f"say> {sent}")
            # Drain any remaining pre-fired tasks in order.
            for task in in_flight:
                try:
                    audio = await task
                    await self._play_buffered("", audio)
                except Exception as e:
                    logger.warning(f"tts buffered task failed: {e}")

        try:
            await asyncio.gather(producer(), consumer())
        except asyncio.CancelledError:
            raise

        full = " ".join(spoken_parts).strip()
        if full:
            self._conv.add_assistant(full, source=intent.kind)
            self._last_spoken = full
            self._last_intent_kind = intent.kind
            # Vision reactions are momentary asides — they must NOT become
            # the continuity anchor for subsequent monologue turns.
            if intent.kind != "vision":
                self._last_monologue_spoken = full
            # Update continuity trackers AFTER each spoken segment.
            self._update_open_threads(full, intent.kind)
            self._update_recent_themes(full)
            self._record_phrase_uses(full)
            self._record_question_ending(full)
            # Vision memory: record what the AI said so it can build on it later.
            if intent.kind in ("vision",):
                self._scene_memory.record_spoken(full)
                self._last_vision_turn_ts = time.time()
            # Viewer log: persist chat interactions for cross-session memory.
            if intent.kind == "chat" and intent.chat is not None and self._memory:
                self._memory.log_viewer(
                    username=intent.chat.username,
                    platform=intent.chat.platform,
                    text=intent.chat.text,
                )
            # Mood + attention bookkeeping.
            self._mood.on_segment_spoken(length_chars=len(full))
            self._attention.on_segment_spoken(intent_kind=intent.kind)

        if intent.kind == "monologue":
            if self._cfg.topics.mode == "list" and random.random() < self._cfg.topics.switch_chance:
                if self._conv.session_seconds() >= self._cfg.topics.switch_min_sec:
                    self._current_topic = self._pick_next_topic()

    # ----- sentence preparation -----
    def _prepare_sentence(self, raw: str, *, allow_repeat: bool) -> list[str]:
        """Run dedupe + scrub + long-sentence-splitting. May yield 0..N pieces."""
        if not raw or not raw.strip():
            return []
        if not allow_repeat and self._conv.is_repeat(
            raw,
            window=self._cfg.orchestrator.dedupe_window,
            threshold=self._cfg.orchestrator.dedupe_threshold,
        ):
            logger.debug(f"dedupe: skipped near-duplicate: {raw[:60]}")
            return []
        scrubbed = _scrub_unspeakable(raw)
        if not scrubbed:
            return []
        return _split_run_on(scrubbed, max_words=self._cfg.orchestrator.max_words_per_sentence)

    # ----- speech execution -----
    async def _stream_sentence_direct(self, sentence: str, spoken_parts: list[str]) -> None:
        """First sentence of a segment: stream chunk-by-chunk for the lowest TTFA."""
        spoken_parts.append(sentence)
        logger.info(f"say> {sentence}")
        await self._avatar_safe_call("trigger_emotion_from_text", sentence)
        await self._avatar_safe_call("set_speaking", True)
        first_chunk = True
        try:
            async for pcm in self._tts.synthesize(sentence):
                if first_chunk:
                    first_chunk = False
                    if _looks_non_pcm(pcm):
                        logger.error(
                            f"tts: non-PCM data detected (header={pcm[:6]!r}); aborting sentence"
                        )
                        break
                await self._player.write(pcm)
                if self._avatar:
                    await self._avatar_safe_call("set_volume", _pcm_rms(pcm))
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"tts (direct) failed: {e}")
        finally:
            # CRITICAL: drop any 1-byte alignment remainder so the next sentence
            # starts on a clean PCM sample boundary.
            self._player.boundary()
            await self._avatar_safe_call("set_speaking", False)

    async def _buffer_tts(self, sentence: str) -> bytes:
        """Used by the consumer to pre-fire a sentence's TTS in parallel."""
        await self._avatar_safe_call("trigger_emotion_from_text", sentence)
        chunks: list[bytes] = []
        first_chunk = True
        try:
            async for pcm in self._tts.synthesize(sentence):
                if first_chunk:
                    first_chunk = False
                    if _looks_non_pcm(pcm):
                        logger.error(
                            f"tts: non-PCM data detected (header={pcm[:6]!r}); aborting sentence"
                        )
                        break
                chunks.append(pcm)
        except Exception as e:
            logger.warning(f"tts (buffered) failed: {e}")
        audio = b"".join(chunks)
        # Always hand the player even-byte payloads to keep alignment safe even
        # if the TTS stream was truncated or corrupted mid-flight.
        if len(audio) & 1:
            audio = audio[:-1]
        return audio

    async def _play_buffered(self, sentence_for_log: str, audio: bytes) -> None:
        if not audio:
            return
        await self._avatar_safe_call("set_speaking", True)
        try:
            # Slice into ~40ms chunks for per-frame RMS lipsync. CHUNK is even.
            CHUNK = 1920 * 2  # ~40ms at 24 kHz mono PCM16
            for i in range(0, len(audio), CHUNK):
                piece = audio[i : i + CHUNK]
                await self._player.write(piece)
                if self._avatar:
                    await self._avatar_safe_call("set_volume", _pcm_rms(piece))
        finally:
            self._player.boundary()
            await self._avatar_safe_call("set_speaking", False)

    # ----- avatar safety wrapper -----
    async def _avatar_safe_call(self, method: str, *args: Any) -> None:
        """Avatar errors must never abort audio. Swallow everything.

        Two special-case methods:
          * ``"trigger_emotion_from_text"`` — keyword classifier on a sentence.
          * ``"trigger_emotion"`` — direct slot trigger (e.g. "hype", "thinking").

        Everything else is a passthrough to the avatar instance.
        """
        if self._avatar is None:
            return
        try:
            if method == "trigger_emotion_from_text":
                await _trigger_emotion_from_text(args[0], self._avatar, self._cfg.avatar)
                return
            fn = getattr(self._avatar, method, None)
            if fn is not None:
                await fn(*args)
        except Exception as e:
            logger.debug(f"avatar.{method} failed: {e}")

    # ----- turn assembly -----
    def _build_user_turn(self, intent: Intent) -> tuple[str, list[ImageBlock], str]:
        if intent.kind == "chat" and intent.chat is not None:
            m = intent.chat
            return (
                self._persona.chat_turn(
                    username=m.username,
                    platform=m.platform,
                    text=m.text,
                    is_highlight=m.is_highlight,
                ),
                [],
                f"chat:{m.platform}:{m.username}",
            )
        if intent.kind == "vision" and intent.vision is not None:
            # Map AttentionEngine reaction → persona vision_turn mode.
            directive = intent.vision_directive
            if directive is None:
                # Defensive default: full deep reaction.
                directive = VisionDirective(reaction=VisionReaction.DEEP, target_sentences=3)
            if directive.reaction == VisionReaction.GLANCE:
                mode = "glance"
            elif directive.reaction == VisionReaction.TANGENT:
                mode = "tangent"
            else:
                mode = intent.vision.change_type.value  # "scene" | "delta"
            return (
                self._persona.vision_turn(
                    change_type=mode,
                    last_description=self._scene_memory.last_description,
                    current_topic=self._current_topic,
                    scene_age_sec=self._scene_memory.scene_age_sec(),
                    target_sentences=directive.target_sentences,
                    glance_style=directive.glance_style,
                    tangent_seed=directive.tangent_seed,
                    mood_label=self._mood.label,
                    adaptation_hint=self._behavior.adaptation_hint(),
                    screen_activity=intent.vision.activity.value,
                ),
                [ImageBlock(data=intent.vision.frame.jpeg, mime="image/jpeg")],
                "vision",
            )
        if intent.kind == "outro":
            mins = (time.time() - self._session_start_ts) / 60.0 if self._session_start_ts else 0.0
            return (self._persona.outro_turn(minutes_streamed=mins), [], "outro")

        # Monologue. NEVER attach a raw JPEG frame here — vision-capable models
        # (Claude, GPT-4o, Gemini) will process any attached image regardless of
        # how strongly the prompt says "IGNORE the screen". The result is the
        # model drifting into a multi-minute screen-narration loop.
        #
        # enrich_monologue works at the text layer only: the orchestrator already
        # grabbed a fresh frame for the activity/adaptation hint when the
        # enrich_probability check passed; we reset the flag and let the normal
        # monologue prompt run without any image payload.
        images: list[ImageBlock] = []
        screen_attached = False
        if self._enrich_this_turn:
            # Reset flag — the grab already happened in _choose_intent.
            # We intentionally do NOT attach the JPEG.
            self._enrich_this_turn = False
            logger.debug("vision: enrich_monologue — flag reset, image intentionally not attached")

        forbidden_phrases = self._phrases_to_forbid()
        # Suppress question-style endings if the last 1-2 segments already
        # ended with one — prevents the "every line ends with a question" loop.
        suppress_question = self._segments_ended_with_question >= 1

        oc = self._cfg.orchestrator
        # Use the last MONOLOGUE segment as continuity anchor, not the last
        # vision reaction. Vision asides are momentary — they must not hijack
        # the monologue thread for the next 10 minutes.
        continuity_segment = self._last_monologue_spoken or None
        return (
            self._persona.monologue_turn(
                topic=self._current_topic,
                last_segment=continuity_segment,
                open_threads=list(self._open_threads) or None,
                recent_themes=list(self._recent_themes) or None,
                forbidden_phrases=forbidden_phrases or None,
                suppress_question=suppress_question,
                screen_attached=screen_attached,
                enrich_last_description=self._scene_memory.last_description if screen_attached else "",
                adaptation_hint=self._behavior.adaptation_hint() if screen_attached else "",
                sentences_min=oc.segment_sentences_min,
                sentences_max=oc.segment_sentences_max,
                topic_drift_style=self._cfg.topics.drift_style,
                after_vision=self._last_intent_kind == "vision",
            ),
            images,
            "monologue",
        )

    def _pick_next_topic(self) -> Optional[str]:
        if self._cfg.topics.mode == "ai_picks":
            return None
        chosen = pick_topic(self._cfg.topics.topics, self._recent_topics)
        if chosen:
            self._recent_topics.append(chosen)
            if len(self._recent_topics) > 5:
                self._recent_topics.pop(0)
        return chosen

    # ----- avatar event cues -----
    async def _cue_intent_expression(self, intent: Intent) -> None:
        """Show an avatar reaction the moment we PICK an intent — before any
        TTS audio is produced. This gives the viewer instant feedback that the
        streamer is reacting (super chat → hype face, vision change → looking)
        instead of staring blankly while the LLM warms up.
        """
        if self._avatar is None:
            return
        try:
            if intent.kind == "chat" and intent.chat is not None:
                if intent.chat.is_highlight:
                    await self._avatar_safe_call("trigger_emotion", "hype")
                # Glance at "the chat" — slight head turn left.
                await self._avatar_safe_call("look_at", -10.0, 0.0)
            elif intent.kind == "vision":
                # Look up at the "screen" briefly + a small surprised tick.
                await self._avatar_safe_call("trigger_emotion", "surprised")
                await self._avatar_safe_call("look_at", 0.0, -8.0)
            elif intent.kind == "monologue":
                # Brief thinking face right before speaking starts.
                await self._avatar_safe_call("trigger_emotion", "thinking")
        except Exception as e:
            logger.debug(f"avatar cue failed: {e}")

    # ----- avatar mood sync -----
    async def _sync_mood_to_avatar(self) -> None:
        """Push current mood state to the avatar for reactive idle behaviour."""
        if self._avatar is None:
            return
        try:
            await self._avatar.update_mood(
                self._mood.arousal,
                self._mood.valence,
                self._mood.focus,
            )
        except Exception as e:
            logger.debug(f"avatar mood sync failed: {e}")

    # ----- continuity trackers -----
    def _phrases_to_forbid(self) -> list[str]:
        """Catchphrases + running gags used within the last 5 segments. The
        next monologue turn passes these as 'do not say' to break the loop
        where the model latches onto one signature line and uses it every
        segment."""
        if not self._phrase_uses:
            return []
        cutoff = self._conv.total_segments - 5
        recent = [phrase for phrase, idx in self._phrase_uses.items() if idx > cutoff]
        return recent

    def _record_phrase_uses(self, spoken: str) -> None:
        """Scan the segment for any catchphrase or gag and stamp it with the
        current segment index. Comparison is case-insensitive substring."""
        low = spoken.lower()
        persona = self._cfg.persona
        all_phrases = list(persona.catchphrases) + list(persona.running_gags)
        for phrase in all_phrases:
            stub = phrase.lower().strip()
            if not stub:
                continue
            # Match on a stable prefix so paraphrases still register: first 4 words.
            stub_key = " ".join(stub.split()[:4])
            if stub_key and stub_key in low:
                self._phrase_uses[phrase] = self._conv.total_segments

    def _record_question_ending(self, spoken: str) -> None:
        """Increment if the segment ended on a question mark in its last
        sentence; reset otherwise. Used to throttle the question-loop pattern."""
        tail = spoken.strip()[-80:].strip()
        if "?" in tail:
            self._segments_ended_with_question += 1
        else:
            self._segments_ended_with_question = 0

    def _update_open_threads(self, spoken: str, intent_kind: str) -> None:
        """Detect questions / teases at the end of the last segment so the next
        one can be told to pay them off."""
        # 1) Anything the streamer just answered is no longer open: drop the most
        #    recent open thread when this turn was a chat or vision reaction (since
        #    those branches naturally interrupt monologue threads), or whenever
        #    the previous tease appears resolved.
        if intent_kind != "monologue" and self._open_threads:
            # Don't accumulate threads across mode switches — keep them small.
            self._open_threads = self._open_threads[-2:]

        # 2) Heuristic: a thread is "opened" if the spoken segment ends on a
        #    question OR contains a tease/setup phrase near the end.
        tail = spoken.strip()[-220:].strip()
        opened: list[str] = []
        # Question at the end?
        last_sentence = re.split(r"[\.!?]\s+", tail)
        last_sentence = last_sentence[-1] if last_sentence else tail
        if "?" in tail[-80:]:
            # Pull out the actual question.
            q_match = re.findall(r"([^\.!?]*\?)", tail)
            if q_match:
                opened.append(q_match[-1].strip())
        # Tease patterns.
        TEASE = (
            "wait until you hear", "let me tell you", "here's the kicker",
            "here is the kicker", "the funny part is", "you'll see why",
            "i'll tell you", "i will tell you", "coming back to this",
            "we'll come back", "we will come back", "spoiler",
            "watch this", "trust me", "it gets better", "it gets worse",
        )
        low = last_sentence.lower()
        if any(t in low for t in TEASE):
            opened.append(last_sentence.strip())
        for o in opened:
            if o and o not in self._open_threads:
                self._open_threads.append(o)
        # Cap.
        if len(self._open_threads) > 4:
            self._open_threads = self._open_threads[-4:]

        # 3) If THIS segment was a monologue and it neither asked nor teased,
        #    assume the prior open thread was paid off implicitly. Drop the
        #    oldest open thread so we don't keep nagging forever.
        if intent_kind == "monologue" and not opened and self._open_threads:
            self._open_threads.pop(0)

    def _update_recent_themes(self, spoken: str) -> None:
        """Tag this segment with a short theme line so the next segment knows
        what's already been covered."""
        # First sentence usually carries the theme; trim hard.
        first = re.split(r"[\.!?]\s+", spoken.strip())[0]
        words = first.split()
        if len(words) > 12:
            first = " ".join(words[:12]) + "…"
        self._recent_themes.append(first.strip())
        if len(self._recent_themes) > 8:
            self._recent_themes.pop(0)

    # ----- rolling summarizer -----
    def _maybe_kick_summarizer(self) -> None:
        self._segments_since_summary += 1
        if self._segments_since_summary < self._cfg.orchestrator.summarize_every_n:
            return
        if self._summarizer_task and not self._summarizer_task.done():
            return  # one in flight is enough
        eligible = self._conv.messages_eligible_for_summary()
        if not eligible:
            self._segments_since_summary = 0
            return
        transcript = "\n".join(
            f"[{m.source}] {m.content}" for m in eligible if m.role == "assistant"
        )
        prior = self._conv.session_notes
        self._segments_since_summary = 0
        logger.info(
            f"summary: kicking (folding {len(eligible)} old turns into running notes)"
        )
        self._summarizer_task = asyncio.create_task(
            self._run_summarizer(transcript, prior),
            name="summarizer",
        )

    async def _run_summarizer(self, transcript: str, prior_notes: str) -> None:
        try:
            prompt = self._persona.summarizer_prompt(transcript=transcript, prior_notes=prior_notes)
            tokens: list[str] = []
            async for tok in self._llm.stream(
                [
                    {"role": "system", "content": "You compress streamer transcripts into tight bullet notes. Match the language of the transcript."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                top_p=0.9,
                max_tokens=400,
                presence_penalty=0.0,
                frequency_penalty=0.0,
            ):
                tokens.append(tok)
            new_notes = "".join(tokens).strip()
            if new_notes:
                self._conv.compact_history(new_notes)
                # Mirror into MemoryStore so notes persist across sessions.
                if self._memory:
                    self._memory.update_notes(new_notes)
                preview = new_notes[:160].replace("\n", " ⤷ ")
                logger.info(
                    f"summary: notes now {len(new_notes)} chars, "
                    f"history compacted to {len(self._conv.messages())} msgs. "
                    f"preview> {preview}"
                )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"summary: failed: {e}")


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
import re
import struct

# Stage directions (e.g. *sigh*, [laughs]) get spoken literally by TTS, which
# shatters the illusion. Match conservatively so we don't eat legitimate text:
#   * "*action*"   — italic, balanced, no stars in between
#   * "[direction]" — square brackets, balanced
# Parentheses NOT matched: they often carry legitimate asides.
_STAGE_DIR = re.compile(
    r"\*[^*\n]{1,60}\*"
    r"|\[[^\]\n]{1,60}\]"
)
# Bold markdown like **word** — preserve the word, drop the markers.
_BOLD = re.compile(r"\*\*([^*\n]{1,80})\*\*")
# Leftover formatting characters that shouldn't be voiced.
_MD_CHARS = re.compile(r"[\*_`#>]")

# Emotion keyword patterns → expression slot name. Order matters — first match wins.
# Tuned to catch the spoken-streamer vocabulary (English + a few common TR markers).
_EMOTION_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Laughter — strongest, fire first.
    (re.compile(r"\b(laugh|haha+|hehe+|lmao+|rofl|giggle|chuckle|cackle|kjjk|ahaha|yha|yhaha)\b", re.I), "laughing"),
    # Hype: shouted excitement, donations, big wins.
    (re.compile(r"\b(hype|let's go|lets go|let's gooo|insane|unreal|huge|massive|on fire|absolute|crazy good|hell yes|hell yeah)\b", re.I), "hype"),
    # Surprise / shock.
    (re.compile(r"\b(wow|whoa+|oh no|omg|what the|holy|no way|surprised|wait what|are you serious|seriously\?|excuse me)\b", re.I), "surprised"),
    # Anger / frustration.
    (re.compile(r"\b(furious|pissed|annoyed|infuriating|hate this|so bad|disaster|trash|garbage|broken|terrible|awful)\b", re.I), "angry"),
    # Sadness / disappointment.
    (re.compile(r"\b(sad|depressing|tragic|breaks my heart|so bleak|gut-wrenching|devastat|sucks)\b", re.I), "sad"),
    # Eyeroll / smug — read sarcasm and dismissal.
    (re.compile(r"\b(of course|naturally|obviously|sure thing|cool cool|riiight|whatever|classic|imagine that)\b", re.I), "eyeroll"),
    # Confused / pondering.
    (re.compile(r"\b(confused|wait|hold on|let me|is that|how does|how did|why is|why are|why would|i don't get)\b", re.I), "confused"),
    # Smug — confident takes.
    (re.compile(r"\b(told you|called it|exactly|knew it|of course it|i was right)\b", re.I), "smug"),
    # Deadpan — flat reactions.
    (re.compile(r"\b(\.\.\.|huh|okay then|alright then|well that|moving on|anyway)\b", re.I), "deadpan"),
    # Generic happy — broad fallback BEFORE we give up.
    (re.compile(r"\b(love it|nice|awesome|great|amazing|beautiful|stunning|incredible|so good|that's the move)\b", re.I), "happy"),
]


def _scrub_unspeakable(text: str) -> str:
    # Order matters: unwrap bold first (preserve word), then drop stage directions.
    text = _BOLD.sub(r"\1", text)
    text = _STAGE_DIR.sub("", text)
    text = _MD_CHARS.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


# Words at which a long sentence can be cleanly broken into two.
_BREAK_AFTER = {"and", "but", "which", "so", "because", "or", "though", "while", "until"}


def _split_run_on(sentence: str, max_words: int = 22) -> list[str]:
    """Safety net for run-on sentences the model spits out as one giant
    comma-spliced thought. Splits at conjunctions only when the original
    sentence would be too long to deliver naturally in one breath.
    """
    words = sentence.split()
    if len(words) <= max_words:
        return [sentence]

    out: list[str] = []
    cur: list[str] = []
    for i, w in enumerate(words):
        cur.append(w)
        prev_had_comma = i > 0 and words[i - 1].endswith(",")
        is_break_word = w.lower().rstrip(",.!?") in _BREAK_AFTER
        long_enough = len(cur) >= 12
        # Strong break: ", and" / ", but" pattern.
        # Soft break: any comma after >= 18 words.
        if long_enough and (
            (prev_had_comma and is_break_word) or (cur[-1].endswith(",") and len(cur) >= 18)
        ):
            piece = " ".join(cur).rstrip(", ")
            if not piece.endswith((".", "!", "?")):
                piece += "."
            out.append(piece)
            cur = []
    if cur:
        rest = " ".join(cur).strip()
        if rest:
            if not rest.endswith((".", "!", "?")):
                rest += "."
            out.append(rest)
    # Final guard: if splitting produced empty pieces, fall back to original.
    out = [p for p in out if p.strip()]
    return out or [sentence]


def _is_skip_signal(text: str) -> bool:
    """True if the model returned the SKIP escape-hatch instead of a real
    sentence. Tolerant of trailing punctuation, capitalization, and a small
    leading whitespace because LLMs are sloppy about exact format compliance.
    """
    if not text:
        return False
    t = text.strip().rstrip(".!?,").strip()
    if not t:
        return False
    return t.upper() in {"SKIP", "SKIP."}


def _looks_non_pcm(chunk: bytes) -> bool:
    """Heuristic: detect when a TTS provider returned MP3/WAV/text instead of
    the PCM we asked for. Wrong format played as PCM = ear-melting static.

    We err on the side of false negatives (let real PCM through) because raw
    PCM can start with literally any byte values.
    """
    if not chunk:
        return False
    head = chunk[:6]
    # WAV containers
    if head[:4] == b"RIFF" or head[:4] == b"OggS":
        return True
    # MP3 signatures: ID3 tag or sync frame.
    if head[:3] == b"ID3":
        return True
    if len(head) >= 2 and head[0] == 0xFF and (head[1] & 0xE0) == 0xE0 and (head[1] & 0x06) != 0:
        # MP3 sync word with valid layer bits — far more likely than the same
        # bytes appearing as legitimate PCM samples.
        return True
    # JSON / HTML error page returned with status 200 by some providers.
    if head[:1] in (b"{", b"<"):
        return True
    return False


def _pcm_rms(pcm: bytes) -> float:
    """Return RMS amplitude (0-1) for a 16-bit little-endian PCM chunk."""
    if not pcm or len(pcm) < 2:
        return 0.0
    try:
        import numpy as np  # type: ignore
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        rms = float(np.sqrt(np.mean(samples ** 2)))
        return min(1.0, rms / 32768.0)
    except Exception:
        # Fallback without numpy: average of absolute sample values.
        n = len(pcm) // 2
        if n == 0:
            return 0.0
        total = sum(
            abs(struct.unpack_from("<h", pcm, i * 2)[0])
            for i in range(n)
        )
        return min(1.0, (total / n) / 32768.0)


async def _trigger_emotion_from_text(
    sentence: str,
    avatar: Any,
    avatar_cfg: Any,
) -> None:
    """Classify a sentence with the keyword table and fire the matching VTS slot.

    The classifier scans both the visible text AND any stage-direction markers
    (e.g. ``[laughs]``) so authors can hint at expressions explicitly without
    leaking them to TTS.
    """
    stage_matches = _STAGE_DIR.findall(sentence)
    combined = " ".join(stage_matches) + " " + sentence

    for pattern, slot in _EMOTION_PATTERNS:
        if pattern.search(combined):
            try:
                fn = getattr(avatar, "trigger_emotion", None)
                if callable(fn):
                    await fn(slot)
                else:
                    hotkey_name = getattr(avatar_cfg, f"expr_{slot}", "")
                    if hotkey_name:
                        await avatar.trigger_expression(hotkey_name)
            except Exception:
                pass
            return  # one expression per sentence
