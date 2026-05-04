"""Runtime configuration.

Secrets live in .env. Everything else lives in a user-editable profile YAML under
profiles/. The dashboard edits those profiles; the orchestrator rebuilds from
disk each time Start is pressed.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
PROFILES_DIR = BASE_DIR / "profiles"
STATE_FILE = BASE_DIR / ".wallie_state.json"


# -------------------------------------------------------------------
# Secrets
# -------------------------------------------------------------------
class Secrets(BaseModel):
    openai_api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    groq_api_key: str = Field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    openrouter_api_key: str = Field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    anthropic_api_key: str = Field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    gemini_api_key: str = Field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))

    fish_api_key: str = Field(default_factory=lambda: os.getenv("FISH_API_KEY", ""))
    elevenlabs_api_key: str = Field(default_factory=lambda: os.getenv("ELEVENLABS_API_KEY", ""))

    youtube_api_key: str = Field(default_factory=lambda: os.getenv("YOUTUBE_API_KEY", ""))
    youtube_client_secret_file: str = Field(
        default_factory=lambda: os.getenv("YOUTUBE_CLIENT_SECRET_FILE", "scripts/client_secret.json")
    )
    youtube_live_chat_id: str = Field(default_factory=lambda: os.getenv("YOUTUBE_LIVE_CHAT_ID", ""))

    twitch_oauth_token: str = Field(default_factory=lambda: os.getenv("TWITCH_OAUTH_TOKEN", ""))
    twitch_channel: str = Field(default_factory=lambda: os.getenv("TWITCH_CHANNEL", ""))
    twitch_nick: str = Field(default_factory=lambda: os.getenv("TWITCH_NICK", ""))

    kick_channel: str = Field(default_factory=lambda: os.getenv("KICK_CHANNEL", ""))


# -------------------------------------------------------------------
# Persona
# -------------------------------------------------------------------
Profanity = Literal["none", "mild", "heavy"]
Formality = Literal["street", "casual", "formal"]
SentenceLength = Literal["short", "medium", "mixed"]
HumorStyle = Literal[
    "ironic", "deadpan", "absurd", "observational",
    "self_deprecating", "roast", "wholesome", "chaotic",
]
Energy = Literal["chill", "warm", "hyped", "unhinged"]


class PersonaConfig(BaseModel):
    # --- Identity ------------------------------------------------------
    name: str = "Wallie"
    handle: str = "@wallie"
    # English by default. Users explicitly switch to Turkish (or any other locale)
    # when they want it. The system prompt only forces a language when the user
    # picked one.
    language: Literal["en", "tr"] = "en"
    pronouns: str = "they/them"
    age_range: str = "early 20s"
    origin: str = "somewhere online"
    # What kind of streamer are you? Shapes self-framing.
    archetype: str = "variety streamer"
    # Free-text bio that gets pasted verbatim into the system prompt.
    backstory: str = (
        "A chronically online streamer who has seen every weird corner of the internet "
        "and is mildly amused by all of it."
    )

    # --- Voice & delivery ---------------------------------------------
    energy: Energy = "warm"
    humor_style: list[HumorStyle] = Field(default_factory=lambda: ["ironic", "observational"])
    profanity: Profanity = "mild"
    formality: Formality = "casual"
    sentence_length: SentenceLength = "short"
    # Catchphrases the streamer uses naturally (not every turn, sparingly).
    catchphrases: list[str] = Field(default_factory=list)
    # Recurring bits / running gags to weave in when relevant.
    running_gags: list[str] = Field(default_factory=list)
    # Words or phrases to never say.
    banned_words: list[str] = Field(default_factory=list)
    # Free-form additional style guidance appended to the prompt.
    extra_style_notes: str = ""

    # --- Worldview ----------------------------------------------------
    strong_opinions: bool = True
    admit_uncertainty: bool = True
    break_fourth_wall: bool = False  # Can reference being on a stream
    favorite_topics: list[str] = Field(default_factory=list)
    taboo_topics: list[str] = Field(default_factory=list)

    # --- Chat behavior ------------------------------------------------
    address_style: Literal["by_name", "generic", "crowd"] = "by_name"
    reply_length: Literal["snappy", "medium", "longer"] = "snappy"
    react_to_highlights_hype: bool = True

    # --- Vision behavior ----------------------------------------------
    # First-person framing: the AI claims it opened/started whatever's on screen.
    vision_first_person: bool = True
    vision_commentary_density: Literal["sparse", "balanced", "dense"] = "balanced"

    # --- Entertainer / broadcaster framing -----------------------------
    # A real streamer doesn't just describe the screen. They perform: they
    # weave personal anecdotes, tease chat, bait reactions, tell stories
    # that orbit the content without being "about" it. These knobs dial
    # that layer up or down.
    #
    # entertainer_mode: when on, the system prompt pushes the model to
    # perform instead of narrate — short takes, punchlines, audience hooks,
    # personal beats, not summaries.
    entertainer_mode: bool = True
    # Probability per segment to drop a short audience-hook aside
    # (rhetorical question to chat, "between us, chat..." moment, etc.).
    audience_hook_rate: float = 0.30
    # Personal anecdote seeds — short reference phrases the model can weave
    # into segments. Example: ["the time my router died mid-raid",
    # "my ex naming our wifi", "the 3am cold brew incident"].
    # These become TANGENT seeds for the AttentionEngine AND personal colour
    # for monologue turns.
    anecdote_seeds: list[str] = Field(default_factory=list)
    # How often the streamer should mention their own life / memories / bits.
    # 0 = strictly on-topic, 1 = constantly self-referential.
    personal_beat_rate: float = 0.35


# -------------------------------------------------------------------
# Other subsystems
# -------------------------------------------------------------------
class LLMConfig(BaseModel):
    provider: Literal["openai", "groq", "openrouter", "anthropic", "gemini", "ollama"] = "groq"
    model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.85
    top_p: float = 0.95
    max_tokens: int = 150
    presence_penalty: float = 0.3
    frequency_penalty: float = 0.4
    vision_capable: bool = False
    # Ollama-specific — ignored by cloud providers
    ollama_base_url: str = "http://localhost:11434"
    ollama_keep_alive: str = "5m"


class TTSConfig(BaseModel):
    provider: Literal["fish", "elevenlabs", "piper"] = "fish"
    voice_id: str = ""
    sample_rate: int = 24000
    # ElevenLabs-specific knobs. Ignored by other providers.
    el_model_id: str = "eleven_turbo_v2_5"
    el_stability: float = 0.45
    el_similarity_boost: float = 0.75
    el_style: float = 0.0
    # Fish-specific: "normal" or "balanced" (balanced = lower TTFA)
    fish_latency_mode: Literal["normal", "balanced"] = "balanced"
    # Piper-specific (local, zero-cost). Path to .onnx voice model.
    # Download with: python scripts/download_piper_voice.py en_US-amy-medium
    piper_model_path: str = ""
    # 1.0 = normal speed, >1 slower, <1 faster.
    piper_length_scale: float = 1.0


class VisionConfig(BaseModel):
    enabled: bool = False
    source: Literal["monitor"] = "monitor"
    monitor_index: int = 1
    interval_sec: float = 3.0
    min_change_threshold: int = 8
    max_edge_px: int = 768
    # Hamming distance >= this → full scene change (vs. a small delta below it)
    scene_change_threshold: int = 20
    # Minimum seconds between any two emitted events (throttle).
    # NOTE: when `organic_vision` is on, the effective floor is also gated
    # by AttentionEngine so the host never monologues every tick.
    min_emit_interval_sec: float = 8.0
    # If a cached frame is older than this, grab a fresh one before using it
    max_frame_age_sec: float = 5.0
    # Image std below this → IDLE (blank screen / desktop / static)
    idle_variance_threshold: float = 15.0
    # Attach a live screen frame to monologue turns to enrich commentary.
    # DISABLED by default: vision-capable models cannot ignore an attached
    # image regardless of prompt instructions, causing them to narrate screen
    # content for minutes. Enable only with non-vision LLMs (e.g. text-only
    # Ollama models) where the frame is sent as a text OCR hint instead.
    enrich_monologue: bool = False
    # Probability [0..1] that any given monologue turn gets a screen snapshot.
    # Kept low intentionally — even with enrich_monologue=True this should
    # fire rarely so the model doesn't drift into a screen-narration loop.
    enrich_probability: float = 0.08

    # --- Organic layer -----------------------------------------------------
    # Master switch for the AttentionEngine + MoodEngine path. When off, every
    # emitted vision event triggers a full deep reaction (old behaviour).
    organic_vision: bool = True
    # 0.0 → rigid (always DEEP on scene change), 1.0 → max drift (often
    # GLANCE/IGNORE/TANGENT/SILENCE). Sweet spot around 0.65–0.85.
    organicity: float = 0.75
    # Never interrupt the streamer mid-speech for a vision event even if
    # marked urgent. Real streamers don't cut themselves off for "oh a new
    # tab opened".
    never_interrupt_speech: bool = True
    # Minimum seconds between two vision-anchored segments. Prevents the
    # streamer from machine-gunning reactions on every screen change.
    # Real streamers glance and move on — they don't narrate every click.
    min_vision_react_interval_sec: float = 18.0
    # Minimum Hamming distance for a DELTA to be *even considered* — filters
    # micro-changes (cursor movement, animated ads, small UI ticks).
    micro_change_threshold: int = 4


class ChatConfig(BaseModel):
    youtube_enabled: bool = False
    twitch_enabled: bool = False
    kick_enabled: bool = False
    reply_probability: float = 0.35
    min_reply_interval_sec: float = 8.0


class TopicConfig(BaseModel):
    mode: Literal["list", "ai_picks"] = "ai_picks"
    topics: list[str] = Field(default_factory=lambda: [
        "Artificial intelligence and the future",
        "Strange decisions from tech companies",
        "Absurd observations from everyday life",
    ])
    # Seconds between possible topic switches (in list mode).
    switch_min_sec: float = 90.0
    switch_chance: float = 0.15
    # How topic transitions happen:
    #   rigid   — hard anchor, explicit bridge phrases required
    #   natural — association-based drift, thoughts lead to adjacent topics
    #   freeform — stream of consciousness, no anchoring at all
    drift_style: Literal["rigid", "natural", "freeform"] = "natural"


class OrchestratorConfig(BaseModel):
    segment_target_sec: float = 12.0
    dedupe_window: int = 8
    dedupe_threshold: float = 0.78
    prebuffer: bool = True
    # Safety net: any sentence longer than this is split at the nearest
    # natural break before being sent to TTS. Most LLMs will produce 80-word
    # comma-spliced run-ons under "Be conversational"; this catches them.
    max_words_per_sentence: int = 22

    # --- Monologue segment length -------------------------------------------
    # How many sentences the LLM should aim for per monologue segment.
    # Longer segments reduce inter-segment gaps and sound more natural.
    # Short (3-5) is choppy, medium (5-10) flows well, long (8-14) for deep dives.
    segment_sentences_min: int = 3
    segment_sentences_max: int = 6
    # How much audio (seconds) can be queued before we hold off starting the
    # next LLM call. Higher = smoother (LLM has more lead time), but also
    # higher interrupt latency. 0 = old behavior (wait for full drain).
    max_audio_lookahead_sec: float = 8.0

    # --- Session length -----------------------------------------------
    # 0 = unlimited (run until manual stop). Otherwise, after this many minutes
    # the orchestrator triggers a natural outro and stops.
    session_duration_min: float = 0.0
    # When time-remaining drops below this many seconds, the next segment is
    # an outro instead of more monologue.
    outro_seconds: float = 30.0

    # --- Long-session memory ------------------------------------------
    # Keep the last N assistant turns verbatim in the conversation. Older
    # turns get folded into a rolling summary.
    recent_verbatim_turns: int = 24
    # Run the rolling summarizer every N delivered segments.
    summarize_every_n: int = 14
    # Soft caps on the working conversation. The summarizer keeps it under
    # these even on very long streams.
    max_messages: int = 200
    max_chars: int = 60000

    # --- Organic pacing ---------------------------------------------------
    # Master switch for MoodEngine + AttentionEngine. When off, behaves like
    # v1: no mood drift, every vision event becomes a full deep reaction.
    organic_enabled: bool = True
    # When the mood model says "hold a silent beat", how long to wait before
    # the next turn (seconds). Real streamers often go quiet for 2-6s while
    # thinking — this prevents non-stop yapping.
    silence_beat_min_sec: float = 2.0
    silence_beat_max_sec: float = 5.5
    # Maximum probability ceiling for any silence beat (even a very tired
    # mood won't exceed this). Prevents dead air on a live stream.
    silence_beat_ceiling: float = 0.35
    # Tiny minimum gap between segments even when the host is hype —
    # gives TTS time to fully drain and the viewer time to register the
    # previous line. 0 disables.
    min_inter_segment_gap_sec: float = 0.35
    # Mood-driven breathing gap ceiling. The actual pause between segments
    # ranges from min_inter_segment_gap_sec to this value, scaled by mood
    # (low talkativity → longer pauses). Set equal to min for fixed gaps.
    breathing_gap_max_sec: float = 2.5

    # --- Periodic breaks --------------------------------------------------
    # Real streamers take natural pauses — sip water, stretch, etc.
    # This creates periodic breaks that interrupt the monologue loop.
    enable_breaks: bool = True
    # Average time between breaks (minutes). Jittered ± break_every_jitter.
    break_every_min: float = 8.0
    break_every_jitter: float = 0.35   # ±35% → roughly 5–11 min
    # Duration range per break (seconds). Actual duration also scales with
    # mood arousal: a tired host takes longer breaks.
    break_min_sec: float = 4.0
    break_max_sec: float = 12.0


class AvatarConfig(BaseModel):
    """VTube Studio integration. Drives a Live2D avatar from the speech pipeline.

    Five layers:
      * lipsync       — mouth animation from PCM amplitude, with attack/release.
      * idle motion   — subtle head sway + occasional eye darts when silent.
      * blink         — periodic natural eye blinks with double-blink variation.
      * body motion   — slow torso sway independent of head movement.
      * expressions   — VTS hotkeys triggered by sentence content or events.

    Mood-reactive: arousal/valence/focus from the MoodEngine modulate idle
    amplitude, eye-dart frequency, blink rate, brow position, and resting smile.
    Expression auto-mapping matches VTS hotkeys to emotion slots on connect.
    """

    enabled: bool = False
    vts_host: str = "127.0.0.1"
    vts_port: int = 8001

    # --- VTS parameter names (must match the model's parameter IDs) -----
    param_mouth_open:  str = "MouthOpen"
    param_mouth_smile: str = "MouthSmile"
    param_face_x:      str = "FaceAngleX"
    param_face_y:      str = "FaceAngleY"
    param_face_z:      str = "FaceAngleZ"
    param_eye_x:       str = "EyeLeftX"
    param_eye_y:       str = "EyeLeftY"
    param_brows:       str = "Brows"

    # --- Lipsync envelope ----------------------------------------------
    # Multiplier applied to RMS before clamping. Most TTS audio sits 0.05-0.2 RMS.
    lipsync_gain:    float = 4.0
    # 0..1 ceiling on MouthOpen — 0.85 looks natural, 1.0 is cartoonish.
    lipsync_ceiling: float = 0.85
    # Below this RMS the mouth is forced closed (kills jitter from background hiss).
    lipsync_floor:   float = 0.02
    # Attack/release smoothing, 0..1 (higher = snappier).
    lipsync_attack:  float = 0.65
    lipsync_release: float = 0.30
    # Smile floor while speaking (so the mouth isn't dead-flat). 0 disables.
    speaking_smile:  float = 0.15

    # --- Idle motion ---------------------------------------------------
    # When NOT speaking, drift FaceAngleX/Y on a slow Perlin-like noise so the
    # avatar feels alive instead of frozen.
    enable_idle_motion: bool = True
    idle_sway_amplitude: float = 4.0   # degrees on FaceAngleX/Y
    idle_sway_period_sec: float = 6.0  # one cycle every N seconds
    # Occasional small saccades (eye darts) every few seconds.
    enable_eye_darts: bool = True
    eye_dart_interval_sec: float = 4.5

    # --- Expressions (VTS hotkey IDs or names) -------------------------
    # Filled per-model; the dashboard can populate these from the discover API.
    expr_happy:      str = ""
    expr_surprised:  str = ""
    expr_laughing:   str = ""
    expr_angry:      str = ""
    expr_sad:        str = ""
    expr_thinking:   str = ""
    expr_smug:       str = ""
    expr_eyeroll:    str = ""
    expr_confused:   str = ""
    expr_hype:       str = ""    # super chats, big chat moments
    expr_deadpan:    str = ""    # the "..." reaction

    # --- Blink -----------------------------------------------------------
    enable_blink:          bool = True
    param_eye_open_left:   str = "EyeOpenLeft"
    param_eye_open_right:  str = "EyeOpenRight"
    blink_interval_sec:    float = 3.8    # avg seconds between blinks
    blink_hold_sec:        float = 0.045  # eyes-closed duration
    double_blink_chance:   float = 0.15   # chance of a rapid double-blink

    # --- Body motion -----------------------------------------------------
    enable_body_motion:    bool = True
    param_body_x:          str = "BodyAngleX"
    param_body_y:          str = "BodyAngleY"
    param_body_z:          str = "BodyAngleZ"
    body_sway_amplitude:   float = 2.5    # degrees
    body_sway_period_sec:  float = 9.0    # one full cycle

    # --- Mood-reactive behaviour -----------------------------------------
    # Links MoodEngine state to avatar idle animation, brow, and smile.
    enable_mood_link:      bool = True
    # Arousal 0..1 maps idle amplitude to this range (multiplier).
    mood_idle_min_scale:   float = 0.5
    mood_idle_max_scale:   float = 1.6
    # Valence -1..1 maps brow offset to this range.
    mood_brow_min:         float = -0.4
    mood_brow_max:         float = 0.3
    # Max resting smile when valence is positive (0 disables).
    mood_smile_max:        float = 0.20

    # --- Expression auto-mapping -----------------------------------------
    # On connect, try to match VTS hotkey names to empty expr_* slots.
    auto_map_expressions:  bool = True


class AppConfig(BaseModel):
    # Display name of this profile (shown in dashboard profile picker).
    profile_name: str = "default"
    persona: PersonaConfig = Field(default_factory=PersonaConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    vision: VisionConfig = Field(default_factory=VisionConfig)
    chat: ChatConfig = Field(default_factory=ChatConfig)
    topics: TopicConfig = Field(default_factory=TopicConfig)
    orchestrator: OrchestratorConfig = Field(default_factory=OrchestratorConfig)
    avatar: AvatarConfig = Field(default_factory=AvatarConfig)


# -------------------------------------------------------------------
# Profiles (multi-persona support)
# -------------------------------------------------------------------
@dataclass
class Runtime:
    config: AppConfig
    secrets: Secrets
    base_dir: Path = BASE_DIR


def _ensure_dirs() -> None:
    PROFILES_DIR.mkdir(exist_ok=True)


def _active_profile_name() -> str:
    _ensure_dirs()
    if STATE_FILE.exists():
        try:
            import json
            return json.loads(STATE_FILE.read_text(encoding="utf-8")).get("active", "default")
        except Exception:
            pass
    return "default"


def _set_active_profile_name(name: str) -> None:
    import json
    STATE_FILE.write_text(json.dumps({"active": name}), encoding="utf-8")


def _profile_path(name: str) -> Path:
    safe = "".join(c for c in name if c.isalnum() or c in "-_") or "default"
    return PROFILES_DIR / f"{safe}.yaml"


def list_profiles() -> list[str]:
    _ensure_dirs()
    return sorted(p.stem for p in PROFILES_DIR.glob("*.yaml"))


def load_profile(name: Optional[str] = None) -> AppConfig:
    name = name or _active_profile_name()
    path = _profile_path(name)
    if not path.exists():
        cfg = AppConfig(profile_name=name)
        save_profile(cfg, name)
        _set_active_profile_name(name)
        return cfg
    try:
        import yaml  # type: ignore
        data: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        data["profile_name"] = name
        return AppConfig(**data)
    except ModuleNotFoundError:
        import json
        data = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        data["profile_name"] = name
        return AppConfig(**data)


def save_profile(cfg: AppConfig, name: Optional[str] = None) -> None:
    _ensure_dirs()
    name = name or cfg.profile_name or "default"
    cfg = cfg.model_copy(update={"profile_name": name})
    path = _profile_path(name)
    data = cfg.model_dump()
    try:
        import yaml  # type: ignore
        path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    except ModuleNotFoundError:
        import json
        path.with_suffix(".json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def activate_profile(name: str) -> AppConfig:
    cfg = load_profile(name)
    _set_active_profile_name(name)
    return cfg


def delete_profile(name: str) -> bool:
    path = _profile_path(name)
    if path.exists():
        path.unlink()
        if _active_profile_name() == name:
            remaining = list_profiles()
            _set_active_profile_name(remaining[0] if remaining else "default")
        return True
    return False


def clone_profile(src: str, dst: str) -> AppConfig:
    cfg = load_profile(src)
    save_profile(cfg, dst)
    return load_profile(dst)


def get_runtime() -> Runtime:
    return Runtime(config=load_profile(), secrets=Secrets())


# --- backward-compatible aliases (older code paths may still call these) ---
def load_config(*_args, **_kwargs) -> AppConfig:
    return load_profile()


def save_config(cfg: AppConfig, *_args, **_kwargs) -> None:
    save_profile(cfg)
