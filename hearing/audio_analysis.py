"""Dependency-free MUSIC understanding for Wallie's ear.

Pure numpy — no librosa, no ML model, zero install weight — but it pulls real
perceptual features out of a window so Wallie reads music the way a listener does:

  • tonality (major/minor) → emotional VALENCE  (the "I don't wanna cry" instinct)
  • tempo + pulse strength  → AROUSAL + whether there's a real beat or it's rubato
  • spectral texture (bands)→ instrumentation feel (bass-heavy / acoustic / dense / airy / lo-fi)
  • dynamics over time      → building / steady / sparse / fading
  • harmonic richness       → lush vs minimal

These collapse into a short, COHERENT, emotionally-loaded descriptor like
"slow, melancholic, minor-key, acoustic" or "fast, euphoric, major-key, dense"
so Wallie reacts to how the music FEELS, not just that "music is playing".

For exact instrument/event tagging (a specific guitar, applause, a gunshot) a
dedicated audio model (YAMNet / PANNs) is the next step — this is the strong,
no-dependency baseline.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MusicFeatures:
    """A musical window distilled into one descriptor PLUS the numbers that let it
    move Wallie's mood (not just the prompt). valence/arousal/energy/brightness are
    the perceptual axes the MoodEngine reads."""
    descriptor: str          # "slow, melancholic, minor-key, acoustic"
    valence: float           # -1..1  (minor/dark/harsh = negative, major/bright = positive)
    arousal: float           # 0..1   (tempo + pulse + loudness + brightness)
    energy: float            # 0..1   (loudness / RMS)
    brightness: float        # 0..1   (spectral centroid)
    mode: str                # "major"|"minor"|"neutral"
    clarity: float           # 0..1   how tonal / confident the key read is
    bpm: float
    pulse: float             # 0..1   beat strength

_MAJOR_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
_MINOR_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)


def _stft_mag(audio: "np.ndarray", n_fft: int = 2048, hop: int = 512) -> "np.ndarray":
    """Magnitude STFT, shape [frames, bins]. Cheap, vectorized."""
    if audio.size < n_fft:
        audio = np.pad(audio, (0, n_fft - audio.size))
    n_frames = 1 + (audio.size - n_fft) // hop
    if n_frames < 1:
        return np.abs(np.fft.rfft(audio[:n_fft] * np.hanning(n_fft)))[None, :]
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = audio[idx] * np.hanning(n_fft)[None, :]
    return np.abs(np.fft.rfft(frames, axis=1))


def _chroma(avg_spec: "np.ndarray", freqs: "np.ndarray") -> "np.ndarray":
    """12-bin pitch-class energy (C..B), normalized to sum 1."""
    band = (freqs >= 55.0) & (freqs <= 4000.0)
    f = freqs[band]
    m = avg_spec[band]
    if f.size == 0 or m.sum() <= 0:
        return np.zeros(12)
    midi = 69.0 + 12.0 * np.log2(np.maximum(f, 1e-6) / 440.0)
    pc = np.mod(np.round(midi).astype(int), 12)
    chroma = np.zeros(12)
    np.add.at(chroma, pc, m)
    s = chroma.sum()
    return chroma / s if s > 0 else chroma


def _estimate_tonality(chroma: "np.ndarray") -> tuple[str, float]:
    """Return (mode, clarity). mode='major'|'minor', clarity 0..1 = how tonal it is."""
    if not np.any(chroma):
        return "neutral", 0.0
    best_r, best_mode = -2.0, "neutral"
    c0 = chroma - chroma.mean()
    cnorm = np.sqrt((c0 ** 2).sum())
    if cnorm <= 0:
        return "neutral", 0.0
    for shift in range(12):
        cs = np.roll(chroma, -shift)
        csc = cs - cs.mean()
        cd = np.sqrt((csc ** 2).sum())
        for prof, mode in ((_MAJOR_PROFILE, "major"), (_MINOR_PROFILE, "minor")):
            p = prof - prof.mean()
            denom = cd * np.sqrt((p ** 2).sum())
            if denom <= 0:
                continue
            r = float((csc * p).sum() / denom)
            if r > best_r:
                best_r, best_mode = r, mode
    return best_mode, max(0.0, min(1.0, best_r))


def _tempo_pulse(stft: "np.ndarray", sr: int, hop: int) -> tuple[float, float]:
    """(bpm, pulse) from the spectral-flux onset envelope. pulse 0..1 = beat strength."""
    if stft.shape[0] < 8:
        return 0.0, 0.0
    flux = np.maximum(0.0, np.diff(stft, axis=0)).sum(axis=1)
    flux = flux - flux.mean()
    if not np.any(flux):
        return 0.0, 0.0
    fps = sr / hop
    ac = np.correlate(flux, flux, mode="full")[flux.size - 1:]
    if ac.size == 0 or ac[0] <= 0:
        return 0.0, 0.0
    lo = max(1, int(fps * 60.0 / 200.0))  # up to 200 BPM
    hi = min(ac.size - 1, int(fps * 60.0 / 50.0))  # down to 50 BPM
    if hi <= lo:
        return 0.0, 0.0
    seg = ac[lo:hi + 1]
    k = lo + int(np.argmax(seg))
    bpm = 60.0 * fps / k if k > 0 else 0.0
    pulse = float(seg.max() / ac[0])
    return bpm, max(0.0, min(1.0, pulse))


def _band_energy(avg_spec: "np.ndarray", freqs: "np.ndarray") -> dict:
    """Normalized energy per perceptual band → texture cues."""
    edges = [(20, 60), (60, 250), (250, 500), (500, 2000), (2000, 6000), (6000, 20000)]
    keys = ["sub", "bass", "lowmid", "mid", "high", "air"]
    p = avg_spec ** 2
    total = p.sum() + 1e-12
    out = {}
    for (lo, hi), k in zip(edges, keys):
        out[k] = float(p[(freqs >= lo) & (freqs < hi)].sum() / total)
    return out


def _dynamics(audio: "np.ndarray") -> str:
    """Loudness shape across the window: building / fading / dynamic / steady / sparse."""
    seg = 8
    L = audio.size // seg
    if L < 1:
        return "steady"
    r = np.array([np.sqrt(np.mean(audio[i * L:(i + 1) * L] ** 2)) for i in range(seg)])
    if r.max() <= 0:
        return "sparse"
    trend = r[-3:].mean() - r[:3].mean()
    spread = (r.max() - r.min()) / (r.max() + 1e-9)
    if trend > 0.4 * r.max():
        return "building"
    if trend < -0.4 * r.max():
        return "fading"
    if spread > 0.7:
        return "dynamic"
    if r.mean() < 0.03:
        return "sparse"
    return "steady"


# ----------------------------------------------------------------------------- mood
def _mood_word(mode: str, clarity: float, bpm: float, rms: float, centroid: float) -> str:
    """Map (valence, arousal, brightness) to one emotionally-loaded word."""
    tonal = clarity >= 0.30
    pos = tonal and mode == "major"
    neg = tonal and mode == "minor"
    high = bpm >= 110 or rms > 0.12
    low = (bpm <= 80 or bpm == 0) and rms < 0.07
    if neg and high:
        return "intense" if centroid > 2200 else "brooding"
    if neg and low:
        return "melancholic" if centroid >= 1400 else "somber"
    if neg:
        return "moody"
    if pos and high:
        return "euphoric" if centroid > 2600 else "triumphant"
    if pos and low:
        return "tender" if centroid >= 1500 else "warm"
    if pos:
        return "upbeat"
    if high:
        return "driving"
    if low:
        return "dreamy" if centroid >= 1800 else "mellow"
    return "atmospheric"


def _production_word(bands: dict, flatness: float, peak: float, pulse: float) -> str:
    """One word about the MIX / production quality — gives Wallie something to judge
    ('this is crisp' / 'kinda muddy' / 'harsh'), the raw material for a good/bad take."""
    low = bands["sub"] + bands["bass"]
    top = bands["high"] + bands["air"]
    if peak >= 0.985 and flatness > 0.20:
        return "harsh"          # likely clipping / blown out
    if low + bands["lowmid"] > 0.70 and top < 0.12:
        return "muddy"
    if low < 0.10 and bands["sub"] < 0.04:
        return "thin"
    if pulse > 0.40 and low > 0.30:
        return "punchy"
    if top > 0.48 and flatness < 0.30:
        return "crisp"
    return ""


def _texture_word(bands: dict, pulse: float, clarity: float, flatness: float) -> str:
    """One word for the instrumentation/production feel."""
    if flatness > 0.45:
        return "lo-fi"
    low_end = bands["sub"] + bands["bass"]
    top = bands["high"] + bands["air"]
    if low_end > 0.55 and pulse > 0.25:
        return "bass-heavy"
    if clarity > 0.45 and low_end < 0.4 and pulse < 0.35:
        return "acoustic"
    if top > 0.45:
        return "airy"
    if bands["mid"] + bands["lowmid"] > 0.6 and pulse > 0.3:
        return "dense"
    return "full"


def _valence_arousal(mode: str, clarity: float, bpm: float, pulse: float,
                     rms: float, centroid: float, production: str,
                     dyn: str) -> tuple[float, float, float, float]:
    """Map raw DSP cues → (valence -1..1, arousal 0..1, energy 0..1, brightness 0..1).
    This is what lets music actually shift Wallie's mood instead of just the prompt."""
    brightness = float(np.clip(centroid / 3800.0, 0.0, 1.0))
    energy = float(np.clip(rms / 0.18, 0.0, 1.0))
    tonal = float(np.clip((clarity - 0.28) / 0.45, 0.0, 1.0))

    valence = 0.0
    if mode == "major":
        valence += 0.62 * tonal
    elif mode == "minor":
        valence -= 0.62 * tonal
    valence += (brightness - 0.5) * 0.32           # brighter timbre reads happier
    if production in ("harsh", "muddy"):
        valence -= 0.12                            # a bad mix sours the vibe
    if dyn == "building":
        valence += 0.08
    elif dyn == "fading":
        valence -= 0.06
    valence = float(np.clip(valence, -1.0, 1.0))

    has_beat = pulse > 0.15 and bpm > 0
    arousal = 0.22
    arousal += energy * 0.42
    arousal += pulse * 0.26
    if has_beat:
        arousal += float(np.clip(bpm / 170.0, 0.0, 1.0)) * 0.22
    arousal += brightness * 0.14
    if dyn == "building":
        arousal += 0.08
    arousal = float(np.clip(arousal, 0.0, 1.0))
    return valence, arousal, energy, brightness


def analyze_music(audio: "np.ndarray", sr: int = 16000) -> "MusicFeatures | None":
    """Full musical read of a window: descriptor + the numeric mood axes. None if
    there's nothing musical to describe."""
    if audio is None or audio.size < sr // 2:
        return None
    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms <= 0:
        return None
    peak = float(np.abs(audio).max())

    hop = 512
    stft = _stft_mag(audio, n_fft=2048, hop=hop)
    avg_spec = stft.mean(axis=0) + 1e-9
    freqs = np.fft.rfftfreq(2048, 1.0 / sr)

    centroid = float(np.sum(freqs * avg_spec) / np.sum(avg_spec))
    flatness = float(np.exp(np.mean(np.log(avg_spec))) / np.mean(avg_spec))
    chroma = _chroma(avg_spec, freqs)
    mode, clarity = _estimate_tonality(chroma)
    bpm, pulse = _tempo_pulse(stft, sr, hop)
    bands = _band_energy(avg_spec, freqs)
    dyn = _dynamics(audio)
    production = _production_word(bands, flatness, peak, pulse)

    parts: list[str] = []
    if pulse > 0.18:
        if bpm >= 120:
            parts.append("fast")
        elif 0 < bpm <= 80:
            parts.append("slow")
    elif bpm and bpm <= 75:
        parts.append("slow")
    parts.append(_mood_word(mode, clarity, bpm, rms, centroid))
    if clarity >= 0.32 and mode in ("major", "minor"):
        parts.append(f"{mode}-key")
    parts.append(_texture_word(bands, pulse, clarity, flatness))
    parts.append(production)
    if dyn in ("building", "fading", "dynamic", "sparse"):
        parts.append(dyn)

    seen, out = set(), []
    for p in parts:
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    descriptor = ", ".join(out[:5])

    valence, arousal, energy, brightness = _valence_arousal(
        mode, clarity, bpm, pulse, rms, centroid, production, dyn)
    return MusicFeatures(descriptor=descriptor, valence=valence, arousal=arousal,
                         energy=energy, brightness=brightness, mode=mode,
                         clarity=clarity, bpm=bpm, pulse=pulse)


def music_character(audio: "np.ndarray", sr: int = 16000) -> str:
    """Backward-compatible text descriptor (e.g. 'slow, melancholic, minor-key,
    acoustic'). Empty string if there's nothing musical to describe."""
    f = analyze_music(audio, sr)
    return f.descriptor if f else ""


def analyze_window(audio: "np.ndarray", sr: int = 16000, *,
                   has_speech: bool = False, silence: float = 0.005) -> tuple[str, str]:
    """Return (sound_type, descriptor).

    sound_type: "quiet" | "speech" | "music" | "sound".
    descriptor: rich phrase for music (e.g. "slow, melancholic, minor-key, acoustic").
    """
    if audio is None or audio.size == 0:
        return "quiet", ""
    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms < silence:
        return "quiet", ""
    if has_speech:
        return "speech", ""

    w = audio * np.hanning(len(audio))
    spec = np.abs(np.fft.rfft(w)) + 1e-9
    freqs = np.fft.rfftfreq(len(audio), 1.0 / sr)
    flatness = float(np.exp(np.mean(np.log(spec))) / np.mean(spec))

    chroma = _chroma(spec, freqs)
    chroma_peak = float(chroma.max() / (chroma.mean() + 1e-9)) if chroma.sum() > 0 else 0.0
    _mode, clarity = _estimate_tonality(chroma)
    is_music = flatness < 0.55 or chroma_peak > 2.4 or clarity > 0.55

    if is_music:
        return "music", music_character(audio, sr) or "steady"
    if rms > 0.15:
        return "sound", "loud, sudden"
    return "sound", "ambient"
