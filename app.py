#!/usr/bin/env python3
"""
Mithril Meditation Audio Editor
- Noise reduction for professional sound
- Auto-detects "let's practice" and adds gentle nature sounds
"""

import os
import sys
import json
import uuid
import threading
import tempfile
import subprocess
import webbrowser

import numpy as np
import scipy.signal as signal
from flask import Flask, request, send_file, jsonify

# ── ffmpeg setup (bundled, no system install needed) ────────────────────────
import imageio_ffmpeg
from pydub import AudioSegment
from pydub.effects import normalize

_FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
AudioSegment.converter = _FFMPEG
os.environ["PATH"] = os.path.dirname(_FFMPEG) + os.pathsep + os.environ.get("PATH", "")


# ── Sound library ────────────────────────────────────────────────────────────
LIBRARY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nature_library")
os.makedirs(LIBRARY_DIR, exist_ok=True)

# ── Presets (ship with the app — real nature recordings) ─────────────────────
PRESETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "presets")
# Maps nature_type value → filename in PRESETS_DIR
PRESET_FILES = {
    "British Woods":  "british_woods.mp3",
    "Spring Rain":    "spring_rain.mp3",
    "Sandy Beach":    "sandy_beach.mp3",
}

# ── In-memory job store ──────────────────────────────────────────────────────
_jobs = {}   # job_id -> {status, log, output_path}


# ── Nature sound generator ───────────────────────────────────────────────────

def _make_nature(duration_ms: int, sound_type: str) -> AudioSegment:
    sr = 44100
    n = int(sr * duration_ms / 1000)
    t = np.linspace(0, duration_ms / 1000, n)
    rng = np.random.default_rng(42)

    if sound_type == "Rain":
        noise = rng.standard_normal(n)
        sos = signal.butter(4, [400, 7000], btype="bandpass", fs=sr, output="sos")
        rain = signal.sosfilt(sos, noise)
        sos_low = signal.butter(4, 150, btype="lowpass", fs=sr, output="sos")
        rumble = signal.sosfilt(sos_low, rng.standard_normal(n)) * 0.25
        audio = (rain + rumble) * 0.18

    elif sound_type == "Forest stream":
        noise = rng.standard_normal(n)
        sos = signal.butter(4, [180, 5000], btype="bandpass", fs=sr, output="sos")
        stream = signal.sosfilt(sos, noise)
        mod = 0.72 + 0.18 * np.sin(2 * np.pi * 0.6 * t) + 0.10 * np.sin(2 * np.pi * 1.9 * t)
        audio = stream * mod * 0.15

    elif sound_type == "Birds only":
        # Frequency-glide notes with Hanning envelope.
        # Hanning starts and ends at exactly 0 → zero clicks, zero ticking.
        # Timing is randomised (Poisson-like gaps, 3.5–10.5 s).
        rng_b = np.random.default_rng(42)

        def _glide_note(f_start, f_end, dur_s):
            length = int(dur_s * sr)
            if length < 4:
                return np.zeros(0)
            inst_freq = np.linspace(f_start, f_end, length)
            phase = np.cumsum(2 * np.pi * inst_freq / sr)
            return np.hanning(length) * np.sin(phase)

        def _make_call():
            amp = rng_b.uniform(0.015, 0.026)
            style = rng_b.integers(5)
            if style == 0:          # single rising whistle
                f = rng_b.uniform(2400, 3000)
                return [(f, f * rng_b.uniform(1.06, 1.20), rng_b.uniform(0.10, 0.18), amp)]
            elif style == 1:        # single falling whistle
                f = rng_b.uniform(2800, 3500)
                return [(f, f * rng_b.uniform(0.82, 0.93), rng_b.uniform(0.10, 0.18), amp)]
            elif style == 2:        # two-note ascending call
                f1 = rng_b.uniform(2200, 2800)
                f2 = f1 * rng_b.uniform(1.12, 1.28)
                return [(f1, f1, rng_b.uniform(0.09, 0.14), amp),
                        (f2, f2 * rng_b.uniform(0.96, 1.05), rng_b.uniform(0.09, 0.15), amp * 0.9)]
            elif style == 3:        # two-note descending call
                f1 = rng_b.uniform(2900, 3400)
                f2 = f1 * rng_b.uniform(0.76, 0.88)
                return [(f1, f1 * rng_b.uniform(0.97, 1.03), rng_b.uniform(0.08, 0.13), amp),
                        (f2, f2, rng_b.uniform(0.10, 0.16), amp * 0.85)]
            else:                   # short high ping
                f = rng_b.uniform(2900, 3700)
                return [(f, f * rng_b.uniform(1.01, 1.07), rng_b.uniform(0.05, 0.09), amp * 0.75)]

        birds = np.zeros(n)
        pos = int(rng_b.uniform(2, 5) * sr)

        while pos < int(0.95 * n):
            for f_start, f_end, dur_s, amp in _make_call():
                note = _glide_note(f_start, f_end, dur_s)
                end = pos + len(note)
                if end < n:
                    birds[pos:end] += note * amp
                pos = end + int(rng_b.uniform(0.05, 0.14) * sr)
            # 30% more frequent than before: 3.5–10.5 s between calls
            pos += int(rng_b.uniform(3.5, 10.5) * sr)

        audio = birds

    elif sound_type == "Forest birds":
        noise = rng.standard_normal(n)
        sos = signal.butter(4, [80, 3500], btype="bandpass", fs=sr, output="sos")
        wind = signal.sosfilt(sos, noise) * 0.10
        birds = np.zeros(n)
        chirps = [(2400, 0.08, 0.06), (2900, 0.22, 0.05), (2200, 0.41, 0.07),
                  (3100, 0.63, 0.04), (2600, 0.79, 0.06), (2350, 0.91, 0.05)]
        for freq, rel_pos, dur in chirps:
            start = int(rel_pos * n)
            length = int(dur * sr)
            if start + length < n:
                env = np.hanning(length)
                sweep = np.linspace(freq, freq * 1.04, length)
                wave = env * np.sin(2 * np.pi * np.cumsum(sweep) / sr) * 0.022
                birds[start : start + length] += wave
        audio = wind + birds

    else:  # Birds & stream
        # ── Stream: stochastic (random) amplitude modulation ──
        # Using filtered noise as the modulator instead of sine waves removes
        # all rhythmic pulsing. Only 3 bands in the gurgling mid range of a
        # small brook — no high-freq hiss band.
        audio = np.zeros(n)
        rng2 = np.random.default_rng(77)
        layers = [
            # (low_hz, high_hz, amp, mod_lowpass_hz)
            (80,   380,  0.32, 0.30),   # low body / depth
            (380,  1000, 0.60, 0.90),   # main gurgle
            (1000, 3000, 0.58, 1.80),   # bright gurgle / ripple
        ]
        for lo, hi, amp, mod_hz in layers:
            carrier = rng2.standard_normal(n)
            sos_c = signal.butter(4, [lo, hi], btype="bandpass", fs=sr, output="sos")
            carrier = signal.sosfilt(sos_c, carrier)
            # Modulator is slow filtered noise → random, non-rhythmic swell
            mod_raw = rng2.standard_normal(n)
            sos_m = signal.butter(2, mod_hz, btype="lowpass", fs=sr, output="sos")
            mod = signal.sosfiltfilt(sos_m, mod_raw)
            lo_m, hi_m = mod.min(), mod.max()
            mod = (mod - lo_m) / (hi_m - lo_m + 1e-10)  # 0→1
            mod = 0.30 + 0.70 * mod                       # 0.3→1.0 (never fully silent)
            audio = audio + carrier * mod * amp
        audio = audio * 0.050

        # ── Birds: FM synthesis with NaN-safe envelope ──
        def _bird_note(f_c, f_mr, f_md, dur_s):
            length = int(dur_s * sr)
            tc = np.arange(length) / sr
            inst_freq = f_c + f_md * np.sin(2 * np.pi * f_mr * tc)
            phase = np.cumsum(2 * np.pi * inst_freq / sr)
            # np.maximum avoids negative-zero ** fractional = NaN
            env = np.maximum(np.sin(np.pi * tc / tc[-1]), 0.0) ** 0.7
            return env * np.sin(phase)

        birds = np.zeros(n)
        songs = [
            (0.04, [(2800, 14, 130, 0.09, 0.017)]),
            (0.13, [(2450,  9,  75, 0.13, 0.013), (2650, 11,  95, 0.09, 0.012)]),
            (0.23, [(3100, 18, 210, 0.07, 0.019)]),
            (0.33, [(2200,  7,  55, 0.15, 0.012)]),
            (0.42, [(2750, 12, 150, 0.10, 0.016), (2950, 14, 170, 0.08, 0.014)]),
            (0.51, [(3200, 20, 260, 0.06, 0.018)]),
            (0.60, [(2500, 10,  90, 0.12, 0.015)]),
            (0.69, [(2350,  8,  65, 0.14, 0.013), (2580, 10,  95, 0.10, 0.012)]),
            (0.79, [(2900, 15, 185, 0.08, 0.017)]),
            (0.88, [(2650, 11, 115, 0.10, 0.015)]),
            (0.95, [(3000, 17, 230, 0.07, 0.016)]),
        ]
        for time_frac, notes in songs:
            pos = int(time_frac * n)
            for f_c, f_mr, f_md, dur_s, amp in notes:
                note = _bird_note(f_c, f_mr, f_md, dur_s)
                end = pos + len(note)
                if end < n:
                    birds[pos:end] += note * amp
                pos = end + int(0.045 * sr)

        audio = audio + birds

    fade = int(15 * sr)
    audio[: min(fade, n)] *= np.linspace(0, 1, min(fade, n))
    audio[-min(fade, n) :] *= np.linspace(1, 0, min(fade, n))
    audio = np.clip(audio, -1, 1)
    s16 = (audio * 32767).astype(np.int16)
    stereo = np.column_stack([s16, s16])
    return AudioSegment(stereo.tobytes(), frame_rate=sr, sample_width=2, channels=2)


# ── iPhone mastering chain ───────────────────────────────────────────────────

def _master_for_iphone(audio: AudioSegment) -> AudioSegment:
    """
    Mastering for iPhone speaker playback:
      1. High-pass 80 Hz  — sub-bass iPhone can't reproduce
      2. Gentle 2.5:1 compression at -20 dBFS
      3. Loudness normalise to -16 dBFS RMS on active content
      4. Peak limit at -1 dBFS
    Note: presence boost removed — the sosfiltfilt bandpass caused audible
    pre/post-ringing on bird calls, perceived as a beeping artefact.
    """
    from scipy.ndimage import uniform_filter1d

    sr = audio.frame_rate
    samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0
    stereo = audio.channels == 2
    arr = samples.reshape(-1, 2) if stereo else samples.reshape(-1, 1)

    out = np.empty_like(arr)
    for ch in range(arr.shape[1]):
        c = arr[:, ch].copy()
        sos_hp = signal.butter(2, 80, btype='highpass', fs=sr, output='sos')
        c = signal.sosfiltfilt(sos_hp, c)
        out[:, ch] = c

    mix = out.mean(axis=1)
    # 2 s RMS window + 0.3 Hz gain smoother → ~3 s transitions.
    # Compressor is too slow to pump with word-level speech/silence cycles
    # (guided "breathe in/out" at ~5–8 s rhythm) — prevents birdsong from
    # modulating in sync with each instruction and sounding like ticking.
    win = int(sr * 2.0)
    rms_env = np.sqrt(uniform_filter1d(mix ** 2, size=win) + 1e-10)
    thr = 10 ** (-20 / 20)
    gain_raw = np.where(rms_env > thr,
                        thr / rms_env * (rms_env / thr) ** (1 / 2.5),
                        1.0)
    sos_gr = signal.butter(2, 0.3, btype='lowpass', fs=sr, output='sos')
    gain = np.clip(signal.sosfiltfilt(sos_gr, gain_raw), 0.0, 2.0)
    out *= gain[:, np.newaxis]

    mix2 = out.mean(axis=1)
    active = np.abs(mix2) > 10 ** (-50 / 20)
    rms_a = np.sqrt((mix2[active] ** 2).mean()) if active.sum() > sr else np.sqrt((mix2 ** 2).mean())
    out *= np.clip(10 ** (-16 / 20) / (rms_a + 1e-10), 0.5, 4.0)

    peak = np.abs(out).max()
    lim = 10 ** (-1 / 20)
    if peak > lim:
        out *= lim / peak

    s16 = (np.clip(out, -1.0, 1.0) * 32767).astype(np.int16)
    if not stereo:
        s16 = s16[:, 0]
    return AudioSegment(s16.tobytes(), frame_rate=sr, sample_width=2, channels=audio.channels)


# ── Audio enhancement ────────────────────────────────────────────────────────

def _enhance(audio: AudioSegment, strength: float) -> AudioSegment:
    from scipy.ndimage import maximum_filter1d

    sr = audio.frame_rate
    samples = np.array(audio.get_array_of_samples())
    original_max = np.abs(samples).max()

    def _speech_mask(c):
        amp = np.abs(c)
        sos_env = signal.butter(2, 8, btype="lowpass", fs=sr, output="sos")
        env = signal.sosfiltfilt(sos_env, amp)
        peak = env.max() if env.max() > 0 else 1e-10

        # Voice regions: 2% of peak, 200 ms expansion.
        # No separate breath mask needed here: the smooth floor in _process_channel
        # (c_peak × 0.5%) already prevents the click detector from catching anything
        # below 4% of peak (ratio < 8), so quiet breaths are safe without extra
        # masking — and adding a mask would also protect swallows.
        raw_v = (env > peak * 0.02).astype(np.uint8)
        return maximum_filter1d(raw_v, size=2 * int(sr * 0.20) + 1).astype(bool)

    def _process_channel(c_int16):
        c = c_int16.astype(np.float32) / 32768.0

        speech = _speech_mask(c)

        amp = np.abs(c)
        sos_env2 = signal.butter(2, 10, btype="lowpass", fs=sr, output="sos")
        smooth = signal.sosfiltfilt(sos_env2, amp)
        # Floor at 0.5% of channel peak — keeps the ratio bounded near silence so
        # breath onsets emerging from digital silence aren't flagged as clicks.
        c_peak = amp.max() if amp.max() > 0 else 1e-10
        smooth = np.maximum(smooth, c_peak * 0.005)

        # Spike >8× local average AND audible (>0.001 ≈ -60 dBFS), outside speech.
        is_click = ((amp / smooth) > 8.0) & (amp > 0.001) & ~speech
        expand_c = int(sr * 0.008)
        is_click = maximum_filter1d(is_click.astype(np.uint8),
                                    size=2 * expand_c + 1).astype(bool)

        # Smooth gain curve: 1.0 everywhere, fades to 0 at click locations
        raw_gain = np.where(is_click, 0.0, 1.0).astype(np.float32)
        sos_g = signal.butter(2, 150, btype="lowpass", fs=sr, output="sos")
        gain = np.clip(signal.sosfiltfilt(sos_g, raw_gain), 0.0, 1.0)

        return c * gain

    if audio.channels == 2:
        samples = samples.reshape((-1, 2))
        channels = [_process_channel(samples[:, ch]) for ch in range(2)]
        out_f = np.column_stack(channels)
    else:
        out_f = _process_channel(samples)

    out_max = np.abs(out_f).max()
    if original_max > 0 and out_max < original_max / 32768.0 * 0.05:
        print("[warn] click removal wiped audio — returning original unchanged")
        return audio

    out = (np.clip(out_f, -1.0, 1.0) * 32767).astype(np.int16)
    return AudioSegment(out.tobytes(), frame_rate=sr, sample_width=2, channels=audio.channels)


# ── Custom nature sound loader ───────────────────────────────────────────────

def _load_custom_nature(path: str, duration_ms: int) -> AudioSegment:
    """Load, pre-fade endpoints to zero, loop by hard-concat, apply block fade.

    WHY every crossfade approach kept glitching
    ─────────────────────────────────────────────
    A crossfade blends the clip's END with its START.  For ocean/wave files
    this always produces an audible artefact because:
      • Level mismatch  (quiet start, loud end or vice-versa), AND/OR
      • Texture mismatch (mid-crash end blended with quiet-ripple start)
    Even smart loop-point search + endpoint normalisation can't fix texture
    mismatch — tested and confirmed still 5–9 dB off at the seam.

    The only approach that is provably inaudible on ANY clip
    ─────────────────────────────────────────────────────────
    1. Apply a short fade-in (≤ 3 s) AND fade-out (≤ 3 s) to the clip itself.
       Both endpoints become exactly 0.
    2. Hard-concatenate (NO crossfade).  Splice point: 0 → 0 = −200 dBFS jump.
       Mathematically silent regardless of what the clip content is.
    3. The seam region sounds like a natural ~6-second lull between waves —
       completely normal for ocean/forest/rain recordings.
    4. Apply the standard 15-s outer fade to the whole practice-section block.

    Tested: −200 dBFS at splice point for worst-case (quiet-start, loud-end).
    """
    wav_path = tempfile.mktemp(suffix=".wav")
    r = subprocess.run(
        ["/usr/bin/afconvert", "-f", "WAVE", "-d", "LEI16@44100",
         "-c", "2", path, wav_path],
        capture_output=True
    )
    if r.returncode != 0:
        raise RuntimeError(f"Could not read nature file: {r.stderr.decode()[-200:]}")
    seg = AudioSegment.from_wav(wav_path)
    os.unlink(wav_path)

    if len(seg) >= duration_ms:
        seg = seg[:duration_ms]
        fade_ms = min(15000, duration_ms // 2)
        return seg.fade_in(fade_ms).fade_out(fade_ms)

    # ── Pre-fade clip endpoints to zero ────────────────────────────────────
    # Max 3 s, max 10 % of clip, min 0.5 s.
    inner_ms = min(3000, max(int(len(seg) * 0.10), 500))
    seg = seg.fade_in(inner_ms).fade_out(inner_ms)
    # Both endpoints are now exactly 0 dBFS → splicing is always silent.

    # ── Hard-concat until long enough ──────────────────────────────────────
    while len(seg) < duration_ms:
        seg = seg + seg          # pydub + = sample-accurate hard concat, no xfade

    seg = seg[:duration_ms]
    # ── Outer block fade (whole practice section) ───────────────────────────
    fade_ms = min(15000, duration_ms // 2)
    return seg.fade_in(fade_ms).fade_out(fade_ms)


# ── Silence expander (reduces gradual sounds like swallows between words) ────

def _silence_reduce(audio: AudioSegment, reduction_db: float = 15.0) -> AudioSegment:
    """
    Reduce non-speech regions by reduction_db.
    The click detector only catches impulsive spikes; gradual sounds (swallows,
    light lip smacks) stay near ratio=1 and are invisible to it.  A downward
    expander in the silence gaps attenuates them without touching the voice.

    Breath protection strategy (bug #14 / updated):
      1. Speech mask: 350 ms expansion, 8 Hz smoothing — protects breaths close
         to speech.
      2. Duration-gated breath mask: detects quiet sustained sounds (0.5–10% of
         peak).  Erode with a 300 ms window removes any event shorter than ~600 ms
         (minimum_filter1d shrinks both sides by N//2 ≈ 300 ms, so swallows at
         100–400 ms disappear entirely).  Expand 500 ms restores the breath
         boundary.  Combined with speech mask via np.maximum so either gate alone
         can protect a region.
    """
    from scipy.ndimage import maximum_filter1d as _mf1d, minimum_filter1d as _min1d
    sr = audio.frame_rate
    samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0
    stereo = audio.channels == 2
    mono = samples.reshape(-1, 2)[:, 0] if stereo else samples

    sos_env = signal.butter(2, 8, btype='lowpass', fs=sr, output='sos')
    env = signal.sosfiltfilt(sos_env, np.abs(mono))
    peak = env.max() if env.max() > 0 else 1e-10

    # --- Speech mask (unchanged from bug #14 fix) ---
    raw_s = (env > peak * 0.02).astype(np.uint8)
    # 350 ms expansion: protects breaths beginning up to 350 ms after speech ends
    speech = _mf1d(raw_s, size=2 * int(sr * 0.35) + 1)

    # --- Duration-gated breath mask ---
    # Detect quiet sustained sounds in the 0.5–10 % band (breaths, not clicks)
    raw_b = ((env > peak * 0.005) & (env < peak * 0.10)).astype(np.uint8)
    # Erode with a 600 ms window: minimum_filter1d(size=N) removes any event
    # shorter than N samples ≈ 1200 ms (erosion shrinks each side by N//2).
    # Swallows top out at ~800–900 ms in the smoothed envelope → eroded away.
    # Deliberate meditation breaths (1.5–3 s) survive with a 300–1800 ms core.
    erode_sz = 2 * int(sr * 0.60) + 1
    raw_b_eroded = _min1d(raw_b, size=erode_sz)
    # Expand 700 ms: restore the full breath boundary from the surviving core
    breath_mask = _mf1d(raw_b_eroded, size=2 * int(sr * 0.70) + 1)

    # --- Combine: either gate protects the region ---
    combined = np.maximum(speech, breath_mask).astype(np.float32)

    # Smooth the gate curve (8 Hz = 125 ms ramp — gentle enough not to clip breath edges)
    sos_sm = signal.butter(2, 8, btype='lowpass', fs=sr, output='sos')
    speech_f = np.clip(signal.sosfiltfilt(sos_sm, combined), 0.0, 1.0)

    silence_gain = 10 ** (-reduction_db / 20)
    scale = speech_f + (1.0 - speech_f) * silence_gain   # 1.0 in speech/breath, ~0.18 in silence

    if stereo:
        result = samples.reshape(-1, 2) * scale[:, np.newaxis]
        s16 = (np.clip(result, -1.0, 1.0) * 32767).astype(np.int16)
    else:
        s16 = (np.clip(samples * scale, -1.0, 1.0) * 32767).astype(np.int16)
    return AudioSegment(s16.tobytes(), frame_rate=sr, sample_width=2, channels=audio.channels)


# ── Time-string parser ────────────────────────────────────────────────────────

def _parse_time(s: str) -> int:
    """Parse 'M:SS[.f]' or 'S[.f]' into milliseconds.  Returns 0 for blank/invalid."""
    s = (s or "").strip()
    if not s:
        return 0
    try:
        if ":" in s:
            parts = s.split(":", 1)
            return int((int(parts[0]) * 60 + float(parts[1])) * 1000)
        return int(float(s) * 1000)
    except Exception:
        return 0


def _apply_cuts(audio: AudioSegment, cuts: list) -> tuple:
    """
    Remove a list of [{start_ms, end_ms}] regions from audio.
    Merges overlapping cuts, stitches the remaining segments with a 50 ms
    crossfade at each splice to prevent clicks.
    Returns (new_audio, adjust_fn) where adjust_fn(original_ms) maps any
    timestamp in the original audio to the equivalent position after cuts.
    """
    if not cuts:
        return audio, lambda ms: ms

    sorted_cuts = sorted(cuts, key=lambda c: c['start_ms'])
    merged = []
    for c in sorted_cuts:
        s = max(0, int(c['start_ms']))
        e = min(len(audio), int(c['end_ms']))
        if s >= e:
            continue
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append([s, e])

    if not merged:
        return audio, lambda ms: ms

    # Build result from non-cut segments
    segments = []
    prev = 0
    for (s, e) in merged:
        if s > prev:
            segments.append(audio[prev:s])
        prev = e
    if prev < len(audio):
        segments.append(audio[prev:])

    if not segments:
        return AudioSegment.silent(duration=0, frame_rate=audio.frame_rate), lambda ms: 0

    result = segments[0]
    for seg in segments[1:]:
        xf = min(50, len(result), len(seg))
        result = result.append(seg, crossfade=xf)

    # Build timestamp mapping function
    def adjust_ms(original_ms):
        offset = 0
        for (s, e) in merged:
            if original_ms <= s:
                break
            elif original_ms < e:
                offset += original_ms - s
                break
            else:
                offset += e - s
        return max(0, original_ms - offset)

    return result, adjust_ms


# ── Phrase detection ─────────────────────────────────────────────────────────

def _find_phrase(audio_path: str):
    try:
        import whisper
        model = whisper.load_model("base")
        result = model.transcribe(audio_path, word_timestamps=True)
        triggers = ["let's practice", "lets practice", "let us practice",
                    "practice for a minute", "practice now", "practice for a few"]
        for seg in result.get("segments", []):
            txt = seg["text"].lower()
            for phrase in triggers:
                if phrase in txt:
                    return float(seg["start"])
    except Exception as e:
        print(f"[whisper] {e}", file=sys.stderr)
    return None


# ── Core processing (runs in background thread) ──────────────────────────────

def _run_job(job_id, input_path, noise_strength, nature_type,
             nature_minutes, manual_ts, use_auto, custom_nature_path=None,
             nature_vol_db=-6.0, trim_start="", trim_end="", cuts=None):

    def log(msg):
        _jobs[job_id]["log"].append(msg)
        print(msg)

    try:
        log("Loading audio…")
        fsize = os.path.getsize(input_path)
        with open(input_path, 'rb') as fh:
            header = fh.read(12)
        log(f"  Upload: {fsize} bytes, header={header.hex()}")
        wav_path = tempfile.mktemp(suffix=".wav")
        # afconvert (macOS built-in) → WAV, then read with Python's wave module (no ffprobe needed)
        wav_path = tempfile.mktemp(suffix=".wav")
        r = subprocess.run(
            ["/usr/bin/afconvert", "-f", "WAVE", "-d", "LEI16@44100",
             "-c", "2", input_path, wav_path],
            capture_output=True
        )
        log(f"  afconvert rc={r.returncode} stderr={r.stderr.decode()[:200]}")
        if r.returncode != 0:
            raise RuntimeError(f"afconvert failed: {r.stderr.decode()[-200:]}")
        log(f"  WAV size: {os.path.getsize(wav_path)} bytes")

        import wave as wave_mod
        with wave_mod.open(wav_path, 'rb') as wf:
            n_ch = wf.getnchannels()
            sw   = wf.getsampwidth()
            fr   = wf.getframerate()
            nf   = wf.getnframes()
            pcm  = wf.readframes(nf)
        os.unlink(wav_path)
        log(f"  WAV read: ch={n_ch} sw={sw} fr={fr} frames={nf} pcm_bytes={len(pcm)}")

        raw = np.frombuffer(pcm, dtype=np.int16).copy()
        log(f"  raw max={np.abs(raw).max()}")

        # resample/remix to 44100 stereo if needed
        if fr != 44100 or n_ch != 2:
            log(f"  resampling {fr}Hz {n_ch}ch → 44100Hz 2ch")
            tmp_in  = tempfile.mktemp(suffix=".wav")
            tmp_out = tempfile.mktemp(suffix=".wav")
            with wave_mod.open(tmp_in, 'wb') as wo:
                wo.setnchannels(n_ch); wo.setsampwidth(sw)
                wo.setframerate(fr);   wo.writeframes(pcm)
            subprocess.run([_FFMPEG, "-y", "-i", tmp_in,
                            "-ar", "44100", "-ac", "2", "-f", "s16le", tmp_out],
                           check=True, capture_output=True)
            raw = np.fromfile(tmp_out, dtype=np.int16).copy()
            os.unlink(tmp_in); os.unlink(tmp_out)
            n_ch = 2
            log(f"  after resample max={np.abs(raw).max()}")

        audio = AudioSegment(raw.tobytes(), frame_rate=44100, sample_width=2, channels=2)
        total_s = len(audio) / 1000
        log(f"  Duration: {int(total_s // 60)}:{total_s % 60:04.1f}")
        log(f"  samples={len(raw)} max={np.abs(raw).max()}")

        # ── Trim (applied before anything else) ─────────────────────────────
        # Both values are positions in the ORIGINAL file.  All downstream
        # timestamps are automatically adjusted by subtracting trim_start_ms.
        trim_start_ms = _parse_time(trim_start)
        trim_end_ms   = _parse_time(trim_end)
        if trim_start_ms > 0 or (0 < trim_end_ms < len(audio)):
            t_end = trim_end_ms if 0 < trim_end_ms < len(audio) else len(audio)
            t_end = max(t_end, trim_start_ms + 500)  # keep at least 500 ms
            audio = audio[trim_start_ms : t_end]
            total_s = len(audio) / 1000
            log(f"  Trimmed → {int(total_s//60)}:{total_s%60:04.1f}  "
                f"(cut {trim_start_ms/1000:.1f}s from start"
                + (f", kept until {t_end/1000:.1f}s" if trim_end_ms > 0 else "")
                + ")")

        # ── Apply user-marked cuts ───────────────────────────────────────────
        # Cuts are in original-file coordinates (as marked on the waveform).
        # We adjust them for trim (subtract trim_start_ms) then apply them.
        _cut_adjust = lambda ms: ms  # identity until cuts are applied
        if cuts:
            adjusted_cuts = []
            for c in cuts:
                cs = int(c.get("start_ms", 0)) - trim_start_ms
                ce = int(c.get("end_ms",   0)) - trim_start_ms
                if ce > cs and ce > 0 and cs < len(audio):
                    adjusted_cuts.append({"start_ms": max(0, cs), "end_ms": min(len(audio), ce)})
            if adjusted_cuts:
                audio, _cut_adjust = _apply_cuts(audio, adjusted_cuts)
                total_s = len(audio) / 1000
                log(f"  Applied {len(adjusted_cuts)} cut(s) → {int(total_s//60)}:{total_s%60:04.1f}")

        # Resolve timestamp first so we only enhance audio before that point.
        # This prevents the click-detector from touching the silence/nature region.
        practice_ms = None

        if manual_ts:
            raw_ms = _parse_time(manual_ts)
            if raw_ms > 0:
                practice_ms = raw_ms - trim_start_ms
                if practice_ms < 0:
                    log(f"  Warning: manual timestamp {manual_ts} is before trim start — ignoring.")
                    practice_ms = None
                else:
                    m = int(practice_ms // 1000 // 60)
                    s = (practice_ms / 1000) % 60
                    log(f"Using manual timestamp: {manual_ts}"
                        + (f" → adjusted to {m}:{s:04.1f} after trim" if trim_start_ms else ""))
            else:
                log("  Could not parse timestamp — trying auto-detect.")

        if practice_ms is None and use_auto:
            log("Auto-detecting 'let's practice'… (may take a moment)")
            ts = _find_phrase(input_path)
            if ts is not None:
                practice_ms = ts * 1000 - trim_start_ms
                practice_ms = max(0, practice_ms)
                m = int(practice_ms // 1000 // 60)
                s = (practice_ms / 1000) % 60
                log(f"  Found at {m}:{s:04.1f}")
            else:
                log("  Phrase not found — adding nature sounds near the end.")
                practice_ms = max(0, len(audio) - int(nature_minutes * 60 * 1000))

        # Apply cut offset to practice_ms
        if practice_ms is not None:
            practice_ms = _cut_adjust(practice_ms)

        log(f"Reducing noise (strength {noise_strength:.0%})…")
        audio = _enhance(audio, noise_strength)
        after = np.array(audio.get_array_of_samples())
        log(f"  After enhance: max={np.abs(after).max()}")
        log("  Done.")

        if practice_ms is None:
            log("No practice point set — noise reduction only.")
            result = audio
        else:
            nature_ms = int(nature_minutes * 60 * 1000)
            log(f"Generating {nature_minutes:.1f} min of {nature_type} sounds…")
            if nature_type == "Custom" and custom_nature_path:
                nature = _load_custom_nature(custom_nature_path, nature_ms)
            elif nature_type in PRESET_FILES:
                preset_path = os.path.join(PRESETS_DIR, PRESET_FILES[nature_type])
                nature = _load_custom_nature(preset_path, nature_ms)
            else:
                nature = _make_nature(nature_ms, nature_type)

            if nature.frame_rate != audio.frame_rate:
                nature = nature.set_frame_rate(audio.frame_rate)

            nature = nature.apply_gain(nature_vol_db)

            before = _silence_reduce(audio[: int(practice_ms)])
            after  = audio[int(practice_ms) :]
            practice_section = after[:nature_ms]
            rest   = after[nature_ms:]

            # Gate the practice section: silence → digital zero, voice guidance → intact.
            # Eliminates the MP3 noise floor that causes ticking under the birdsong.
            from scipy.ndimage import maximum_filter1d as _mf1d, minimum_filter1d as _min1d
            ps_sr = practice_section.frame_rate
            ps_samples = np.array(practice_section.get_array_of_samples()).astype(np.float32) / 32768.0
            if practice_section.channels == 2:
                ps_mono = ps_samples.reshape(-1, 2)[:, 0]
            else:
                ps_mono = ps_samples
            amp_ps = np.abs(ps_mono)
            sos_ps = signal.butter(2, 8, btype="lowpass", fs=ps_sr, output="sos")
            env_ps = signal.sosfiltfilt(sos_ps, amp_ps)
            peak_ps = env_ps.max() if env_ps.max() > 0 else 1e-10

            # Voice mask: 2% of peak, 400 ms expansion
            raw_v    = (env_ps > peak_ps * 0.02).astype(np.uint8)
            voice_ps = _mf1d(raw_v, size=2 * int(ps_sr * 0.40) + 1)

            # Breath mask: 0.5–10% band, erode < 100 ms, expand 600 ms.
            # Catches inhales/exhales that fall below the voice threshold.
            raw_b     = ((env_ps > peak_ps * 0.005) & (env_ps < peak_ps * 0.10)).astype(np.uint8)
            raw_b     = _min1d(raw_b, size=int(ps_sr * 0.10))
            breath_ps = _mf1d(raw_b, size=2 * int(ps_sr * 0.60) + 1)

            speech_ps = np.maximum(voice_ps, breath_ps).astype(np.float32)
            # 5 Hz (not 20 Hz) keeps gate ramps slow enough (~100 ms) that
            # closing/opening transitions don't thud or click.
            sos_fade  = signal.butter(2, 5, btype="lowpass", fs=ps_sr, output="sos")
            gate      = np.clip(signal.sosfiltfilt(sos_fade, speech_ps), 0.0, 1.0)

            # Apply a gate floor so any existing background music/ambience in the
            # original track is ATTENUATED rather than silenced.  Hard-gating to
            # zero causes background content to "come and go" in sync with the
            # voice, which sounds like a rhythmic noise loop.
            # -30 dB floor: drops MP3 noise floor to -110 dBFS (inaudible) while
            # keeping background music above ~-60 dBFS (below audibility in a
            # meditation context, and without the abrupt on/off pumping effect).
            gate_floor = 10 ** (-30 / 20)   # ≈ 0.032
            gate = gate * (1.0 - gate_floor) + gate_floor
            if practice_section.channels == 2:
                ps_clean = ps_samples.reshape(-1, 2) * gate[:, np.newaxis]
                s16_ps = (np.clip(ps_clean, -1.0, 1.0) * 32767).astype(np.int16)
            else:
                s16_ps = (np.clip(ps_samples * gate, -1.0, 1.0) * 32767).astype(np.int16)
            practice_section = AudioSegment(s16_ps.tobytes(), frame_rate=ps_sr,
                                            sample_width=2, channels=practice_section.channels)
            log(f"  Practice section gated: {gate.mean():.2%} speech detected.")

            nature = nature[: len(practice_section)]
            mixed  = practice_section.overlay(nature)

            # 100 ms crossfade at each splice prevents sample-level click from
            # the gain-scale difference between _silence_reduce (floor ≈ −15 dB)
            # and the practice gate (floor ≈ −30 dB) at the junction points.
            result = before.append(mixed, crossfade=100)
            if len(rest) > 0:
                result = result.append(rest, crossfade=100)
            log("  Nature sounds mixed in.")

        log("Mastering for iPhone speakers…")
        result = _master_for_iphone(result)
        log("  Done.")

        out = tempfile.mktemp(suffix=".mp3")
        log("Exporting MP3…")
        result.export(out, format="mp3", bitrate="192k",
                      tags={"comment": "Processed by Mithril Meditation Editor"})

        _jobs[job_id]["output"] = out
        _jobs[job_id]["status"] = "done"
        log("Done! Click Download below.")

    except Exception as e:
        _jobs[job_id]["status"] = "error"
        _jobs[job_id]["log"].append(f"Error: {e}")
        print(f"[job error] {e}", file=sys.stderr)
    finally:
        for p in [input_path, custom_nature_path]:
            try:
                if p:
                    os.unlink(p)
            except Exception:
                pass


# ── Flask app ────────────────────────────────────────────────────────────────

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB max upload


@app.route("/")
def index():
    return _HTML


@app.route("/process", methods=["POST"])
def process_upload():
    f = request.files.get("audio")
    if not f or not f.filename:
        return _page("No file selected. <a href='/'>Go back</a>")

    ext = os.path.splitext(f.filename)[1].lower() or ".mp3"
    tmp = tempfile.mktemp(suffix=ext)
    f.save(tmp)

    noise_strength = float(request.form.get("noise_strength", 0.75))
    nature_type = request.form.get("nature_type", "British Woods")
    nature_minutes = float(request.form.get("nature_minutes", 1.5))
    nature_vol_db = float(request.form.get("nature_vol_db", -14.0))
    manual_ts = request.form.get("manual_ts", "").strip()
    use_auto = "use_auto" in request.form
    trim_start = request.form.get("trim_start", "").strip()
    trim_end   = request.form.get("trim_end",   "").strip()

    import json as _json
    try:
        cuts_raw = request.form.get("cuts", "[]")
        cuts = _json.loads(cuts_raw) if cuts_raw else []
        if not isinstance(cuts, list):
            cuts = []
    except Exception:
        cuts = []

    import shutil as _shutil
    custom_nature_path = None
    nature_source = request.form.get("nature_source", "upload")

    if nature_source == "library":
        lib_name = os.path.basename(request.form.get("library_file", "").strip())
        if lib_name:
            lib_full = os.path.join(LIBRARY_DIR, lib_name)
            if os.path.exists(lib_full):
                nf_ext = os.path.splitext(lib_name)[1].lower() or ".mp3"
                custom_nature_path = tempfile.mktemp(suffix=nf_ext)
                _shutil.copy2(lib_full, custom_nature_path)
    else:
        nf = request.files.get("nature_file")
        if nf and nf.filename:
            nf_ext = os.path.splitext(nf.filename)[1].lower() or ".mp3"
            custom_nature_path = tempfile.mktemp(suffix=nf_ext)
            nf.save(custom_nature_path)
            # Deduplicate: if a file with this name already exists, add _2 / _3 …
            safe_name = os.path.basename(nf.filename)
            base, ext = os.path.splitext(safe_name)
            lib_path  = os.path.join(LIBRARY_DIR, safe_name)
            counter   = 2
            while os.path.exists(lib_path):
                safe_name = f"{base}_{counter}{ext}"
                lib_path  = os.path.join(LIBRARY_DIR, safe_name)
                counter  += 1
            try:
                _shutil.copy2(custom_nature_path, lib_path)
                print(f"[library] Saved as {safe_name}")
            except Exception as e:
                print(f"[library] Could not save {safe_name}: {e}")

    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "processing", "log": [], "output": None}

    threading.Thread(
        target=_run_job,
        args=(job_id, tmp, noise_strength, nature_type,
              nature_minutes, manual_ts, use_auto, custom_nature_path,
              nature_vol_db),
        kwargs={"trim_start": trim_start, "trim_end": trim_end, "cuts": cuts},
        daemon=True,
    ).start()

    from flask import redirect
    return redirect(f"/waiting/{job_id}")


@app.route("/library")
def library_list():
    files = sorted(
        f for f in os.listdir(LIBRARY_DIR)
        if not f.startswith(".") and os.path.isfile(os.path.join(LIBRARY_DIR, f))
    )
    return jsonify(files)


@app.route("/library/delete/<path:filename>", methods=["POST"])
def library_delete(filename):
    safe = os.path.basename(filename)
    target = os.path.join(LIBRARY_DIR, safe)
    if os.path.exists(target):
        os.unlink(target)
    return jsonify({"ok": True})


@app.route("/analyze", methods=["POST"])
def analyze_audio():
    """
    Transcribe + flag potential edit points.
    Returns JSON: {transcript: [{word, start, end}], suggestions: [{start_ms, end_ms, text, reason}]}
    """
    f = request.files.get("audio")
    if not f:
        return jsonify({"error": "No file"}), 400

    ext = os.path.splitext(f.filename or "")[1].lower() or ".mp3"
    tmp = tempfile.mktemp(suffix=ext)
    f.save(tmp)

    suggestions = []
    transcript = None

    try:
        # ── Try Whisper (word-level transcript) ──────────────────────────────
        try:
            import whisper as _whisper
            model = _whisper.load_model("base")
            result = model.transcribe(tmp, word_timestamps=True)
            transcript = []
            hesitations = {"uh", "um", "er", "hmm", "ah", "eh", "uhh", "umm"}
            prev_word = None
            prev_info = None

            for seg in result.get("segments", []):
                for w in seg.get("words", []):
                    word_clean = w["word"].strip()
                    word_lower = word_clean.lower().strip(".,!?;:-")
                    w_start = float(w["start"])
                    w_end   = float(w["end"])
                    transcript.append({"word": word_clean, "start": w_start, "end": w_end})

                    # Hesitation / filler
                    if word_lower in hesitations:
                        suggestions.append({
                            "start_ms": max(0, int(w_start * 1000) - 80),
                            "end_ms":   int(w_end * 1000) + 150,
                            "text":     word_clean,
                            "reason":   f'Filler word: "{word_clean}"',
                        })

                    # Immediate word repetition (e.g. "the the", "and and")
                    if prev_word is not None and prev_word == word_lower and prev_info is not None:
                        suggestions.append({
                            "start_ms": max(0, int(prev_info["start"] * 1000) - 50),
                            "end_ms":   int(w_end * 1000) + 100,
                            "text":     f"{prev_info['word']} {word_clean}",
                            "reason":   f'Repeated word: "{word_lower} {word_lower}"',
                        })

                    prev_word = word_lower
                    prev_info = {"word": word_clean, "start": w_start, "end": w_end}

        except ImportError:
            # ── Amplitude-only fallback ──────────────────────────────────────
            wav_path = tempfile.mktemp(suffix=".wav")
            r = subprocess.run(
                ["/usr/bin/afconvert", "-f", "WAVE", "-d", "LEI16@44100",
                 "-c", "1", tmp, wav_path],
                capture_output=True
            )
            if r.returncode == 0 and os.path.exists(wav_path):
                import wave as _wm
                with _wm.open(wav_path, "rb") as wf:
                    sr_a = wf.getframerate()
                    pcm  = wf.readframes(wf.getnframes())
                os.unlink(wav_path)
                arr = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0

                sos_e = signal.butter(2, 8, btype="lowpass", fs=sr_a, output="sos")
                env   = signal.sosfiltfilt(sos_e, np.abs(arr))
                peak  = env.max() or 1e-10
                is_sp = (env > peak * 0.02).astype(np.int8)

                changes  = np.diff(is_sp)
                sp_on    = np.where(changes ==  1)[0]
                sp_off   = np.where(changes == -1)[0]

                # Align on/off arrays
                if len(sp_on) and len(sp_off):
                    if sp_off[0] < sp_on[0]:
                        sp_off = sp_off[1:]
                    min_len = min(len(sp_on), len(sp_off))
                    sp_on, sp_off = sp_on[:min_len], sp_off[:min_len]

                    for st, en in zip(sp_on, sp_off):
                        dur = (en - st) / sr_a
                        # Very short speech burst: possible false start
                        if 0.05 < dur < 0.4:
                            s_ms = max(0, int(st / sr_a * 1000) - 300)
                            e_ms = min(int(len(arr) / sr_a * 1000),
                                       int(en / sr_a * 1000) + 300)
                            suggestions.append({
                                "start_ms": s_ms,
                                "end_ms":   e_ms,
                                "text":     f"Short burst ({dur:.2f}s)",
                                "reason":   "Possible false start or stutter",
                            })

    finally:
        try:
            os.unlink(tmp)
        except Exception:
            pass

    return jsonify({
        "transcript": transcript,
        "suggestions": suggestions[:25],
        "whisper_available": transcript is not None,
    })


@app.route("/waiting/<job_id>")
def waiting(job_id):
    job = _jobs.get(job_id)
    if not job:
        return _page("Job not found. <a href='/'>Go back</a>")

    log_text = "\n".join(job["log"]) or "Starting…"

    if job["status"] == "done":
        return _page(
            f'<h2 style="color:#111">Done!</h2>'
            f'<p style="margin:16px 0">Your meditation is ready.</p>'
            f'<a href="/download/{job_id}" style="display:inline-block;padding:14px 32px;'
            f'background:#111;color:#fff;text-decoration:none;border-radius:10px;'
            f'font-weight:600;font-size:1rem">Download meditation_edited.mp3</a>'
            f'<p style="margin-top:20px"><a href="/">Process another file</a></p>'
        )
    elif job["status"] == "error":
        return _page(
            f'<h2 style="color:#b00">Something went wrong</h2>'
            f'<pre style="margin-top:12px;background:#fff8f8;padding:12px;'
            f'border-radius:8px;font-size:0.85rem">{log_text}</pre>'
            f'<p style="margin-top:16px"><a href="/">Try again</a></p>'
        )
    else:
        return _page(
            f'<meta http-equiv="refresh" content="2">'
            f'<h2>Processing your meditation…</h2>'
            f'<p style="color:#666;margin:12px 0">This page refreshes automatically.</p>'
            f'<pre style="background:#f8f8f8;padding:14px;border-radius:8px;'
            f'font-size:0.85rem;line-height:1.6;white-space:pre-wrap">{log_text}</pre>'
        )


@app.route("/download/<job_id>")
def download(job_id):
    job = _jobs.get(job_id)
    if not job or job["status"] != "done":
        return "Not ready", 404
    return send_file(
        job["output"],
        as_attachment=True,
        download_name="meditation_edited.mp3",
        mimetype="audio/mpeg",
    )


def _page(body):
    return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mithril Meditation Editor</title>
<style>
  body{{font-family:-apple-system,sans-serif;background:#ffffff;color:#111111;
       min-height:100vh;padding:40px 16px;}}
  .wrap{{max-width:680px;margin:0 auto;}}
  h1{{color:#111;margin-bottom:24px;}}
  h2{{color:#111;margin-bottom:8px;}}
  a{{color:#444;}}
</style></head>
<body><div class="wrap">
<h1>Mithril Meditation Editor</h1>
{body}
</div></body></html>"""


# ── HTML UI (single-file, no external assets needed) ────────────────────────

_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Mithril Meditation Editor</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: #ffffff;
    color: #111111;
    min-height: 100vh;
    padding: 0 0 80px;
  }
  .wrap { max-width: 680px; margin: 0 auto; padding: 0 16px; }
  h1 { font-size: 1.7rem; color: #111; margin-bottom: 6px; }
  .subtitle { color: #666; margin-bottom: 36px; font-size: 0.97rem; }

  /* ── Hero ─────────────────────────────────────────────── */
  .hero-outer {
    background: #ffffff;
    padding: 56px 16px 48px;
    text-align: center;
    border-bottom: 1px solid #ececec;
    position: relative;
    overflow: hidden;
    margin-bottom: 36px;
  }
  .hero-outer .wrap { max-width: 600px; position: relative; }

  /* ✦ sparkles — each drifts along its own path */
  .spark {
    position: absolute;
    pointer-events: none; user-select: none;
    color: #aaa;
  }
  .spark:nth-child(1){
    top:18%; left:6%; font-size:0.7rem;
    animation: drift1 5.0s ease-in-out infinite;
  }
  .spark:nth-child(2){
    top:15%; right:9%; font-size:1.0rem; color:#888;
    animation: drift2 4.2s ease-in-out infinite; animation-delay:-1.3s;
  }
  .spark:nth-child(3){
    top:65%; left:11%; font-size:0.55rem; color:#bbb;
    animation: drift3 6.1s ease-in-out infinite; animation-delay:-2.7s;
  }
  .spark:nth-child(4){
    top:60%; right:8%; font-size:0.8rem;
    animation: drift4 4.8s ease-in-out infinite; animation-delay:-0.9s;
  }
  .spark:nth-child(5){
    top:40%; left:3%; font-size:0.5rem; color:#ccc;
    animation: drift1 7.0s ease-in-out infinite; animation-delay:-3.5s;
  }
  .spark:nth-child(6){
    top:35%; right:4%; font-size:0.65rem; color:#999;
    animation: drift3 5.5s ease-in-out infinite; animation-delay:-1.8s;
  }
  @keyframes drift1 {
    0%   { opacity:0;   transform: translate(0px,   0px)  scale(0.6) rotate(0deg);   }
    20%  { opacity:0.9; transform: translate(8px,  -18px) scale(1.4) rotate(30deg);  }
    50%  { opacity:0.4; transform: translate(18px,  -8px) scale(0.9) rotate(80deg);  }
    75%  { opacity:1;   transform: translate(6px,  -28px) scale(1.2) rotate(130deg); }
    100% { opacity:0;   transform: translate(0px,  -40px) scale(0.5) rotate(180deg); }
  }
  @keyframes drift2 {
    0%   { opacity:0;   transform: translate(0px,  0px)   scale(1.0) rotate(0deg);   }
    30%  { opacity:1;   transform: translate(-12px,-14px) scale(1.5) rotate(-45deg); }
    60%  { opacity:0.3; transform: translate(-6px, -26px) scale(0.8) rotate(-90deg); }
    80%  { opacity:0.8; transform: translate(-18px,-18px) scale(1.3) rotate(-130deg);}
    100% { opacity:0;   transform: translate(-4px, -42px) scale(0.4) rotate(-180deg);}
  }
  @keyframes drift3 {
    0%   { opacity:0;   transform: translate(0px,   0px)  scale(0.7) rotate(0deg);   }
    25%  { opacity:0.8; transform: translate(10px, -10px) scale(1.2) rotate(60deg);  }
    55%  { opacity:0.2; transform: translate(20px, -22px) scale(1.6) rotate(110deg); }
    80%  { opacity:0.9; transform: translate(8px,  -32px) scale(0.9) rotate(150deg); }
    100% { opacity:0;   transform: translate(2px,  -44px) scale(0.4) rotate(200deg); }
  }
  @keyframes drift4 {
    0%   { opacity:0;   transform: translate(0px,   0px)  scale(0.8) rotate(0deg);   }
    35%  { opacity:1;   transform: translate(-8px, -20px) scale(1.4) rotate(-60deg); }
    65%  { opacity:0.5; transform: translate(-15px,-10px) scale(1.0) rotate(-100deg);}
    85%  { opacity:0.9; transform: translate(-5px, -30px) scale(1.3) rotate(-150deg);}
    100% { opacity:0;   transform: translate(-2px, -42px) scale(0.5) rotate(-200deg);}
  }

  .hero-mark {
    font-size: 2.0rem; color: #222; margin-bottom: 12px; display: block;
    animation: markPulse 5s ease-in-out infinite;
  }
  @keyframes markPulse {
    0%,100%{ filter: none; transform: rotate(0deg); }
    50%    { filter: drop-shadow(0 0 8px rgba(0,0,0,0.15)); transform: rotate(15deg); }
  }
  .hero-title {
    font-size: 3.6rem; font-weight: 800; letter-spacing: -2.5px; margin-bottom: 4px;
    color: #111;
  }
  .hero-app-name {
    font-size: 0.82rem; color: #aaa; text-transform: uppercase;
    letter-spacing: .20em; margin-bottom: 18px;
  }
  .hero-tagline {
    font-size: 1.15rem; color: #222; font-weight: 500;
    margin-bottom: 14px; line-height: 1.5;
  }
  .hero-body {
    font-size: 0.95rem; color: #666; line-height: 1.75;
    max-width: 500px; margin: 0 auto 26px;
  }
  .hero-pills {
    display: flex; flex-wrap: wrap; gap: 8px; justify-content: center;
  }
  .hero-pills span {
    background: #fff; border: 1px solid #ddd;
    color: #444; padding: 5px 13px; border-radius: 20px;
    font-size: 0.84rem; font-weight: 500;
  }

  /* ── Form area ────────────────────────────────────────── */
  .card {
    background: #fff;
    border-radius: 16px;
    padding: 28px 28px 24px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04), 0 4px 14px rgba(0,0,0,0.06);
    margin-bottom: 16px;
    border: 1px solid #ececec;
  }
  .card h2 { font-size: 0.78rem; text-transform: uppercase; letter-spacing: .12em;
             color: #bbb; margin-bottom: 16px; }

  /* File upload area */
  .upload-area {
    border: 2px dashed #ddd;
    border-radius: 10px;
    padding: 32px 20px;
    text-align: center;
  }
  .upload-area .icon { font-size: 2.2rem; display: block; margin-bottom: 10px; }
  .upload-area .hint { color: #999; font-size: 0.9rem; margin-top: 8px; }
  .upload-area .fname { color: #111; font-weight: 600; margin-top: 10px; font-size: 0.95rem; }
  input[type=file] { margin-top: 14px; font-size: 0.92rem; color: #333; }
  input[type=file]::file-selector-button {
    padding: 8px 20px; background: #f5f5f5; border: 1.5px solid #ddd;
    border-radius: 8px; font-size: 0.92rem; color: #333;
    cursor: pointer; margin-right: 10px; transition: background .15s;
  }
  input[type=file]::file-selector-button:hover { background: #eee; }

  /* Controls */
  label { display: block; font-size: 0.9rem; color: #222; margin-bottom: 4px; font-weight: 500; }
  .hint-text { font-size: 0.8rem; color: #aaa; margin-bottom: 14px; }
  input[type=range] { width: 100%; accent-color: #111; margin-bottom: 4px; }
  .range-row { display: flex; justify-content: space-between;
               font-size: 0.8rem; color: #aaa; margin-bottom: 16px; }

  .radio-group { display: flex; gap: 8px; margin-bottom: 16px; flex-wrap: wrap; }
  .radio-group label {
    display: flex; align-items: center; gap: 6px;
    background: #fafafa; border: 1.5px solid #e0e0e0;
    border-radius: 8px; padding: 8px 14px; cursor: pointer;
    font-size: 0.9rem; font-weight: 400; color: #444;
    transition: all .15s;
  }
  .radio-group input[type=radio] { accent-color: #111; }
  .radio-group label:has(input:checked) {
    background: #fff; border-color: #111; color: #111; font-weight: 600;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
  }

  input[type=text] {
    width: 100%; padding: 9px 12px; border: 1.5px solid #e0e0e0;
    border-radius: 8px; font-size: 0.9rem; color: #111;
    background: #fff; outline: none; transition: border-color .15s;
  }
  input[type=text]:focus { border-color: #111; }

  .toggle-row { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
  input[type=checkbox] { width: 18px; height: 18px; accent-color: #111; cursor: pointer; }
  .toggle-row label { margin-bottom: 0; cursor: pointer; }

  select {
    width: 100%; padding: 9px 12px; border: 1.5px solid #e0e0e0;
    border-radius: 8px; font-size: 0.9rem; color: #111;
    background: #fff; outline: none; cursor: pointer;
    -webkit-appearance: none; appearance: none;
  }
  select:focus { border-color: #111; }

  /* Library file list */
  .lib-list {
    max-height: 210px; overflow-y: auto;
    display: flex; flex-direction: column; gap: 6px; margin-bottom: 6px;
  }
  .lib-file-row {
    display: flex; align-items: center; gap: 10px;
    padding: 9px 12px; border-radius: 8px; cursor: pointer;
    border: 1.5px solid #e0e0e0; background: #fafafa;
    transition: background .15s, border-color .15s;
  }
  .lib-file-row:hover { background: #f0f0f0; }
  .lib-file-row.lib-selected { background: #fff; border-color: #111; }
  .lib-fname {
    flex: 1; font-size: 0.88rem; color: #333;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  }
  .lib-del {
    flex-shrink: 0; background: none; border: none; cursor: pointer;
    font-size: 1.05rem; padding: 2px 4px; opacity: 0.35;
    transition: opacity .15s; line-height: 1; color: #333;
  }
  .lib-del:hover { opacity: 1; }

  /* Button */
  button#processBtn {
    width: 100%; padding: 15px;
    background: #111; color: #fff;
    border: none; border-radius: 12px;
    font-size: 1rem; font-weight: 600; letter-spacing: 0.02em;
    cursor: pointer;
    box-shadow: 0 2px 12px rgba(0,0,0,0.15);
    transition: all .2s;
  }
  button#processBtn:hover {
    background: #000;
    box-shadow: 0 4px 20px rgba(0,0,0,0.25);
    transform: translateY(-1px);
  }

  /* Progress */
  #progressCard { display: none; }
  .log-box {
    background: #f8f8f8; border-radius: 8px;
    padding: 14px 16px;
    font-family: "SF Mono", "Fira Code", monospace;
    font-size: 0.82rem; color: #444;
    min-height: 80px; max-height: 200px;
    overflow-y: auto;
    line-height: 1.6;
  }
  .spinner {
    display: inline-block; width: 14px; height: 14px;
    border: 2px solid #ddd; border-top-color: #555;
    border-radius: 50%; animation: spin .7s linear infinite;
    vertical-align: middle; margin-right: 6px;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* Download */
  #downloadCard { display: none; }
  .dl-btn {
    display: inline-block; padding: 13px 28px;
    background: #111; color: #fff; text-decoration: none;
    border-radius: 10px; font-weight: 600; font-size: 1rem;
    transition: background .2s;
  }
  .dl-btn:hover { background: #000; }

  /* Waveform card */
  .wave-canvas-wrap {
    position: relative; width: 100%; height: 88px;
    border-radius: 10px; overflow: hidden; cursor: crosshair;
    background: #f8f8f8; border: 1.5px solid #d8d8d8;
  }
  #waveCanvas { display: block; width: 100%; height: 88px; }
  .wave-controls {
    display: flex; align-items: center; gap: 10px; margin-top: 10px; flex-wrap: wrap;
  }
  .ctrl-btn {
    padding: 7px 14px; background: #f5f5f5; border: 1.5px solid #d0d0d0;
    border-radius: 8px; font-size: 0.88rem; color: #333; cursor: pointer;
    transition: background .15s; white-space: nowrap;
  }
  .ctrl-btn:hover:not(:disabled) { background: #ebebeb; }
  .ctrl-btn:disabled { opacity: 0.5; cursor: default; }
  .ctrl-btn-red {
    background: #fff5f5; border-color: #e8c0c0; color: #a04040;
  }
  .ctrl-btn-red:hover:not(:disabled) { background: #fce8e8; }
  .wave-time {
    font-size: 0.82rem; color: #999; margin-left: auto;
  }
  .cut-item {
    display: flex; align-items: center; gap: 10px; padding: 7px 12px;
    background: #fff5f5; border: 1px solid #e8c8c8; border-radius: 7px;
    margin-bottom: 5px;
  }
  .cut-item span { flex: 1; font-size: 0.86rem; color: #a04040; }
  .cut-item button {
    border: none; background: none; cursor: pointer;
    font-size: 1rem; opacity: 0.55; padding: 0; line-height: 1;
    color: #a04040;
  }
  .cut-item button:hover { opacity: 1; }
  .sugg-item {
    display: flex; align-items: flex-start; gap: 10px; padding: 9px 12px;
    border: 1.5px solid #e0e0e0; border-radius: 9px; margin-bottom: 7px;
    background: #fff;
  }
  .sugg-item .sugg-text { flex: 1; }
  .sugg-item .sugg-reason {
    font-size: 0.87rem; color: #333; font-weight: 500;
  }
  .sugg-item .sugg-ts {
    font-size: 0.79rem; color: #999; margin-top: 2px;
  }
  .sugg-actions { display: flex; gap: 6px; align-items: center; flex-shrink: 0; }
  .transcript-box {
    font-size: 0.84rem; color: #333; line-height: 1.9;
    background: #f8f8f8; padding: 10px 13px; border-radius: 8px;
    margin-bottom: 12px; max-height: 160px; overflow-y: auto;
  }
  .transcript-box span { cursor: pointer; border-radius: 3px; padding: 0 1px; }
  .transcript-box span:hover { background: #e8e8e8; }
  .card-sep { border: none; border-top: 1px solid #ececec; margin: 20px 0 18px; }
</style>
</head>
<body>
<div class="hero-outer">
  <span class="spark">✦</span>
  <span class="spark">✦</span>
  <span class="spark">✦</span>
  <span class="spark">✦</span>
  <div class="wrap">
    <div class="hero-mark">◈</div>
    <h1 class="hero-title">Mithril</h1>
    <p class="hero-app-name">Meditation Audio Editor</p>
    <p class="hero-tagline">Your raw recording, refined to studio quality</p>
    <p class="hero-body">Upload a voice memo or field recording. Mithril automatically removes clicks, mouth noise and background hiss, detects your practice moment, and weaves in immersive nature sounds — so every track sounds like it was made in a professional studio.</p>
    <div class="hero-pills">
      <span>🎙 Noise cleanup</span>
      <span>🌿 Nature sounds</span>
      <span>✂️ Waveform editor</span>
      <span>🎧 iPhone-ready mastering</span>
    </div>
  </div>
</div>
<div class="wrap">

  <form method="POST" action="/process" enctype="multipart/form-data">

    <div class="card">
      <h2>Your recording</h2>
      <div class="upload-area">
        <span class="icon">🎙</span>
        <div>Select your meditation audio file</div>
        <input type="file" name="audio" accept="audio/*,.mp3,.wav,.m4a,.aac" required style="margin-top:14px">
        <div class="hint">MP3, WAV, M4A · up to 200 MB</div>
      </div>
    </div>

    <div class="card" id="waveformCard" style="display:none">
      <h2>Preview &amp; edit</h2>
      <div class="wave-canvas-wrap">
        <canvas id="waveCanvas"></canvas>
      </div>
      <div class="wave-controls">
        <button type="button" class="ctrl-btn" id="playBtn">&#9654; Play</button>
        <button type="button" class="ctrl-btn" id="analyzeBtn">&#128269; Analyze for errors</button>
        <span class="wave-time"><span id="waveCurrentTime">0:00</span> / <span id="waveDuration">&#8212;</span></span>
      </div>
      <p class="hint-text" style="margin-top:8px">Click to seek &middot; drag to mark a cut region &middot; drag short = seek</p>
      <div id="cutList" style="display:none; margin-top:12px">
        <p style="font-size:0.85rem;font-weight:600;color:#a04040;margin-bottom:6px">Regions to cut out:</p>
        <div id="cutItems"></div>
      </div>
      <div id="analysisPanel" style="display:none; margin-top:14px"></div>
      <input type="hidden" name="cuts" id="cutsField" value="[]">
    </div>

    <div class="card">
      <h2>Trim</h2>
      <p class="hint-text" style="margin-bottom:14px">Remove unwanted content from the start or end. Leave blank to keep the full track.</p>
      <div style="display:flex;gap:16px">
        <div style="flex:1">
          <label for="trimStart">Cut from start</label>
          <input type="text" id="trimStart" name="trim_start" placeholder="e.g.  0:10  or  10">
        </div>
        <div style="flex:1">
          <label for="trimEnd">Keep until</label>
          <input type="text" id="trimEnd" name="trim_end" placeholder="e.g.  9:00  or  540">
        </div>
      </div>
      <p class="hint-text" style="margin-top:8px">Use timestamps from your original file. The practice-moment timestamp is adjusted automatically.</p>
    </div>

    <div class="card">
      <h2>Noise cleanup</h2>
      <label for="noiseSlider">Strength</label>
      <input type="range" id="noiseSlider" name="noise_strength" min="0.3" max="1.0" step="0.05" value="0.75">
      <div class="range-row"><span>Gentle</span><span>75% (recommended)</span><span>Full</span></div>
      <p class="hint-text">75% is a great starting point. Go higher for noisier recordings.</p>
    </div>

    <div class="card">
      <h2>Nature sounds</h2>

      <label>Sound</label>
      <div class="radio-group">
        <label><input type="radio" name="nature_type" value="British Woods" checked> 🌲 British Woods</label>
        <label><input type="radio" name="nature_type" value="Spring Rain"> 🌧 Spring Rain</label>
        <label><input type="radio" name="nature_type" value="Sandy Beach"> 🌊 Sandy Beach</label>
        <label><input type="radio" name="nature_type" value="Custom"> ↑ My library / upload</label>
      </div>

      <div id="customNaturePanel" style="display:none">
        <div class="radio-group" style="margin-bottom:10px">
          <label><input type="radio" name="nature_source" value="library" id="srcLibrary" checked> My library</label>
          <label><input type="radio" name="nature_source" value="upload" id="srcUpload"> Upload new</label>
        </div>
        <div id="libraryPanel">
          <div class="lib-list" id="libraryList"></div>
          <input type="hidden" id="librarySelect" name="library_file" value="">
          <p class="hint-text" id="libraryCount"></p>
        </div>
        <div id="uploadPanel" style="display:none; margin-top:4px">
          <input type="file" id="natureFile" name="nature_file" accept="audio/*,.mp3,.wav,.m4a,.aac">
          <p class="hint-text" style="margin-top:6px">Free recordings: <strong>pixabay.com</strong> or <strong>freesound.org</strong>. Saved to your library automatically.</p>
        </div>
      </div>

      <details style="margin-top:10px">
        <summary style="font-size:0.82rem;color:#999;cursor:pointer;user-select:none">Generative (no file needed) ▾</summary>
        <div class="radio-group" style="margin-top:8px">
          <label><input type="radio" name="nature_type" value="Birds only"> Birds only</label>
          <label><input type="radio" name="nature_type" value="Birds &amp; stream"> Birds &amp; stream</label>
          <label><input type="radio" name="nature_type" value="Rain"> Rain</label>
        </div>
      </details>

      <hr class="card-sep">

      <label for="durationSlider">Duration</label>
      <input type="range" id="durationSlider" name="nature_minutes" min="1.0" max="3.0" step="0.5" value="1.5">
      <div class="range-row"><span>1 min</span><span>1.5 min</span><span>3 min</span></div>

      <label for="natureVolSlider" style="margin-top:16px">Volume</label>
      <input type="range" id="natureVolSlider" name="nature_vol_db" min="-24" max="0" step="1" value="-14"
             oninput="document.getElementById('natureVolVal').textContent = this.value + ' dB'">
      <div class="range-row">
        <span>Quiet (−24 dB)</span>
        <span id="natureVolVal">−14 dB</span>
        <span>Full (0 dB)</span>
      </div>

      <hr class="card-sep">

      <label>Starts</label>
      <div class="radio-group">
        <label><input type="radio" name="start_mode" value="auto" checked> 🤖 Auto-detect "let's practice"</label>
        <label><input type="radio" name="start_mode" value="full"> ▶ Entire track (from 0:00)</label>
        <label><input type="radio" name="start_mode" value="manual"> 🕐 Specific time</label>
      </div>
      <p class="hint-text" id="autoDetectHint" style="margin-top:6px">Downloads a ~150 MB model on first use, then it's instant.</p>
      <div id="manualTsWrap" style="display:none; margin-top:10px">
        <input type="text" id="manualTs" name="manual_ts" placeholder="e.g.  3:45  or  225  (seconds)">
      </div>
      <!-- use_auto checkbox — hidden, synced by JS based on start_mode -->
      <input type="checkbox" name="use_auto" id="useAutoCheck" value="true" checked style="display:none">
    </div>

    <button type="submit" id="processBtn">Process meditation</button>

  </form>
</div>
<script>
(function () {
  /* ── Utility ─────────────────────────────────────────────────────────────── */
  function esc(s) {
    return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
  }
  function fmt(s) {
    s = Math.max(0, +s || 0);
    var m = Math.floor(s / 60), sec = s % 60;
    return m + ':' + (sec < 10 ? '0' : '') + sec.toFixed(1);
  }

  /* ── Nature type / source sync ───────────────────────────────────────────── */
  function syncType() {
    var val = document.querySelector('input[name="nature_type"]:checked').value;
    document.getElementById('customNaturePanel').style.display = (val === 'Custom') ? '' : 'none';
  }
  document.querySelectorAll('input[name="nature_type"]').forEach(function(r) { r.addEventListener('change', syncType); });
  syncType();

  /* ── Start mode sync ─────────────────────────────────────────────────────── */
  function syncStartMode() {
    var mode = document.querySelector('input[name="start_mode"]:checked').value;
    document.getElementById('autoDetectHint').style.display = (mode === 'auto') ? '' : 'none';
    document.getElementById('manualTsWrap').style.display   = (mode === 'manual') ? '' : 'none';
    var useAuto = document.getElementById('useAutoCheck');
    var manualTs = document.getElementById('manualTs');
    if (mode === 'auto') {
      useAuto.checked = true;
      if (manualTs) manualTs.value = '';
    } else if (mode === 'full') {
      useAuto.checked = false;
      if (manualTs) manualTs.value = '0:00';
    } else {
      useAuto.checked = false;
      // keep whatever the user typed
    }
  }
  document.querySelectorAll('input[name="start_mode"]').forEach(function(r) { r.addEventListener('change', syncStartMode); });
  syncStartMode();

  function syncSource() {
    var src = document.querySelector('input[name="nature_source"]:checked').value;
    document.getElementById('libraryPanel').style.display = src === 'library' ? '' : 'none';
    document.getElementById('uploadPanel').style.display  = src === 'upload'  ? '' : 'none';
    if (src === 'library') loadLibrary();
  }
  document.querySelectorAll('input[name="nature_source"]').forEach(function(r) { r.addEventListener('change', syncSource); });
  syncSource();

  function loadLibrary() {
    fetch('/library').then(function(r) { return r.json(); }).then(function(files) {
      var list = document.getElementById('libraryList');
      var hidden = document.getElementById('librarySelect');
      var hint = document.getElementById('libraryCount');
      if (files.length === 0) {
        list.innerHTML = '<p style="color:#999;font-size:0.9rem;padding:4px 0">No files yet — switch to "Upload new" to add your first track.</p>';
        hidden.value = ''; hint.textContent = ''; return;
      }
      var prev = hidden.value;
      if (files.indexOf(prev) === -1) prev = files[0];
      list.innerHTML = files.map(function(f) {
        var sel = f === prev ? ' lib-selected' : '';
        return '<div class="lib-file-row' + sel + '" data-file="' + esc(f) + '">' +
               '<span class="lib-fname" title="' + esc(f) + '">' + esc(f) + '</span>' +
               '<button type="button" class="lib-del" title="Delete from library">&#128465;</button></div>';
      }).join('');
      hidden.value = prev;
      hint.textContent = files.length + ' file' + (files.length !== 1 ? 's' : '') + ' in library  ·  click a row to select, 🗑 to delete';
      list.querySelectorAll('.lib-file-row').forEach(function(row) {
        row.addEventListener('click', function(e) {
          if (e.target.classList.contains('lib-del')) return;
          list.querySelectorAll('.lib-file-row').forEach(function(r) { r.classList.remove('lib-selected'); });
          row.classList.add('lib-selected');
          hidden.value = row.getAttribute('data-file');
        });
      });
      list.querySelectorAll('.lib-del').forEach(function(btn) {
        btn.addEventListener('click', function(e) {
          e.stopPropagation();
          var name = btn.closest('.lib-file-row').getAttribute('data-file');
          if (!confirm('Delete "' + name + '" from your library?')) return;
          fetch('/library/delete/' + encodeURIComponent(name), { method: 'POST' }).then(loadLibrary);
        });
      });
    }).catch(function() {
      document.getElementById('libraryList').innerHTML = '<p style="color:#a04040;font-size:0.9rem">Could not load library</p>';
    });
  }
  loadLibrary();

  /* ── Waveform player ─────────────────────────────────────────────────────── */
  var wfCtx = null, wfBuffer = null, wfSource = null;
  var wfPlaying = false, wfStartedAt = 0, wfOffset = 0, wfDuration = 0, wfRafId = null;
  var wfCuts = [], wfNextId = 1;
  var selStart = null, selEnd = null, isDragging = false;

  function wfPos() {
    if (!wfCtx) return wfOffset;
    if (wfPlaying) return Math.min(wfOffset + (wfCtx.currentTime - wfStartedAt), wfDuration);
    return wfOffset;
  }

  function wfDraw() {
    var canvas = document.getElementById('waveCanvas');
    if (!canvas) return;
    var dpr = window.devicePixelRatio || 1;
    var W = Math.round(canvas.offsetWidth * dpr);
    var H = Math.round(canvas.offsetHeight * dpr);
    if (canvas.width !== W || canvas.height !== H) { canvas.width = W; canvas.height = H; }
    var ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#f8f8f8'; ctx.fillRect(0, 0, W, H);
    if (!wfBuffer) return;

    /* cut regions */
    for (var ci = 0; ci < wfCuts.length; ci++) {
      var cc = wfCuts[ci];
      var cx1 = (cc.start_ms / 1000 / wfDuration) * W;
      var cx2 = (cc.end_ms   / 1000 / wfDuration) * W;
      ctx.fillStyle = 'rgba(200,50,50,0.13)'; ctx.fillRect(cx1, 0, cx2 - cx1, H);
      ctx.strokeStyle = 'rgba(200,50,50,0.5)'; ctx.lineWidth = 1;
      ctx.strokeRect(cx1 + 0.5, 0.5, cx2 - cx1 - 1, H - 1);
      ctx.save(); ctx.rect(cx1, 0, cx2 - cx1, H); ctx.clip();
      ctx.strokeStyle = 'rgba(200,50,50,0.2)';
      for (var xi = cx1 - H; xi < cx2 + H; xi += 8) { ctx.beginPath(); ctx.moveTo(xi,0); ctx.lineTo(xi+H,H); ctx.stroke(); }
      ctx.restore();
    }

    /* drag selection */
    if (isDragging && selStart !== null && selEnd !== null) {
      var xs = Math.min(selStart, selEnd) / wfDuration * W;
      var xe = Math.max(selStart, selEnd) / wfDuration * W;
      ctx.fillStyle = 'rgba(180,60,60,0.22)'; ctx.fillRect(xs, 0, xe - xs, H);
    }

    /* waveform */
    var data = wfBuffer.getChannelData(0);
    var step = Math.max(1, Math.floor(data.length / W));
    var mid = H / 2;
    ctx.strokeStyle = '#555'; ctx.lineWidth = 1; ctx.beginPath();
    for (var i = 0; i < W; i++) {
      var mn = 1.0, mx = -1.0;
      for (var j = 0; j < step; j++) { var d = data[i*step+j]||0; if(d<mn)mn=d; if(d>mx)mx=d; }
      ctx.moveTo(i+0.5, mid + mn*mid*0.90); ctx.lineTo(i+0.5, mid + mx*mid*0.90);
    }
    ctx.stroke();

    /* playhead */
    var ph = (wfPos() / wfDuration) * W;
    ctx.strokeStyle = '#c0392b'; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(ph, 0); ctx.lineTo(ph, H); ctx.stroke();
  }

  function wfRender() {
    wfDraw();
    document.getElementById('waveCurrentTime').textContent = fmt(wfPos());
    if (wfPlaying) wfRafId = requestAnimationFrame(wfRender);
  }

  function wfSeek(t) {
    var was = wfPlaying;
    if (wfPlaying) wfStopPlayback();
    wfOffset = Math.max(0, Math.min(+t||0, wfDuration));
    document.getElementById('waveCurrentTime').textContent = fmt(wfOffset);
    if (was) wfStartPlayback(); else wfDraw();
  }

  function wfStartPlayback() {
    if (!wfBuffer || wfPlaying) return;
    if (wfCtx.state === 'suspended') wfCtx.resume();
    wfSource = wfCtx.createBufferSource();
    wfSource.buffer = wfBuffer;
    wfSource.connect(wfCtx.destination);
    wfSource.onended = function() {
      if (wfPlaying) { wfOffset = 0; wfPlaying = false; cancelAnimationFrame(wfRafId); document.getElementById('playBtn').textContent = '▶ Play'; wfDraw(); }
    };
    wfSource.start(0, wfOffset);
    wfStartedAt = wfCtx.currentTime; wfPlaying = true;
    document.getElementById('playBtn').textContent = '⏸ Pause';
    wfRafId = requestAnimationFrame(wfRender);
  }

  function wfStopPlayback() {
    if (!wfPlaying) return;
    wfOffset = wfPos();
    if (wfSource) { try { wfSource.stop(); } catch(e){} wfSource = null; }
    wfPlaying = false; cancelAnimationFrame(wfRafId);
    document.getElementById('playBtn').textContent = '▶ Play';
  }

  function wfToggle() { if (wfPlaying) wfStopPlayback(); else wfStartPlayback(); wfDraw(); }

  function canvasSec(e) {
    var canvas = document.getElementById('waveCanvas');
    var rect = canvas.getBoundingClientRect();
    return ((e.clientX - rect.left) / rect.width) * wfDuration;
  }

  async function wfLoad(file) {
    try {
      wfCtx = new (window.AudioContext || window.webkitAudioContext)();
      var ab = await file.arrayBuffer();
      wfBuffer = await wfCtx.decodeAudioData(ab);
      wfDuration = wfBuffer.duration; wfOffset = 0; wfPlaying = false;
      document.getElementById('waveDuration').textContent = fmt(wfDuration);
      document.getElementById('waveCurrentTime').textContent = fmt(0);
      document.getElementById('waveformCard').style.display = '';
      document.getElementById('analysisPanel').style.display = 'none';
      document.getElementById('analysisPanel').innerHTML = '';
      setTimeout(wfDraw, 60);
    } catch(err) { console.warn('[waveform]', err); }
  }

  /* ── Cuts ────────────────────────────────────────────────────────────────── */
  function addCut(start_ms, end_ms, btnEl) {
    var id = 'cut_' + (wfNextId++);
    wfCuts.push({ id: id, start_ms: start_ms, end_ms: end_ms });
    saveCuts(); renderCuts(); wfDraw();
    if (btnEl) { btnEl.disabled = true; btnEl.textContent = '✓ Queued'; btnEl.style.opacity = '0.5'; }
  }
  window.wfRemoveCut = function(id) { wfCuts = wfCuts.filter(function(c){return c.id!==id;}); saveCuts(); renderCuts(); wfDraw(); };
  function saveCuts() { document.getElementById('cutsField').value = JSON.stringify(wfCuts.map(function(c){return{start_ms:c.start_ms,end_ms:c.end_ms};})); }
  function renderCuts() {
    var wrap = document.getElementById('cutList'), items = document.getElementById('cutItems');
    if (!wfCuts.length) { wrap.style.display = 'none'; return; }
    wrap.style.display = '';
    items.innerHTML = wfCuts.map(function(c) {
      return '<div class="cut-item"><span>&#9986; ' + fmt(c.start_ms/1000) + ' – ' + fmt(c.end_ms/1000) +
             '  (' + ((c.end_ms-c.start_ms)/1000).toFixed(1) + 's)</span>' +
             '<button type="button" onclick="wfRemoveCut(\\'' + c.id + '\\')">✕</button></div>';
    }).join('');
  }

  /* ── Analysis ────────────────────────────────────────────────────────────── */
  async function analyzeAudio() {
    var fi = document.querySelector('input[name="audio"]');
    if (!fi || !fi.files || !fi.files[0]) { alert('Please select an audio file first.'); return; }
    var btn = document.getElementById('analyzeBtn');
    btn.disabled = true; btn.textContent = '⏳ Analyzing…';
    var panel = document.getElementById('analysisPanel');
    panel.style.display = ''; panel.innerHTML = '<p style="color:#666;font-size:0.87rem;padding:4px 0">Analyzing — please wait…</p>';
    var fd = new FormData(); fd.append('audio', fi.files[0]);
    try {
      var resp = await fetch('/analyze', { method: 'POST', body: fd });
      var data = await resp.json();
      if (data.error) { panel.innerHTML = '<p style="color:#a04040;font-size:0.87rem">Error: ' + esc(data.error) + '</p>'; return; }
      var html = '';
      var tx = data.transcript;
      if (tx && tx.length) {
        html += '<p style="font-size:0.84rem;font-weight:600;color:#333;margin-bottom:5px">Transcript <span style="font-weight:400;color:#999">(tap word to seek)</span></p>';
        html += '<div class="transcript-box">';
        tx.forEach(function(w) { html += '<span onclick="wfSeek(' + w.start + ')" title="' + fmt(w.start) + '">' + esc(w.word) + ' </span>'; });
        html += '</div>';
      } else if (!data.whisper_available) {
        html += '<p style="font-size:0.82rem;color:#999;margin-bottom:10px">&#128161; Install <code>openai-whisper</code> for full word-level transcript and smarter error detection.</p>';
      }
      var suggs = data.suggestions || [];
      if (!suggs.length) {
        html += '<p style="font-size:0.87rem;color:#555">&#10003; No obvious issues detected' + (data.whisper_available ? '.' : ' (amplitude analysis only).') + '</p>';
      } else {
        html += '<p style="font-size:0.84rem;font-weight:600;color:#333;margin-bottom:8px">Suggested edits (' + suggs.length + '):</p>';
        suggs.forEach(function(s, i) {
          var bid = 'sb_' + i;
          html += '<div class="sugg-item"><div class="sugg-text"><div class="sugg-reason">' + esc(s.reason) + '</div>' +
                  '<div class="sugg-ts">' + fmt(s.start_ms/1000) + ' – ' + fmt(s.end_ms/1000) + '</div></div>' +
                  '<div class="sugg-actions">' +
                  '<button type="button" class="ctrl-btn" onclick="wfSeek(' + (s.start_ms/1000) + ')">&#9654;</button>' +
                  '<button type="button" class="ctrl-btn ctrl-btn-red" id="' + bid + '" onclick="wfAcceptSugg(' + s.start_ms + ',' + s.end_ms + ',\\'' + bid + '\\')">&#9986; Cut</button>' +
                  '</div></div>';
        });
      }
      panel.innerHTML = html;
    } catch(e) {
      panel.innerHTML = '<p style="color:#a04040;font-size:0.87rem">Analysis failed: ' + esc(e.message) + '</p>';
    } finally {
      btn.disabled = false; btn.textContent = '&#128269; Analyze for errors';
    }
  }

  window.wfAcceptSugg = function(s, e, bid) { addCut(s, e, document.getElementById(bid)); };
  window.wfSeek = wfSeek;

  /* ── Init ────────────────────────────────────────────────────────────────── */
  document.addEventListener('DOMContentLoaded', function() {
    var canvas = document.getElementById('waveCanvas');
    canvas.addEventListener('mousedown', function(e) {
      if (!wfBuffer) return; isDragging = true; selStart = canvasSec(e); selEnd = selStart; e.preventDefault();
    });
    canvas.addEventListener('mousemove', function(e) { if (!isDragging) return; selEnd = canvasSec(e); wfDraw(); });
    canvas.addEventListener('mouseup', function(e) {
      if (!isDragging) return; isDragging = false;
      var s = Math.min(selStart, selEnd), en = Math.max(selStart, selEnd);
      if (en - s < 0.25) { wfSeek(s); } else { addCut(Math.round(s*1000), Math.round(en*1000), null); }
      selStart = null; selEnd = null; wfDraw();
    });
    canvas.addEventListener('mouseleave', function() { if (isDragging) { isDragging = false; selStart = null; selEnd = null; wfDraw(); } });
    window.addEventListener('resize', wfDraw);

    document.getElementById('playBtn').addEventListener('click', wfToggle);
    document.getElementById('analyzeBtn').addEventListener('click', analyzeAudio);
    document.querySelector('input[name="audio"]').addEventListener('change', function() {
      if (this.files && this.files[0]) wfLoad(this.files[0]);
    });
  });
})();
</script>

<footer class="endor-footer">
  <a href="https://endor.global" target="_blank" rel="noopener">
    <span class="endor-mark">◈</span>
    <span class="endor-text">A gift for you — check in &amp; reset at <strong>endor.global</strong></span>
    <span class="endor-arrow">→</span>
  </a>
</footer>

<style>
.endor-footer {
  text-align: center;
  padding: 40px 16px 48px;
  margin-top: 12px;
}
.endor-footer a {
  display: inline-flex; align-items: center; gap: 10px;
  text-decoration: none;
  color: #999;
  font-size: 0.88rem;
  letter-spacing: 0.01em;
  border: 1px solid #ececec;
  border-radius: 999px;
  padding: 10px 22px;
  background: #fafafa;
  transition: color 0.2s, border-color 0.2s, background 0.2s;
}
.endor-footer a:hover {
  color: #111;
  border-color: #bbb;
  background: #fff;
}
.endor-mark {
  font-size: 1.0rem;
  opacity: 0.6;
  transition: opacity 0.2s;
}
.endor-footer a:hover .endor-mark {
  opacity: 1;
}
.endor-arrow {
  font-size: 0.85rem;
  opacity: 0.4;
  transition: transform 0.2s, opacity 0.2s;
}
.endor-footer a:hover .endor-arrow {
  opacity: 0.8;
  transform: translateX(3px);
}
</style>
</body>
</html>
"""


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    host = "0.0.0.0" if os.environ.get("PORT") else "127.0.0.1"
    url = f"http://127.0.0.1:{port}"
    print(f"\nMithril Meditation Editor → {url}\n")
    if host == "127.0.0.1":
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    app.run(host=host, port=port, debug=False)
