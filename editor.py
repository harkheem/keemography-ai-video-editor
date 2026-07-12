# =============================
# editor.py
# =============================

import os
import sys
import subprocess
import tempfile
import random
from typing import List, Dict, Optional, Union
import re
import librosa
import numpy as np

from transition import apply_transition, list_available_transitions

# Project root — used to resolve bundled assets regardless of CWD
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))


def _get_api_key(explicit_key: Optional[str] = None) -> Optional[str]:
    if explicit_key:
        return explicit_key
    return os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")


def normalize_clip_paths(clip_paths: list[Union[str, None]]) -> list[str]:
    """Filter to string paths. (The legacy web-page / Drive-folder scraping
    lived here; the FastAPI backend only ever passes server-side temp paths,
    so that code was dead — removed along with its bs4/googleapiclient deps.)"""
    return [p for p in clip_paths if isinstance(p, str)]


def detect_beats(audio_path: str) -> list[float]:
    """
    Detect beat times (in seconds) in an audio file using librosa.
    Returns a list of beat timestamps.
    """
    try:
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units='time')
        return beats.tolist()
    except Exception as e:
        print(f"⚠️ Beat detection failed: {repr(e)}")
        return []


def _music_energy_and_beats(audio_path: str):
    """One librosa pass over the music: beat grid + smoothed energy envelope.

    Returns (beat_times, envelope_times, envelope) where envelope is the
    0-1-normalized blend of RMS (60%) and onset strength (40%), smoothed to
    ~1s — the same "highlight" model analyze_music_for_trim uses. Both the
    beat-snapping and the energy-adaptive pacing read from this single
    analysis instead of loading the track twice.
    Returns ([], None, None) on failure.
    """
    try:
        from scipy.ndimage import gaussian_filter1d
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")

        hop = 512
        rms = librosa.feature.rms(y=y, hop_length=hop)[0]
        onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
        n = min(len(rms), len(onset))
        curve = (0.6 * (rms[:n] / (rms[:n].max() + 1e-9))
                 + 0.4 * (onset[:n] / (onset[:n].max() + 1e-9)))
        curve = gaussian_filter1d(curve, sigma=max(1, int(sr / hop)))
        cmin, cmax = float(curve.min()), float(curve.max())
        if cmax - cmin > 1e-6:
            curve = (curve - cmin) / (cmax - cmin)
        times = librosa.frames_to_time(np.arange(len(curve)), sr=sr, hop_length=hop)
        return list(beats), times, curve
    except Exception as e:
        print(f"⚠️ Music analysis failed: {repr(e)}")
        return [], None, None


def analyze_music_for_trim(audio_path: str, target_duration_sec: float = 45.0) -> dict:
    """
    Analyze a music file and suggest the best trim window using librosa.
    Blends RMS energy (60%) + onset strength (40%) into a highlight score,
    then finds the sliding window of target_duration_sec with the highest score.

    Returns:
        { duration, suggested_start, suggested_end, energy_peaks }
    Falls back to { duration, suggested_start=0, suggested_end=min(dur, target), energy_peaks=[] }
    on any error.
    """
    try:
        from scipy.ndimage import gaussian_filter1d
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        total_duration = float(librosa.get_duration(y=y, sr=sr))

        if total_duration <= target_duration_sec:
            # Track is shorter than target — no trimming needed; suggest full range
            return {
                "duration": round(total_duration, 2),
                "suggested_start": 0.0,
                "suggested_end": round(total_duration, 2),
                "energy_peaks": [],
            }

        hop_length = 512
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

        # Pad or trim to same length
        min_len = min(len(rms), len(onset_env))
        rms = rms[:min_len]
        onset_env = onset_env[:min_len]

        rms_norm = rms / (rms.max() + 1e-9)
        onset_norm = onset_env / (onset_env.max() + 1e-9)
        highlight = 0.6 * rms_norm + 0.4 * onset_norm

        # Smooth to avoid picking a single loud transient
        sigma = max(1, int(sr / hop_length))  # ~1 second of smoothing
        highlight = gaussian_filter1d(highlight, sigma=sigma)

        times = librosa.frames_to_time(np.arange(len(highlight)), sr=sr, hop_length=hop_length)
        window_frames = int(target_duration_sec * sr / hop_length)
        window_frames = min(window_frames, len(highlight) - 1)

        best_start_frame = 0
        best_score = -1.0
        for i in range(len(highlight) - window_frames):
            score = float(np.mean(highlight[i: i + window_frames]))
            if score > best_score:
                best_score = score
                best_start_frame = i

        suggested_start = float(times[best_start_frame])
        suggested_end = min(total_duration, suggested_start + target_duration_sec)

        # Top-5 energy peaks for optional frontend display
        peak_frames = np.argsort(highlight)[-5:][::-1].tolist()
        energy_peaks = sorted([
            round(float(times[min(f, len(times) - 1)]), 2) for f in peak_frames
        ])

        return {
            "duration": round(total_duration, 2),
            "suggested_start": round(suggested_start, 2),
            "suggested_end": round(suggested_end, 2),
            "energy_peaks": energy_peaks,
        }
    except Exception as e:
        print(f"⚠️ [MUSIC] Analysis failed for {os.path.basename(audio_path)}: {repr(e)}")
        try:
            from app_utils import probe_duration_seconds
            dur = probe_duration_seconds(audio_path)
        except Exception:
            dur = 0.0
        return {
            "duration": round(dur, 2),
            "suggested_start": 0.0,
            "suggested_end": round(min(dur, target_duration_sec), 2),
            "energy_peaks": [],
        }


# ── 4K proxy helpers ──────────────────────────────────────────────────────────

def _probe_resolution(path: str) -> tuple:
    """Returns (width, height) of the first video stream; (0, 0) on failure."""
    if not path or not os.path.exists(path):
        return (0, 0)
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=s=x:p=0",
        path,
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        parts = (res.stdout or "").strip().split("x")
        if len(parts) == 2:
            return (int(parts[0]), int(parts[1]))
    except Exception:
        pass
    return (0, 0)


def _is_4k(path: str) -> bool:
    """True when clip resolution is 4K or higher (≥ 3840 wide or ≥ 2160 tall)."""
    w, h = _probe_resolution(path)
    return w >= 3840 or h >= 2160


def _create_proxy(src_path: str, height: int = 1080) -> Optional[str]:
    """
    Downscale *src_path* to *height* lines for fast MoviePy processing.
    Tries VideoToolbox (Apple GPU) first, falls back to libx264 ultrafast.
    Returns a temp .mp4 path, or None on failure.
    """
    proxy = tempfile.NamedTemporaryFile(delete=False, suffix="_proxy.mp4")
    proxy_path = proxy.name
    proxy.close()

    def _try(cmd):
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return os.path.exists(proxy_path) and os.path.getsize(proxy_path) > 4096
        except subprocess.CalledProcessError:
            return False

    scale_filter = f"scale=-2:{height}:flags=lanczos"

    if _try(["ffmpeg", "-y", "-hwaccel", "videotoolbox", "-i", src_path,
             "-vf", scale_filter, "-c:v", "h264_videotoolbox", "-b:v", "8M",
             "-c:a", "aac", "-b:a", "192k", proxy_path]):
        return proxy_path

    if _try(["ffmpeg", "-y", "-i", src_path,
             "-vf", scale_filter, "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
             "-c:a", "aac", "-b:a", "192k", proxy_path]):
        return proxy_path

    try:
        os.remove(proxy_path)
    except Exception:
        pass
    return None


def _ffmpeg_4k_render(
    expanded_entries: list,   # list of (path, raw, meta) — path may be proxy
    trim_log: dict,           # {expanded_index: (start_sec, end_sec)} on raw/proxy
    transition_log: list,     # [{transition, duration, applied}] len = n_clips-1
    proxy_to_orig: dict,      # proxy_path → original_4k_path
    proxy_output_path: str,   # the proxy MP4 (its audio is muxed into 4K output)
    target_fps: int,
) -> str:
    """
    Re-render the final edit at the original 4K resolution using ffmpeg.

    1. Each clip is trimmed to exactly the window chosen during proxy editing
       (recorded in trim_log), but applied to the ORIGINAL 4K source file.
    2. All segments are scaled to a uniform resolution (that of the first 4K clip).
    3. xfade / concat transitions matching the proxy edit are applied.
    4. The audio track from the proxy output (perfectly mixed by MoviePy) is muxed
       in — so music, original-audio blending, etc. are all preserved.

    Falls back: caller catches RuntimeError and serves the proxy output instead.
    """
    _XFADE_MAP = {
        "crossfade":   "fade",
        "fadein":      "fadeblack",
        "fadeout":     "fadeblack",
        "slide_left":  "slideleft",
        "slide_right": "slideright",
        "slide_up":    "slideup",
        "slide_down":  "slidedown",
        "zoom_in":     "zoomin",
        "zoom_out":    "fadeblack",
    }

    n = len(expanded_entries)
    if n == 0:
        raise ValueError("No clips to render")

    # ── Build (orig_path, start_sec, end_sec) per segment ────────────────────
    segments = []
    for i, (path, raw, _m) in enumerate(expanded_entries):
        orig = proxy_to_orig.get(path, path)
        d = float(getattr(raw, "duration", 0.0) or 0.0)
        s, e = trim_log.get(i, (0.0, d))
        s = max(0.0, float(s))
        e = max(s + 0.1, float(e))
        segments.append((orig, s, e))

    # ── Determine output resolution ───────────────────────────────────────────
    out_w, out_h = 0, 0
    for orig, _s, _e in segments:
        if orig in proxy_to_orig.values():
            w, h = _probe_resolution(orig)
            if w > 0 and h > 0:
                out_w, out_h = w, h
                break
    if out_w == 0:
        out_w, out_h = 3840, 2160

    # ── ffmpeg inputs ─────────────────────────────────────────────────────────
    # Software decode only — hardware decode (videotoolbox) bypasses software
    # filter timebase normalization, causing xfade to see 1/1000000 vs 1/fps
    # mismatches on chained transitions. Hardware encode is still used below.
    cmd = ["ffmpeg", "-y"]
    for orig, _s, _e in segments:
        cmd += ["-i", orig]

    # ── filter_complex: trim + normalize each clip ────────────────────────────
    filter_parts = []
    clip_durs = []
    for i, (_orig, s, e) in enumerate(segments):
        d = e - s
        clip_durs.append(d)
        # Cover-crop (scale to fill + center-crop) rather than a plain
        # scale=W:H — sources can mix portrait/landscape and other aspect
        # ratios, and forcing an exact W:H would stretch/distort those.
        filter_parts.append(
            f"[{i}:v]trim=start={s:.4f}:end={e:.4f},"
            f"setpts=PTS-STARTPTS,"
            f"fps={target_fps},"
            f"settb=1/{target_fps},"
            f"scale={out_w}:{out_h}:flags=lanczos:force_original_aspect_ratio=increase,"
            f"crop={out_w}:{out_h},setsar=1,"
            f"format=yuv420p[v{i}]"
        )

    fc = ";".join(filter_parts)

    if n == 1:
        fc += ";[v0]copy[vout]"
    else:
        prev_label = "v0"
        cum_dur = clip_durs[0]
        for i in range(1, n):
            tlog = transition_log[i - 1] if i - 1 < len(transition_log) else {}
            tdur = float(tlog.get("duration", 0.0))
            ttype = tlog.get("transition", "crossfade")
            applied = bool(tlog.get("applied", False))
            xftype = _XFADE_MAP.get(ttype, "fade")
            out_label = f"xv{i}"
            if applied and tdur >= 0.1 and cum_dur > tdur + 0.05:
                offset = max(0.0, cum_dur - tdur)
                # After xfade, force timebase back to 1/fps so the next chained
                # xfade sees matching timebases on both inputs.
                fc += (
                    f";[{prev_label}][v{i}]xfade=transition={xftype}"
                    f":duration={tdur:.4f}:offset={offset:.4f},"
                    f"settb=1/{target_fps},setpts=PTS-STARTPTS[{out_label}]"
                )
                cum_dur += clip_durs[i] - tdur
            else:
                fc += f";[{prev_label}][v{i}]concat=n=2:v=1:a=0[{out_label}]"
                cum_dur += clip_durs[i]
            prev_label = out_label
        fc += f";[{prev_label}]copy[vout]"

    cmd += ["-filter_complex", fc, "-map", "[vout]"]

    # ── Video-only pass (VideoToolbox → libx264 fallback) ─────────────────────
    tmp_vid = tempfile.NamedTemporaryFile(delete=False, suffix="_4k_vid.mp4")
    tmp_vid.close()

    def _run_video_pass(extra_enc_args: list) -> subprocess.CompletedProcess:
        full_cmd = cmd + ["-an"] + extra_enc_args + ["-fps_mode", "cfr", tmp_vid.name]
        return subprocess.run(full_cmd, capture_output=True, text=True)

    result = _run_video_pass(["-c:v", "h264_videotoolbox", "-b:v", "50M"])
    if result.returncode != 0:
        print(f"[4K] VideoToolbox failed (rc={result.returncode}), retrying with libx264…")
        result = _run_video_pass(["-c:v", "libx264", "-preset", "slow", "-crf", "18"])
    if result.returncode != 0:
        try:
            os.remove(tmp_vid.name)
        except Exception:
            pass
        raise RuntimeError(f"4K video pass failed (both VideoToolbox and libx264):\n{(result.stderr or '')[-1200:]}")

    # ── Mux 4K video + proxy audio track ─────────────────────────────────────
    tmp_4k = tempfile.NamedTemporaryFile(delete=False, suffix="_4k_final.mp4")
    tmp_4k.close()
    mux = [
        "ffmpeg", "-y",
        "-i", tmp_vid.name,
        "-i", proxy_output_path,
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        tmp_4k.name,
    ]
    try:
        r = subprocess.run(mux, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"Audio mux failed:\n{(r.stderr or '')[-600:]}")
    finally:
        try:
            os.remove(tmp_vid.name)
        except Exception:
            pass

    print(f"✅ 4K re-render complete → {out_w}×{out_h}  {tmp_4k.name}")
    return tmp_4k.name


def generate_video(
    clip_paths: List[str],
    storyline: str,
    transition_duration: float = 0.3,
    tone: str = "Cinematic",
    target_duration_sec: Optional[int] = None,
    mix_original_audio: bool = False,
    show_opening_card: bool = True,
    custom_music_path: Optional[str] = None,
    music_start_sec: float = 0.0,
    music_end_sec: Optional[float] = None,
    clip_metadata: Optional[Dict[str, Dict]] = None,  # keyed by path; from scoring
) -> str:
    """
    Builds a final video by stitching clips with adaptive transitions,
    optional title card, and background music.

    This version NEVER over-trims short clips and provides better
    error details if no clips can be opened.
    """

    from moviepy import (
        VideoFileClip,
        concatenate_videoclips,
        CompositeVideoClip,
        CompositeAudioClip,
        concatenate_audioclips,
        TextClip,
        AudioFileClip,
    )

    temp_files = []

    # ── Tone profile ─────────────────────────────────────────────────────────
    # All per-tone knobs live here so the rest of the function just reads them.
    _tone_key = (tone or "cinematic").strip().lower()
    _TONE_PROFILES = {
        #              fps  trim_mult  tdur_mult  motion_w  audio_w  music_vol
        "energetic":  (30,   0.65,      0.45,      0.82,     0.12,    0.40),
        "epic":       (30,   0.85,      0.70,      0.72,     0.18,    0.35),
        "cinematic":  (24,   1.00,      1.00,      0.60,     0.30,    0.22),
        "sentimental":(24,   1.20,      1.40,      0.35,     0.55,    0.18),
        "calm":       (24,   1.35,      1.60,      0.30,     0.60,    0.15),
    }
    _fps, _trim_mult, _tdur_mult, _motion_w, _audio_w, _music_vol = \
        _TONE_PROFILES.get(_tone_key, _TONE_PROFILES["cinematic"])
    # transition type bias pools per tone
    _TONE_TRANSITIONS = {
        "energetic":   ["slide_left", "slide_right", "zoom_in", "slide_up"],
        "epic":        ["zoom_in", "zoom_out", "slide_left", "slide_right"],
        "cinematic":   ["crossfade", "fadein", "zoom_in"],
        "sentimental": ["crossfade", "fadein", "fadeout"],
        "calm":        ["crossfade", "fadein"],
    }
    _tone_transition_pool = _TONE_TRANSITIONS.get(_tone_key, list_available_transitions())
    # ─────────────────────────────────────────────────────────────────────────

    # ── Storyline signal ─────────────────────────────────────────────────────
    # Extract keywords from the user's stated storyline. These are used in two
    # ways: (1) scaling the target duration of matching clips so they breathe
    # more on screen; (2) biasing the content window search toward the
    # GPT-4o-identified peak moment when the clip description overlaps the story.
    def _extract_keywords(text: str) -> set:
        stop = {
            "the", "a", "an", "and", "or", "but", "to", "of", "in", "on",
            "for", "with", "is", "are", "was", "were", "be", "been", "being",
            "that", "this", "it", "as", "at", "by", "from", "i", "you",
            "we", "they", "he", "she", "them", "our", "your", "its",
        }
        tokens = re.findall(r"[a-zA-Z0-9']+", (text or "").lower())
        return {t for t in tokens if len(t) >= 4 and t not in stop}

    _story_keywords = _extract_keywords(storyline or "")

    def _story_weight(path: str) -> float:
        """0.0-1.0: how much this clip's description overlaps with the storyline."""
        if not _story_keywords:
            return 0.0
        meta = (_meta if isinstance(clip_metadata, dict) else {}).get(path, {})
        description = str(meta.get("description", "") or "")
        desc_tokens = _extract_keywords(description)
        if not desc_tokens:
            return 0.0
        overlap = len(desc_tokens & _story_keywords)
        return min(1.0, overlap / max(1, len(_story_keywords)))
    # ─────────────────────────────────────────────────────────────────────────

    MIN_KEEP_SEC = 0.40
    IDEAL_TRANSITION = max(0.15, float(transition_duration)) * _tdur_mult

    # ── Music analysis (single pass) ─────────────────────────────────────────
    # Beats + energy envelope feed two downstream systems: beat-snapped cut
    # boundaries and energy-adaptive pacing. Analyze the track exactly once.
    _music_beats_raw: list = []
    _energy_times = None
    _energy_curve = None
    if custom_music_path and os.path.exists(custom_music_path):
        _music_beats_raw, _energy_times, _energy_curve = _music_energy_and_beats(custom_music_path)

    def _energy_at_video_t(t: float) -> Optional[float]:
        """Music energy (0-1) at video time t, honouring the music start offset
        and looping when the track is shorter than the video."""
        if _energy_curve is None or _energy_times is None or len(_energy_curve) == 0:
            return None
        track_end = float(_energy_times[-1]) or 1.0
        mt = float(music_start_sec or 0.0) + max(0.0, float(t))
        if mt > track_end:
            mt = mt % track_end
        idx = int(np.searchsorted(_energy_times, mt))
        idx = min(max(idx, 0), len(_energy_curve) - 1)
        return float(_energy_curve[idx])

    def apply_fadein(clip, duration: float):
        if not duration or duration <= 0:
            return clip
        if hasattr(clip, "fadein"):
            return clip.fadein(duration)
        from moviepy import vfx
        return clip.with_effects([vfx.FadeIn(duration)])

    def apply_fadeout(clip, duration: float):
        if not duration or duration <= 0:
            return clip
        if hasattr(clip, "fadeout"):
            return clip.fadeout(duration)
        from moviepy import vfx
        return clip.with_effects([vfx.FadeOut(duration)])

    def apply_end(clip, end_time: float):
        if end_time is None:
            return clip
        if hasattr(clip, "set_end"):
            return clip.set_end(end_time)
        if hasattr(clip, "with_end"):
            return clip.with_end(end_time)
        return clip

    def apply_without_mask(clip):
        if clip is None:
            return clip
        if hasattr(clip, "without_mask"):
            return clip.without_mask()
        if hasattr(clip, "set_mask"):
            return clip.set_mask(None)
        try:
            clip.mask = None
        except Exception:
            pass
        return clip

    def apply_cover_resize(clip, target_w: int, target_h: int):
        """Scale + center-crop *clip* to exactly (target_w, target_h) without
        distorting its aspect ratio (like CSS `object-fit: cover`)."""
        if clip is None:
            return clip
        w, h = clip.size
        if w == target_w and h == target_h:
            return clip
        scale = max(target_w / w, target_h / h)
        new_w, new_h = max(target_w, round(w * scale)), max(target_h, round(h * scale))
        resized = clip.resized((new_w, new_h)) if hasattr(clip, "resized") else clip.resize((new_w, new_h))
        x1 = max(0, (new_w - target_w) // 2)
        y1 = max(0, (new_h - target_h) // 2)
        if hasattr(resized, "cropped"):
            return resized.cropped(x1=x1, y1=y1, width=target_w, height=target_h)
        return resized.crop(x1=x1, y1=y1, width=target_w, height=target_h)

    def apply_audio_volume(audio_clip, vol: float):
        if audio_clip is None:
            return None
        if hasattr(audio_clip, "volumex"):
            return audio_clip.volumex(vol)
        if hasattr(audio_clip, "with_volume_scaled"):
            return audio_clip.with_volume_scaled(vol)
        return audio_clip

    def apply_audio_subclip(audio_clip, start_t: float, end_t: float):
        if audio_clip is None:
            return None
        if hasattr(audio_clip, "subclipped"):
            return audio_clip.subclipped(start_t, end_t)
        if hasattr(audio_clip, "subclip"):
            return audio_clip.subclip(start_t, end_t)
        return audio_clip

    def apply_set_audio(video_clip, audio_clip):
        if video_clip is None:
            return None
        if hasattr(video_clip, "set_audio"):
            return video_clip.set_audio(audio_clip)
        if hasattr(video_clip, "with_audio"):
            return video_clip.with_audio(audio_clip)
        return video_clip

    def normalize_audio_for_mix(
        input_path: str,
        start_sec: float = 0.0,
        end_sec: Optional[float] = None,
    ) -> Optional[str]:
        if not input_path or not os.path.exists(input_path):
            return None
        out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        cmd = ["ffmpeg", "-y"]
        # Seek before input for fast seeking (re-encode anyway so quality is fine)
        if start_sec and start_sec > 0.0:
            cmd += ["-ss", str(round(start_sec, 3))]
        cmd += ["-i", input_path]
        if end_sec is not None and end_sec > (start_sec or 0.0):
            cmd += ["-t", str(round(end_sec - (start_sec or 0.0), 3))]
        cmd += ["-vn", "-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le", out_path]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            temp_files.append(out_path)
            return out_path
        except Exception as e:
            print(f"⚠️ Audio normalization failed for {input_path}: {repr(e)}")
            try:
                os.remove(out_path)
            except Exception:
                pass
            return None

    def apply_subclip(clip, start_t: float, end_t: float):
        start_t = max(0.0, float(start_t))
        end_t = max(start_t + 0.05, float(end_t))
        if hasattr(clip, "subclipped"):
            return clip.subclipped(start_t, end_t)
        if hasattr(clip, "subclip"):
            return clip.subclip(start_t, end_t)
        return clip

    def enforce_target_duration(video_clip, requested_sec: Optional[int]):
        if not requested_sec:
            return video_clip

        requested = float(max(1, int(requested_sec)))
        current = float(getattr(video_clip, "duration", 0.0) or 0.0)
        if current <= 0.05:
            return video_clip

        if current > requested + 0.02:
            return apply_subclip(video_clip, 0.0, requested)

        return video_clip

    def apply_smart_trim(
        clip,
        idx: int,
        total: int,
        alloc_sec: float,           # pre-computed weighted budget for this clip
        vision_trim_start: Optional[float] = None,
        vision_trim_end: Optional[float] = None,
        best_moment_sec: Optional[float] = None,
        story_weight: float = 0.0,
    ):
        duration = float(getattr(clip, "duration", 0.0) or 0.0)
        if duration <= 0.0:
            return clip

        # Clamp allocation to what the clip actually has
        target = min(duration, max(1.5, float(alloc_sec)))

        # Keep the whole clip only when the allocation IS the whole clip —
        # the old +0.35s slop leaked up to 0.35s per clip past the budget.
        if duration <= target + 0.05:
            _trim_log[idx] = (0.0, duration)
            return clip

        # ── VISION-GUIDED TRIM (primary path) ─────────────────────────────
        # Clamp GPT-4o's trim recommendations to actual clip bounds BEFORE use.
        # GPT-4o sometimes hallucinates timestamps beyond the real clip length.
        if vision_trim_start is not None and vision_trim_end is not None:
            vts = max(0.0, min(duration, float(vision_trim_start)))
            vte = max(0.0, min(duration, float(vision_trim_end)))
            v_window = vte - vts
            if v_window >= target * 0.5:
                if v_window > target:
                    # Window longer than budget — centre-crop to exactly target
                    center = (vts + vte) / 2.0
                    half   = target / 2.0
                    vts = max(0.0, center - half)
                    vte = min(duration, vts + target)
                elif v_window < target - 0.05:
                    # Window shorter than budget — expand around the window to
                    # fill the full allocation without leaving the clip bounds
                    extra = target - v_window
                    vts = max(0.0, vts - extra / 2.0)
                    vte = min(duration, vts + target)
                    vts = max(0.0, vte - target)  # re-anchor if vte hit boundary
                _trim_log[idx] = (vts, vte)
                return apply_without_mask(apply_subclip(clip, vts, vte))

        # ── BEST-MOMENT ANCHOR (secondary path) ───────────────────────────
        # No valid window but we have the peak frame timestamp.
        # Centre the heuristic search window around it instead of scanning all.
        if best_moment_sec is not None:
            bm = float(best_moment_sec)
            bm = max(0.0, min(duration - target, bm - target / 2.0))
            bm_end = min(duration, bm + target)
            if bm_end - bm >= target * 0.75:
                _trim_log[idx] = (bm, bm_end)
                return apply_without_mask(apply_subclip(clip, bm, bm_end))

        def _content_window_start() -> Optional[float]:
            # Pick the most informative window using lightweight motion + audio activity.
            max_start_local = max(0.0, duration - target)
            if max_start_local <= 0.01:
                return 0.0

            # Use a coarse scan — 3 sample frames per candidate is enough to
            # distinguish active from static windows. Fewer frames = less RAM.
            step = max(0.8, min(2.0, target / 3.0))
            candidates = np.arange(0.0, max_start_local + 1e-6, step)
            sample_count = int(max(3, min(5, round(target * 0.8))))
            offsets = np.linspace(0.0, max(0.05, target - 0.05), sample_count)

            best_score = -1.0
            best_start_local = None

            for cand_start in candidates:
                motion_vals = []
                audio_vals = []
                for off in offsets:
                    t1 = min(duration - 0.06, float(cand_start + off))
                    t2 = min(duration - 0.01, t1 + 0.05)
                    if t2 <= t1:
                        continue
                    try:
                        f1 = np.asarray(clip.get_frame(t1), dtype=np.float32)
                        f2 = np.asarray(clip.get_frame(t2), dtype=np.float32)
                        if f1.ndim == 3:
                            f1 = np.mean(f1, axis=2)
                        if f2.ndim == 3:
                            f2 = np.mean(f2, axis=2)
                        motion = float(np.mean(np.abs(f2 - f1)) / 255.0)
                        motion_vals.append(motion)
                    except Exception:
                        pass

                    try:
                        if getattr(clip, "audio", None) is not None:
                            sample = np.asarray(clip.audio.get_frame(t1), dtype=np.float32)
                            audio_vals.append(float(np.mean(np.abs(sample))))
                    except Exception:
                        pass

                if not motion_vals and not audio_vals:
                    continue

                motion_score = float(np.mean(motion_vals)) if motion_vals else 0.0
                audio_score = float(np.mean(audio_vals)) if audio_vals else 0.0

                # Keep tiny preference away from dead starts/ends.
                center = cand_start + (target * 0.5)
                center_norm = center / max(0.001, duration)
                center_bonus = 1.0 - abs(center_norm - 0.5)

                # Story proximity bonus: if a peak moment is known and this
                # clip is story-relevant, prefer windows that include it.
                story_bonus = 0.0
                if story_weight > 0.05 and best_moment_sec is not None:
                    win_start = float(cand_start)
                    win_end   = win_start + target
                    dist = 0.0 if win_start <= best_moment_sec <= win_end \
                           else min(abs(best_moment_sec - win_start),
                                    abs(best_moment_sec - win_end))
                    story_bonus = story_weight * max(0.0, 1.0 - dist / max(1.0, target))

                total_score = (_motion_w * motion_score) + (_audio_w * audio_score) + (0.08 * center_bonus) + (0.15 * story_bonus)

                if total_score > best_score:
                    best_score = total_score
                    best_start_local = float(cand_start)

            return best_start_local

        # First try true content-aware cutting.
        try:
            smart_start = _content_window_start()
            if smart_start is not None:
                smart_end = min(duration, smart_start + target)
                smart_clip = apply_subclip(clip, smart_start, smart_end)
                _trim_log[idx] = (smart_start, smart_end)
                return apply_without_mask(smart_clip)
        except Exception:
            pass

        max_start = max(0.0, duration - target)
        if total <= 1:
            start_t = max_start * 0.25
        elif idx == 0:
            # Opening clip: bias to earlier moment.
            start_t = max_start * 0.12
        elif idx == total - 1:
            # Closing clip: bias to later moment.
            start_t = max_start * 0.72
        else:
            # Middle clips: move through source progressively.
            progression = idx / max(1, total - 1)
            start_t = max_start * (0.22 + 0.46 * progression)

        end_t = min(duration, start_t + target)
        trimmed = apply_subclip(clip, start_t, end_t)
        _trim_log[idx] = (start_t, end_t)
        return apply_without_mask(trimmed)

    _load_errors: dict = {}  # path → error string for diagnostics

    def load_clip(path):
        # First attempt: normal load (with audio)
        try:
            clip = VideoFileClip(path)
            if not clip.duration or clip.duration < MIN_KEEP_SEC:
                clip.close()
                _load_errors[path] = f"duration too short ({getattr(clip, 'duration', None)}s)"
                return None
            return clip
        except Exception as _exc:
            print(f"⚠️ [load_clip] Normal load failed for {os.path.basename(path)}: {repr(_exc)}")
            _load_errors[path] = repr(_exc)

        # Fallback: try without audio (handles broken/incompatible audio streams)
        try:
            clip = VideoFileClip(path, audio=False)
            if not clip.duration or clip.duration < MIN_KEEP_SEC:
                clip.close()
                _load_errors[path] = f"duration too short ({getattr(clip, 'duration', None)}s)"
                return None
            print(f"[load_clip] Loaded WITHOUT audio: {os.path.basename(path)}")
            return clip
        except Exception as _exc2:
            print(f"⚠️ [load_clip] audio=False fallback also failed for {os.path.basename(path)}: {repr(_exc2)}")
            _load_errors[path] = repr(_exc2)
            return None

    import gc

    # Normalize clip paths
    clip_paths = normalize_clip_paths(clip_paths)

    # ── Proxy workflow for 4K sources ────────────────────────────────────────
    # Any clip at 4K (≥ 3840 wide) is downscaled to a 1080p proxy so MoviePy
    # works on ~4× less data per frame.  After the proxy edit, we replay the
    # exact trim windows on the original 4K files in one fast ffmpeg pass.
    _proxy_to_orig: dict = {}   # proxy_path → original_4k_path
    _proxied_paths: list = []
    for _p in clip_paths:
        if _is_4k(_p):
            _px = _create_proxy(_p)
            if _px:
                _proxy_to_orig[_px] = _p
                _proxied_paths.append(_px)
                temp_files.append(_px)   # cleaned up in finally
                print(f"[PROXY] 4K→1080p proxy created for {os.path.basename(_p)}")
            else:
                print(f"[PROXY] Could not create proxy for {os.path.basename(_p)}, using original")
                _proxied_paths.append(_p)
        else:
            _proxied_paths.append(_p)
    clip_paths = _proxied_paths
    has_4k = bool(_proxy_to_orig)

    # Logs filled during editing — used by the 4K re-render pass
    _trim_log: dict = {}       # expanded_index → (start_sec, end_sec) on proxy/raw
    _transition_log: list = [] # one entry per consecutive clip pair

    # ── Load + trim one clip at a time ──────────────────────────────────────
    # Loading all clips in parallel peaks at (n_clips × clip_RAM). Instead we
    # load one, trim it immediately (so the full source is no longer needed),
    # close the original, and only keep the small trimmed version. Peak RAM
    # stays at ~1 clip rather than all clips simultaneously.
    _meta = clip_metadata or {}
    # Mirror metadata from 4K originals to their proxies so scoring signals carry over
    for _px_path, _orig_path in _proxy_to_orig.items():
        if _orig_path in _meta:
            _meta[_px_path] = _meta[_orig_path]
    clips: list = []
    load_failures: list = []
    _deferred_close: list = []  # raw clips to close AFTER write_videofile
    n_paths = len(clip_paths)

    # ── Pre-load all raws ────────────────────────────────────────────────────
    _raws: list = []
    for path in clip_paths:
        raw = load_clip(path)
        if raw is None:
            load_failures.append(path)
            _raws.append(None)
        else:
            _raws.append(raw)

    # ── Orientation / resolution normalization ───────────────────────────────
    # Uploaded clips routinely mix portrait phone footage with landscape
    # footage, and low-res clips with high-res ones. Without a common canvas,
    # concatenate_videoclips falls back to "compose" mode: every mismatched
    # clip gets letterboxed onto a canvas sized to the largest clip, so the
    # edit visibly jumps between full-frame and pillarboxed/letterboxed shots.
    # Normalize every clip up front to one shared canvas — chosen from the
    # majority orientation among the clips actually in this edit — using a
    # cover-crop (scale to fill + center-crop) so nothing is stretched.
    _sized = [r.size for r in _raws if r is not None]
    if _sized:
        _landscape = [s for s in _sized if s[0] >= s[1]]
        _portrait = [s for s in _sized if s[0] < s[1]]
        _majority = _landscape if len(_landscape) >= len(_portrait) else _portrait
        _best_w, _best_h = max(_majority, key=lambda s: s[0] * s[1])
        _scale_cap = min(1.0, 1920 / max(_best_w, _best_h))
        _canvas_w = max(2, int(round(_best_w * _scale_cap / 2) * 2))
        _canvas_h = max(2, int(round(_best_h * _scale_cap / 2) * 2))
        if any(r is not None and tuple(r.size) != (_canvas_w, _canvas_h) for r in _raws):
            print(f"[CANVAS] Normalizing clips to {_canvas_w}x{_canvas_h} "
                  f"(mixed orientation/resolution detected across uploads)")
            _raws = [
                apply_cover_resize(r, _canvas_w, _canvas_h) if r is not None else None
                for r in _raws
            ]

    # ── Weighted screen-time allocation ─────────────────────────────────────
    # Each clip earns screen time in proportion to how much it deserves,
    # based on GPT-4o metadata — not equal fair share.
    #
    # Weight dimensions:
    #   narrative_role  — payoff clips get the most time, broll the least
    #   visual_score    — higher quality = more time (scaled to avoid starving)
    #   emotion         — emotionally heavy clips need longer to land
    #   story_weight    — overlap with user storyline = extra budget
    #   _trim_mult      — tone compression: caps per-clip ceiling (energetic = shorter shots)
    _ROLE_WEIGHT = {
        "hook":        1.20,   # sets the tone — short but visible
        "payoff":      1.65,   # climax — deserves the most time
        "turn":        1.15,   # story pivot
        "development": 0.90,   # connective tissue
        "broll":       0.55,   # pure context — keep short
    }
    _EMOTION_WEIGHT = {
        "dramatic":  1.30,
        "sad":       1.25,   # needs time to land emotionally
        "inspiring": 1.20,
        "exciting":  1.10,
        "tense":     1.10,
        "happy":     1.00,
        "calm":      0.85,
        "neutral":   0.80,
    }

    desired_total = float(target_duration_sec) if target_duration_sec else 45.0
    _valid_pairs = [(p, r) for p, r in zip(clip_paths, _raws) if r is not None]

    # ── Multi-segment expansion ──────────────────────────────────────────────
    # If scoring identified 2-3 distinct good windows inside a single source
    # clip, we create one independent timeline entry per segment.  Each entry
    # carries its own trim window and narrative metadata so the allocation
    # engine treats them as independent clips competing for screen time.
    #
    # The raw VideoFileClip object is intentionally shared across segments from
    # the same source — MoviePy's lazy pipeline means we don't read the file
    # twice.  _deferred_close deduplicates by id() to avoid double-closing.
    _expanded: list = []   # (path, raw, per_segment_meta_dict)
    for path, raw in _valid_pairs:
        base_m = _meta.get(path, {})
        segments = base_m.get("segments") or []
        if len(segments) >= 2:
            print(f"[SEGMENTS] {os.path.basename(path)} → {len(segments)} windows")
            for _seg_i, seg in enumerate(segments):
                s_start = seg.get("start_sec")
                s_end   = seg.get("end_sec")
                if s_start is None or s_end is None:
                    continue
                seg_meta = {
                    **base_m,
                    "trim_start_sec":  float(s_start),
                    "trim_end_sec":    float(s_end),
                    "best_moment_sec": seg.get("best_moment_sec", (float(s_start) + float(s_end)) / 2),
                    "narrative_role":  seg.get("narrative_role", base_m.get("narrative_role", "development")),
                    "emotion":         seg.get("emotion",        base_m.get("emotion",        "neutral")),
                    "visual_score":    float(seg.get("visual_score", base_m.get("visual_score", 0.5))),
                    "description":     seg.get("description",    base_m.get("description",    "")),
                    "segments":        [],  # don't recurse
                    # arc_transition was chosen for the cut from the PRECEDING
                    # TRANSCRIPT-level clip into this one — only the first
                    # segment's incoming cut is that transition. Later segments'
                    # incoming cut is between two segments of the SAME source
                    # clip, which the arc pass never reasoned about, so let the
                    # heuristic picker handle those instead.
                    "arc_transition":  base_m.get("arc_transition") if _seg_i == 0 else None,
                }
                _expanded.append((path, raw, seg_meta))
        else:
            _expanded.append((path, raw, base_m))

    if not _expanded:
        _expanded = [(p, r, _meta.get(p, {})) for p, r in _valid_pairs]
    # ────────────────────────────────────────────────────────────────────────

    # Raw unnormalized weight per segment entry
    _raw_weights: list[float] = []
    for path, raw, m in _expanded:
        # The scoring arc pass (GPT-4o, sees every clip + the storyline
        # together) can assign its own screen_time_weight per clip — its
        # editorial judgment of what THIS clip deserves in THIS edit, rather
        # than a fixed per-role/emotion table. Use it when present; otherwise
        # fall back to the heuristic formula below (arc pass didn't run, e.g.
        # no OpenAI key).
        _arc_w = m.get("arc_weight")
        if _arc_w is not None:
            _raw_weights.append(max(0.01, float(_arc_w)))
            continue

        role = (m.get("narrative_role") or "development").lower()
        emo  = (m.get("emotion")        or "neutral").lower()
        vs   = float(m.get("visual_score") or 0.5)
        sw   = _story_weight(path)

        w = (
            _ROLE_WEIGHT.get(role, 0.90)
            * (0.35 + 0.65 * max(0.0, min(1.0, vs)))  # quality, not starving low scores
            * _EMOTION_WEIGHT.get(emo, 0.90)
            * (1.0 + 0.25 * sw)                        # story overlap bonus
            # _trim_mult intentionally NOT here — it cancels in normalization;
            # instead applied to per-clip ceilings below so it actually has effect
        )
        _raw_weights.append(max(0.01, w))

    # Normalise weights (for iterative saturation below)
    _total_w = sum(_raw_weights) or 1.0

    # Per-clip available window
    # Always use the FULL raw clip duration as the ceiling — GPT-4o's narrow
    # trim windows are HINTS for WHERE to cut, not hard limits on how much footage
    # is available. Using window sizes as ceilings caused short outputs when GPT-4o
    # hallucinated timestamps beyond the actual clip length.
    def _entry_avail(raw_clip, m: dict) -> float:
        return float(getattr(raw_clip, "duration", 0.0) or 0.0)

    _clip_durs = [_entry_avail(r, m) for _, r, m in _expanded]
    _N = len(_expanded)
    FLOOR = 1.5

    # ── Drop overflow entries ────────────────────────────────────────────────
    # The editor is the final authority on clip count: if even at the per-clip
    # FLOOR the entries can't fit the target (scoring over-selected, or
    # continuity repair inserted extras), drop the lowest-weight entries until
    # the budget is feasible. The 5% grace absorbs transition overlap headroom.
    while _N > 1 and _N * FLOOR > desired_total * 1.05:
        _drop = min(range(_N), key=lambda i: _raw_weights[i])
        print(f"[ALLOC] Budget overflow — dropping lowest-weight entry "
              f"{os.path.basename(_expanded[_drop][0])} (w={_raw_weights[_drop]:.2f})")
        for _lst in (_expanded, _raw_weights, _clip_durs):
            _lst.pop(_drop)
        _N -= 1

    # CEIL_RATIO must be >= 1/N so that N clips can collectively fill 100% of the
    # budget. With 2 clips, 0.45*2=0.90 — always 10% short. Use max(0.45, 1/N).
    CEIL_RATIO = max(0.45, 1.0 / max(1, _N))

    # Tone-modulated per-clip ceiling: _trim_mult controls how long individual shots
    # can run — energetic (0.65) = shorter shots/more cuts, calm (1.35) = longer shots.
    # Floor at 50% of equal share so budget holes don't force short outputs.
    _tone_ceil = max(
        (desired_total / max(1, _N)) * _trim_mult,
        (desired_total / max(1, _N)) * 0.50,
    )

    # ── Ceiling relaxation stages ────────────────────────────────────────────
    # Preferred pacing first; when the footage can't fill the budget under a
    # stage's caps, relax to the next stage. Hitting the requested duration
    # beats preserving tone pacing; only genuine footage shortage loses.
    #   stage 0: footage ∧ monopoly cap ∧ tone cap
    #   stage 1: footage ∧ monopoly cap ∧ tone cap scaled up just enough
    #   stage 2: footage ∧ monopoly cap
    #   stage 3: footage only
    def _ceilings_for(stage: int, budget: float) -> list:
        if stage == 0:
            caps = [min(_clip_durs[i], desired_total * CEIL_RATIO, _tone_ceil) for i in range(_N)]
        elif stage == 1:
            # Proportional relaxation: grow the tone cap by exactly the deficit
            # ratio so pacing character survives small budget shortfalls.
            base = [min(_clip_durs[i], desired_total * CEIL_RATIO, _tone_ceil) for i in range(_N)]
            ratio = (budget / max(sum(max(FLOOR, c) for c in base), 1e-6)) * 1.05
            caps = [min(_clip_durs[i], desired_total * CEIL_RATIO, _tone_ceil * max(1.0, ratio))
                    for i in range(_N)]
        elif stage == 2:
            caps = [min(_clip_durs[i], desired_total * CEIL_RATIO) for i in range(_N)]
        else:
            caps = [_clip_durs[i] for i in range(_N)]
        return [max(FLOOR, c) for c in caps]

    def _waterfill(budget: float, ceilings: list) -> list:
        """Distribute `budget` across entries proportional to weight, each
        allocation clamped to [FLOOR, ceiling_i], with total equal to
        min(budget, sum(ceilings)) exactly — no leak in either direction."""
        alloc = [0.0] * _N
        fixed = [False] * _N
        rem = budget
        for _ in range(_N + 2):
            free = [i for i in range(_N) if not fixed[i]]
            if not free:
                break
            wsum = sum(_raw_weights[i] for i in free) or 1e-9
            pinned = False
            for i in free:
                t = rem * _raw_weights[i] / wsum
                if t >= ceilings[i] - 1e-6:
                    alloc[i], fixed[i] = ceilings[i], True
                    rem -= ceilings[i]
                    pinned = True
                elif t <= FLOOR + 1e-6:
                    alloc[i], fixed[i] = FLOOR, True
                    rem -= FLOOR
                    pinned = True
            if not pinned:
                # Every remaining share is strictly inside (FLOOR, ceiling) —
                # assign proportionally; total is exact by construction.
                wsum = sum(_raw_weights[i] for i in free) or 1e-9
                for i in free:
                    alloc[i] = rem * _raw_weights[i] / wsum
                break
        return alloc

    # ── Fixed-point allocation: budget = target + expected transition overlap ─
    # Applied transitions overlap the incoming clip's head, consuming ~tdur of
    # timeline per cut, so clips must collectively be allocated target + Σtdur.
    # tdur depends on the allocations themselves (20% cap on short clips), so
    # iterate to a fixed point — converges in 2-3 passes.
    def _expected_overlap(allocs: list) -> float:
        total = 0.0
        prev_d = allocs[0] if allocs else 0.0
        for i in range(1, len(allocs)):
            safe = min(IDEAL_TRANSITION, 0.20 * min(prev_d, allocs[i]))
            tdur = safe if safe >= 0.12 else 0.0
            total += tdur
            prev_d = allocs[i] - tdur
        return total

    _overlap_est = 0.0
    _alloc_final = [FLOOR] * _N
    _ceilings = _ceilings_for(3, desired_total)
    for _fp in range(4):
        _budget = max(desired_total + _overlap_est, _N * FLOOR)
        for _stage in range(4):
            _ceilings = _ceilings_for(_stage, _budget)
            if sum(_ceilings) >= _budget - 0.01 or _stage == 3:
                break
        _alloc_final = _waterfill(_budget, _ceilings)
        _new_est = _expected_overlap(_alloc_final)
        if abs(_new_est - _overlap_est) < 0.05:
            break
        _overlap_est = _new_est

    # Footage-limited: deliver a shorter edit and warn — never freeze-pad.
    _planned_total = sum(_alloc_final)
    if _planned_total < desired_total + _overlap_est - 0.5:
        print(f"[DUR] ⚠️ Only {_planned_total:.1f}s of usable footage for a "
              f"{desired_total:.0f}s target — output will be "
              f"~{max(0.0, _planned_total - _overlap_est):.0f}s "
              f"(delivering shorter instead of padding)")

    # ── Pacing modulation (zero-sum) ─────────────────────────────────────────
    # With music: shot length follows the track's energy envelope — short cuts
    # through high-energy passages, longer holds in lulls — so pacing is felt
    # against the soundtrack, not a fixed positional pattern.
    # Without music: fall back to the tone-specific rhythm pattern.
    # Either way the modulation residual is redistributed among entries that
    # still have head-room so the planned total is EXACTLY preserved.
    _RHYTHM_PAT: Dict[str, List[float]] = {
        "energetic":   [1.00, 0.80, 0.80, 1.35],  # fast-fast-fast-BREATH (every 4th)
        "epic":        [0.80, 0.80, 1.60, 1.00],  # fast-fast-HOLD-reset (every 3rd)
        "cinematic":   [1.00, 1.00, 1.30, 1.00],  # steady with periodic visual breath
        "sentimental": [1.00, 1.10, 1.10, 1.50],  # gradual build towards long holds
        "calm":        [1.15, 1.05, 1.25, 1.10],  # slow & gently varied
    }
    _rhy_pat = _RHYTHM_PAT.get(_tone_key, [1.0])
    if _N > 1:
        _pre_total = sum(_alloc_final)
        _mults: List[float] = []
        if _energy_curve is not None:
            _cum = 0.0
            for i in range(_N):
                _center = _cum + _alloc_final[i] / 2.0
                _e = _energy_at_video_t(_center)
                # energy 0 → 1.35× hold, energy 1 → 0.73× quick cut
                _mults.append(1.0 if _e is None else (1.35 - 0.62 * _e))
                _cum += _alloc_final[i]
            print(f"[PACING] Music-energy pacing active "
                  f"(mult {min(_mults):.2f}–{max(_mults):.2f})")
        else:
            _mults = [_rhy_pat[i % len(_rhy_pat)] for i in range(_N)]
        _mod = [
            min(_ceilings[i], max(FLOOR, _alloc_final[i] * _mults[i]))
            for i in range(_N)
        ]
        _residual = _pre_total - sum(_mod)
        for _pass in range(3):
            if abs(_residual) < 0.01:
                break
            room = [
                (_ceilings[i] - _mod[i]) if _residual > 0 else (_mod[i] - FLOOR)
                for i in range(_N)
            ]
            total_room = sum(r for r in room if r > 0)
            if total_room <= 1e-6:
                break
            for i in range(_N):
                if room[i] > 0:
                    share = _residual * (room[i] / total_room)
                    _mod[i] = min(_ceilings[i], max(FLOOR, _mod[i] + share))
            _residual = _pre_total - sum(_mod)
        _alloc_final = _mod
    # ────────────────────────────────────────────────────────────────────────

    # Log allocation table
    print(f"[ALLOC] target={desired_total:.0f}s  entries={len(_expanded)}")
    for (path, _, m), alloc, avail in zip(_expanded, _alloc_final, _clip_durs):
        seg_tag = f"[{m.get('trim_start_sec', '?'):.0f}-{m.get('trim_end_sec', '?'):.0f}s]" \
            if m.get("trim_start_sec") is not None else ""
        print(f"  {os.path.basename(path):<28}{seg_tag:<12}  "
              f"role={m.get('narrative_role','?'):<11}  "
              f"emotion={m.get('emotion','?'):<10}  vs={float(m.get('visual_score',0.5)):.2f}  "
              f"alloc={alloc:.1f}s / {avail:.1f}s")
    # ────────────────────────────────────────────────────────────────────────

    _closed_raw_ids: set = set()  # avoid double-closing shared raws
    alloc_iter = iter(_alloc_final)
    for i, (path, raw, m) in enumerate(_expanded):
        alloc = next(alloc_iter)
        # Anchor priority: explicit best moment, else the grounded emotional
        # peak (laughter/tears/cheer timestamp from the video-native model).
        _bm_anchor = m.get("best_moment_sec")
        if _bm_anchor is None:
            _bm_anchor = m.get("emotional_peak_sec")
        trimmed = apply_smart_trim(
            raw, i, len(_expanded),
            alloc_sec         = alloc,
            vision_trim_start = m.get("trim_start_sec"),
            vision_trim_end   = m.get("trim_end_sec"),
            best_moment_sec   = _bm_anchor,
            story_weight      = _story_weight(path),
        )
        # Defer closing raw — the entire lazy MoviePy pipeline chains back to
        # raw's reader through write_videofile. Closed in the finally block.
        # Use id() dedup so shared raws (multi-segment clips) are only closed once.
        if trimmed is not raw and id(raw) not in _closed_raw_ids:
            _closed_raw_ids.add(id(raw))
            _deferred_close.append(raw)
        clips.append(trimmed)
        gc.collect()  # release frame buffers before loading the next clip

    if not clips:
        # Build a rich failure report so the error is actionable
        failure_lines = []
        for fp in load_failures[:12]:
            reason = _load_errors.get(fp, "unknown")
            failure_lines.append(f"  {fp}  →  {reason}")
        preview = "\n".join(failure_lines) if failure_lines else "(no failure details)"
        # Clean up any temp files before raising
        for f in temp_files:
            try:
                os.remove(f)
            except Exception:
                pass
        raise ValueError(
            "No usable clips provided to generate_video().\n"
            "Top load failures:\n" + preview
        )

    # --- Beat grid (in video-timeline coordinates) if user music provided ---
    # Music playback at video time t plays music sample (music_start_sec + t),
    # so a beat at music time b lands on screen at video time b - music_start_sec.
    # Beats come from the single up-front music analysis; detect_beats is only
    # a fallback if that failed.
    beat_times = []
    if custom_music_path and os.path.exists(custom_music_path):
        _raw_beats = _music_beats_raw or detect_beats(custom_music_path)
        _music_offset = float(music_start_sec or 0.0)
        total_clip_duration = sum(c.duration for c in clips)
        beat_times = [
            b - _music_offset
            for b in _raw_beats
            if b >= _music_offset and (b - _music_offset) < total_clip_duration
        ]

    # --- Adaptive transitions, metadata-driven ---
    available_transitions = list_available_transitions()

    # Emotional-direction → transition mapping.
    # Escalating pairs (tension/energy rising): use kinetic transitions.
    # Resolving pairs (releasing tension): use smooth dissolves.
    _EMO_ESCALATE = frozenset({
        ("calm", "tense"), ("calm", "exciting"), ("tense", "dramatic"),
        ("neutral", "exciting"), ("happy", "inspiring"), ("neutral", "dramatic"),
        ("calm", "dramatic"), ("neutral", "tense"), ("happy", "dramatic"),
    })
    _EMO_RESOLVE = frozenset({
        ("dramatic", "calm"), ("tense", "calm"), ("exciting", "neutral"),
        ("dramatic", "inspiring"), ("tense", "neutral"), ("sad", "inspiring"),
        ("exciting", "calm"), ("dramatic", "happy"), ("tense", "happy"),
    })

    # Tracks the last transition actually used so back-to-back cuts don't
    # repeat the same effect twice in a row purely by chance.
    _last_transition_used = [None]

    def _choice_avoid_repeat(pool: list) -> str:
        candidates = list(pool)
        if len(candidates) > 1 and _last_transition_used[0] in candidates:
            without_repeat = [c for c in candidates if c != _last_transition_used[0]]
            if without_repeat:
                candidates = without_repeat
        pick = random.choice(candidates)
        _last_transition_used[0] = pick
        return pick

    def _pick_transition(meta: dict, prev_meta: Optional[dict] = None) -> str:
        """
        Choose transition blending:
          0. The scoring arc pass's own transition choice for this cut, if it ran
          1. Narrative role (hook / payoff always override)
          2. Emotional direction between the two clips (escalate → kinetic, resolve → smooth)
          3. Per-clip shot/emotion in context of tone
          4. Tone's default pool

        Takes the per-timeline-entry metadata dicts directly (not a path lookup
        into the global _meta) so multi-segment clips — where two entries share
        the same source path but carry different per-segment shot/emotion/role —
        get the correct metadata for the segment actually at this position.
        """
        meta = meta or {}
        # 0. Arc pass already reasoned about the emotional/energy change between
        # this clip and its predecessor (in its own proposed order) — trust it
        # over the heuristic cascade below when it made a call for this cut.
        arc_t = meta.get("arc_transition")
        if arc_t and arc_t in available_transitions:
            _last_transition_used[0] = arc_t
            return arc_t

        shot  = (meta.get("shot_type", "") or "").lower()
        emo   = (meta.get("emotion",   "") or "").lower()
        role  = (meta.get("narrative_role", "") or "").lower()

        # 1. Narrative role overrides (same for all tones)
        if role == "hook":
            if _tone_key in ("energetic", "epic"):
                return _choice_avoid_repeat(["zoom_in", "slide_left", "slide_right"])
            return _choice_avoid_repeat(["zoom_in", "crossfade"])
        if role == "payoff":
            if _tone_key in ("sentimental", "calm"):
                return _choice_avoid_repeat(["fadein", "crossfade"])
            return _choice_avoid_repeat(["zoom_out", "fadein", "crossfade"])

        # 2. Emotion direction: how does mood change between consecutive clips?
        if prev_meta:
            prev_emo = (prev_meta.get("emotion") or "").lower()
            if prev_emo and emo:
                pair = (prev_emo, emo)
                if pair in _EMO_ESCALATE:
                    # Tension rising → kinetic cut
                    return _choice_avoid_repeat(["zoom_in", "slide_up", "slide_left"])
                if pair in _EMO_RESOLVE:
                    # Tension releasing → smooth dissolve
                    return _choice_avoid_repeat(["crossfade", "fadein"])

        # 3. Clip-level emotion/shot in context of tone
        if emo in ("exciting", "dramatic") or shot == "action":
            pool = ["slide_left", "slide_right", "zoom_in"] if _tone_key in ("calm", "sentimental") \
                   else _tone_transition_pool
            return _choice_avoid_repeat(pool)
        if shot in ("talking_head", "close_up") or emo in ("calm", "sad", "inspiring"):
            pool = ["crossfade", "fadein"] if _tone_key not in ("energetic", "epic") \
                   else ["slide_left", "slide_right"]
            return _choice_avoid_repeat(pool)

        # 4. Default: tone's preferred pool
        return _choice_avoid_repeat(_tone_transition_pool)

    def transition_for_pair(a, b) -> float:
        # At most 20% of the shorter clip
        max_allowed = 0.20 * min(a.duration, b.duration)
        safe = min(IDEAL_TRANSITION, max_allowed)
        return safe if safe >= 0.12 else 0.0

    # ── Planned transition overlaps ──────────────────────────────────────────
    # Per-pair transition durations computed from the ACTUAL trimmed clip
    # lengths, exactly as the assembly loop will apply them. Each applied
    # transition overlaps the incoming clip's head, consuming tdur of timeline,
    # so expected runtime = Σclip_durations − Σtdur.
    def _plan_tdurs() -> list:
        tdurs = []
        prev_d = float(clips[0].duration) if clips else 0.0
        for _pi in range(1, len(clips)):
            d_i = float(clips[_pi].duration)
            safe = min(IDEAL_TRANSITION, 0.20 * min(prev_d, d_i))
            tdur = safe if safe >= 0.12 else 0.0
            tdurs.append(tdur)
            prev_d = d_i - tdur
        return tdurs

    def _recut(idx: int, new_start: float, new_end: float) -> None:
        """Re-cut clips[idx] from its raw source and keep _trim_log in sync."""
        raw_i = _expanded[idx][1]
        # Whole-clip entries were never registered for deferred close (trimmed
        # is raw); once replaced by a subclip, the raw must be closed later.
        if id(raw_i) not in _closed_raw_ids:
            _closed_raw_ids.add(id(raw_i))
            _deferred_close.append(raw_i)
        clips[idx] = apply_without_mask(apply_subclip(raw_i, new_start, new_end))
        _trim_log[idx] = (new_start, new_end)

    # ── Beat-aligned cut boundaries (zero-sum) ───────────────────────────────
    # Snap each cut to the nearest music beat by shifting the boundary between
    # a clip pair: clip k's out-point moves by +delta, clip k+1's out-point by
    # −delta. Total runtime is EXACTLY unchanged and later cuts don't move.
    if beat_times and len(beat_times) >= 2 and len(clips) >= 2:
        _intervals = [b2 - b1 for b1, b2 in zip(beat_times, beat_times[1:])]
        _med_interval = sorted(_intervals)[len(_intervals) // 2]
        _max_shift = min(0.45 * _med_interval, 0.8)
        _tdurs_beat = _plan_tdurs()
        _cursor = 0.0
        for k in range(len(clips) - 1):
            _cursor += float(clips[k].duration) - (_tdurs_beat[k] if k < len(_tdurs_beat) else 0.0)
            _nearest = min(beat_times, key=lambda b: abs(b - _cursor))
            delta = _nearest - _cursor
            if abs(delta) < 0.02 or abs(delta) > _max_shift:
                continue
            s_k, e_k = _trim_log.get(k, (0.0, float(clips[k].duration)))
            s_n, e_n = _trim_log.get(k + 1, (0.0, float(clips[k + 1].duration)))
            src_k = float(getattr(_expanded[k][1], "duration", 0.0) or 0.0)
            src_n = float(getattr(_expanded[k + 1][1], "duration", 0.0) or 0.0)
            new_e_k = e_k + delta
            new_e_n = e_n - delta
            # Guards: stay inside source footage, keep both clips >= FLOOR
            if not (s_k + FLOOR <= new_e_k <= src_k):
                continue
            if new_e_n - s_n < FLOOR or new_e_n > src_n:
                continue
            _recut(k, s_k, new_e_k)
            _recut(k + 1, s_n, new_e_n)
            _cursor += delta  # boundary k now sits on the beat
            print(f"[BEAT] Cut {k + 1} snapped {delta:+.2f}s to beat @ {_nearest:.2f}s")

    # ── Residual correction: land on target BEFORE rendering ────────────────
    # Analytic expected runtime = Σclip_durations − Σplanned_tdurs. Any drift
    # from ceilings/floors/beat clamps is repaired by nudging clip out-points:
    # extend into unused source tail when short, shave the roomiest clips when
    # long. Iterated because tdur caps shift slightly as durations change.
    if target_duration_sec and clips:
        for _rc_pass in range(3):
            _tdurs_plan = _plan_tdurs()
            _expected = sum(float(c.duration) for c in clips) - sum(_tdurs_plan)
            _residual = float(desired_total) - _expected
            if abs(_residual) <= 0.15:
                break
            _adjs = []
            for _ri, c in enumerate(clips):
                s_i, e_i = _trim_log.get(_ri, (0.0, float(c.duration)))
                src_d = float(getattr(_expanded[_ri][1], "duration", 0.0) or 0.0)
                if _residual > 0:
                    room = max(0.0, src_d - e_i)          # unused source tail
                else:
                    room = max(0.0, (e_i - s_i) - FLOOR)  # trimmable excess
                _adjs.append(room)
            _total_room = sum(_adjs)
            if _total_room <= 1e-6:
                if _residual > 0.5:
                    print(f"[DUR] ⚠️ No footage left to extend — output will be "
                          f"~{_expected:.1f}s of a {desired_total:.0f}s target")
                break
            _needed = min(abs(_residual), _total_room)
            for _ri, room in enumerate(_adjs):
                if room <= 0:
                    continue
                share = _needed * (room / _total_room)
                s_i, e_i = _trim_log.get(_ri, (0.0, float(clips[_ri].duration)))
                new_e = e_i + share if _residual > 0 else e_i - share
                _recut(_ri, s_i, new_e)
        _tdurs_plan = _plan_tdurs()
    else:
        _tdurs_plan = _plan_tdurs()

    # NOTE: _deferred_close raws are kept alive through the entire lazy pipeline
    # (beat align → fades → transitions → concat → write). Closed in finally below.

    # First clip fade
    first_fade = min(IDEAL_TRANSITION, 0.25 * clips[0].duration)
    first_fade = first_fade if first_fade >= 0.12 else 0.0
    final_clips: List["VideoFileClip"] = [apply_fadein(clips[0], first_fade)]

    for i in range(1, len(clips)):
        prev = final_clips[-1]
        nxt = clips[i]
        # Use the PLANNED transition duration (same values the duration budget
        # accounted for) so runtime stays deterministic.
        tdur = _tdurs_plan[i - 1] if (i - 1) < len(_tdurs_plan) else transition_for_pair(prev, nxt)

        if tdur > 0 and available_transitions:
            # Index metadata via _expanded, not clip_paths — after multi-segment
            # expansion (and overflow drops) clip_paths no longer lines up 1:1
            # with the timeline entries. Pass the per-entry meta dicts directly
            # so multi-segment clips (same path, different segment meta) resolve
            # to the segment actually at this timeline position.
            _prev_meta = _expanded[i - 1][2] if (i - 1) < len(_expanded) else None
            _nxt_meta = _expanded[i][2] if i < len(_expanded) else {}
            transition = _pick_transition(_nxt_meta, prev_meta=_prev_meta)
            try:
                transitioned = apply_transition(prev, nxt, transition, tdur)
                final_clips[-1] = apply_without_mask(apply_end(transitioned, transitioned.duration))
                _transition_log.append({"transition": transition, "duration": tdur, "applied": True})
                # The transition composite already played nxt's first tdur
                # seconds blended over prev's tail. Resume nxt AFTER that head
                # so no frames repeat and the overlap consumes exactly tdur of
                # timeline — the same math as the 4K xfade re-render. (The old
                # code re-appended nxt in full, replaying its head twice.)
                nxt_rest = apply_subclip(nxt, tdur, float(nxt.duration))
                final_clips.append(apply_without_mask(nxt_rest))
                continue
            except Exception as e:
                print(f"⚠️ Transition failed ({transition}) at clip {i}: {repr(e)}")

        # fallback: no transition. If an overlap was planned, trim nxt's tail
        # by the planned amount so total runtime still matches the budget.
        _transition_log.append({"transition": "crossfade", "duration": 0.0, "applied": False})
        if tdur > 0 and float(nxt.duration) - tdur >= MIN_KEEP_SEC:
            nxt = apply_subclip(nxt, 0, float(nxt.duration) - tdur)
            if i in _trim_log:
                _s0, _e0 = _trim_log[i]
                _trim_log[i] = (_s0, max(_s0 + MIN_KEEP_SEC, _e0 - tdur))
        fade_next = min(0.12, 0.15 * nxt.duration)
        fade_next = fade_next if fade_next >= 0.08 else 0.0
        final_clips.append(apply_without_mask(apply_fadein(nxt, fade_next)))

    # --- Validate final_clips BEFORE concatenation so errors are readable ---
    for idx, c in enumerate(final_clips):
        if not hasattr(c, 'duration') or not hasattr(c, 'fps') or not hasattr(c, 'size'):
            raise ValueError(f"final_clips[{idx}] is missing required attributes (duration, fps, size).")
        if c.duration is None or c.duration < 0.05:
            raise ValueError(f"final_clips[{idx}] has invalid duration: {c.duration}")
        if c.fps is None or c.fps < 1:
            raise ValueError(f"final_clips[{idx}] has invalid fps: {c.fps}")
        if not c.size or not isinstance(c.size, (tuple, list)) or len(c.size) != 2:
            raise ValueError(f"final_clips[{idx}] has invalid size: {c.size}")

    # Use method='chain' if all final_clips have the same size/fps
    same_size = all(
        (getattr(c, "size", None) == getattr(final_clips[0], "size", None)
         and getattr(c, "fps", None) == getattr(final_clips[0], "fps", None))
        for c in final_clips
    )
    concat_method = "chain" if same_size else "compose"
    final = concatenate_videoclips(final_clips, method=concat_method, padding=0.0)
    final = apply_without_mask(final)

    if target_duration_sec and float(getattr(final, "duration", 0.0) or 0.0) > float(target_duration_sec) + 0.2:
        final = apply_subclip(final, 0.0, float(target_duration_sec))

    # Optional opening card — the frontend's "Opening title card" toggle isn't
    # tone-conditional, so honor it for every tone rather than silently no-op
    # whenever a user picks anything other than Cinematic.
    if show_opening_card:
        overlays = []
        try:
            # Build a TextClip compatible with both MoviePy v1 and v2
            try:
                tc = TextClip(
                    text="KEEMOGRAPHY PRESENTS",
                    font_size=70,
                    color="white",
                    font="Arial-Bold",
                )
                tc = tc.with_duration(3).with_position("center")
            except (TypeError, AttributeError):
                tc = TextClip(
                    "KEEMOGRAPHY PRESENTS",
                    fontsize=70,
                    color="white",
                    font="Arial-Bold",
                ).set_duration(3).set_position("center")
            overlays.append(tc)
            overlays[-1] = apply_fadein(overlays[-1], 0.5)
            overlays[-1] = apply_fadeout(overlays[-1], 0.5)
        except Exception as e:
            print(f"ℹ️ Skipping TextClip overlay (likely missing ImageMagick/font): {repr(e)}")
            overlays = []

        if overlays:
            final = CompositeVideoClip([final] + overlays)
            final = apply_without_mask(final)

    # Background music
    music_path = None
    if custom_music_path and os.path.exists(custom_music_path):
        normalized_custom = normalize_audio_for_mix(
            custom_music_path,
            start_sec=music_start_sec,
            end_sec=music_end_sec,
        )
        music_path = normalized_custom or custom_music_path
    else:
        music_path = _get_music_for_tone(tone)
    bg_music_clip = None
    if music_path and os.path.exists(music_path):
        try:
            # Volume is NOT applied here — it depends on whether music ends up
            # ducked under original audio or carrying the track alone, decided
            # in the mixing step below. Applying _music_vol unconditionally
            # here used to double-attenuate the mix_original_audio branch
            # (tone volume, then another 0.20 on top) and left the music-only
            # branch far too quiet (tone volumes like 0.15-0.22 as the SOLE
            # track).
            bg_music_clip = AudioFileClip(music_path)

            # Fit music to final duration (loop when too short, trim when too long).
            if bg_music_clip.duration < final.duration:
                loops = int(final.duration // max(0.1, bg_music_clip.duration)) + 1
                bg_music_clip = concatenate_audioclips([bg_music_clip] * max(1, loops))
            bg_music_clip = apply_audio_subclip(bg_music_clip, 0, min(bg_music_clip.duration, final.duration))
        except Exception as e:
            print(f"⚠️ Could not load background music from {music_path}: {repr(e)}")
            print("   → Add MP3 files to assets/music/ (see assets/music/README.md)")
            bg_music_clip = None
    elif music_path:
        print(f"⚠️ Background music file not found: {music_path}")
        print("   → Add MP3 files to assets/music/ (see assets/music/README.md)")

    # Audio mixing
    # SOLO_MUSIC_VOL: music is the only audio track (either mix_original_audio
    # is off, or the clips had no usable audio at all), so it should carry the
    # edit at a normal listening level rather than the ducked "background bed"
    # level _music_vol is tuned for.
    SOLO_MUSIC_VOL = 0.9
    try:
        if bg_music_clip and final.audio is not None:
            if mix_original_audio:
                mixed = CompositeAudioClip([
                    apply_audio_volume(final.audio, 1.0),
                    apply_audio_volume(bg_music_clip, _music_vol),
                ])
                final = apply_set_audio(final, mixed)
            else:
                final = apply_set_audio(final, apply_audio_volume(bg_music_clip, SOLO_MUSIC_VOL))
        elif bg_music_clip and final.audio is None:
            final = apply_set_audio(final, apply_audio_volume(bg_music_clip, SOLO_MUSIC_VOL))
    except Exception as e:
        print(f"⚠️ Audio mix failed, keeping existing audio: {repr(e)}")

    # Final hard target safety net (trim-only; the budget upstream should
    # already have landed the timeline on target).
    final = enforce_target_duration(final, target_duration_sec)

    # Write output
    # h264_videotoolbox is macOS-only; skip the attempt entirely on Linux (Railway)
    # to avoid an exception inside the except block aborting the libx264 fallback.
    _use_hw = sys.platform == "darwin"
    temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    try:
        if _use_hw:
            try:
                final.write_videofile(
                    temp_out.name,
                    codec="h264_videotoolbox",
                    audio_codec="aac",
                    fps=_fps,
                    threads=2,
                )
            except Exception as hw_err:
                print(f"⚠️ Hardware encoding unavailable ({hw_err}), falling back to libx264...")
                final.write_videofile(
                    temp_out.name,
                    codec="libx264",
                    audio_codec="aac",
                    fps=_fps,
                    threads=2,
                    preset="medium",
                )
        else:
            final.write_videofile(
                temp_out.name,
                codec="libx264",
                audio_codec="aac",
                fps=_fps,
                threads=2,
                preset="medium",
            )
    except Exception as e:
        print(f"❌ Failed to write video file: {repr(e)}")
        raise
    finally:
        try:
            final.close()
        except Exception:
            pass
        for c in clips:
            try:
                c.close()
            except Exception:
                pass
        # Close raw source clips now — write_videofile is done so the lazy
        # reader chain is no longer needed.
        for _raw in _deferred_close:
            try:
                _raw.close()
            except Exception:
                pass
        _deferred_close.clear()
        if bg_music_clip:
            try:
                bg_music_clip.close()
            except Exception:
                pass
        # Clean up any temp files created for URLs
        for f in temp_files:
            try:
                os.remove(f)
            except Exception:
                pass

    # ── Duration report ───────────────────────────────────────────────────────
    # Verify the actual output duration with ffprobe. The budget is enforced
    # upstream (duration-aware selection → exact waterfill → residual
    # correction), so this is a diagnostic. Freeze-frame padding was removed by
    # design: when footage genuinely can't fill the target we deliver a shorter
    # edit and warn instead of cloning the last frame.
    if target_duration_sec:
        try:
            _probe_cmd = [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                temp_out.name,
            ]
            _probe_res = subprocess.run(_probe_cmd, capture_output=True, text=True)
            _actual_dur = float((_probe_res.stdout or "0").strip() or "0")
            _target_dur = float(target_duration_sec)
            _delta = _actual_dur - _target_dur
            print(f"[DUR] output={_actual_dur:.2f}s  target={_target_dur:.2f}s  delta={_delta:+.2f}s")
            if _delta < -2.5:
                print(f"[DUR] ⚠️ Output is {-_delta:.1f}s short of target — source footage "
                      f"was insufficient; delivered shorter edit instead of padding.")
        except Exception as _dur_err:
            print(f"[DUR] Duration check skipped: {_dur_err}")

    # ── 4K re-render: replay trim windows on original 4K sources ─────────────
    # All MoviePy work was done on 1080p proxies (fast, low-RAM).
    # Now we hand the exact trim windows + transition sequence to ffmpeg, which
    # decodes the ORIGINAL 4K files and produces a native-res output.
    # The proxy MP4 is kept alive as an audio reference (it has the mixed
    # music/original-audio track), then deleted after the mux.
    if has_4k:
        print(f"[4K] Re-rendering {len(_expanded)} segments at original resolution…")
        k4_path = _ffmpeg_4k_render(
            expanded_entries  = _expanded,
            trim_log          = _trim_log,
            transition_log    = _transition_log,
            proxy_to_orig     = _proxy_to_orig,
            proxy_output_path = temp_out.name,
            target_fps        = _fps,
        )
        # Duration parity check: proxy and 4K timelines share the same trim/
        # transition math now (real head overlap on both paths), so they must
        # agree within encoder padding.
        if target_duration_sec:
            try:
                _r4 = subprocess.run(
                    ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                     "-of", "default=noprint_wrappers=1:nokey=1", k4_path],
                    capture_output=True, text=True,
                )
                _dur4 = float((_r4.stdout or "0").strip() or "0")
                print(f"[DUR] 4k={_dur4:.2f}s  target={float(target_duration_sec):.2f}s  "
                      f"delta={_dur4 - float(target_duration_sec):+.2f}s")
            except Exception:
                pass
        # 4K re-render succeeded — proxy output is no longer needed
        try:
            os.remove(temp_out.name)
        except Exception:
            pass
        return k4_path

    return temp_out.name


def _get_music_for_tone(tone: str) -> Optional[str]:
    tone = (tone or "").lower()
    if tone not in ("cinematic", "energetic", "sentimental", "epic", "calm"):
        return None
    # Absolute path — the old CWD-relative "assets/music/…" broke whenever the
    # server wasn't started from the project root.
    return os.path.join(_MODULE_DIR, "assets", "music", f"{tone}.mp3")
