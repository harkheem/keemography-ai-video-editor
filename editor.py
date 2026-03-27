# =============================
# editor.py (FULL, CORRECTED)
# =============================

# ---- bootstrap critical deps (handy on Streamlit Cloud before import) ----
import sys, subprocess, importlib.util

def ensure(spec: str, import_name: str | None = None):
    name = import_name or spec.split("==")[0].split(">=")[0].split("[")[0]
    if importlib.util.find_spec(name) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", spec])

ensure("moviepy==2.1.1", "moviepy")
ensure("imageio-ffmpeg>=0.5.1", "imageio_ffmpeg")
ensure("imageio>=2.34.0", "imageio")
ensure("Pillow>=10.4.0", "PIL")
ensure("numpy>=2.0.2", "numpy")
ensure("librosa>=0.9.2", "librosa")
# -------------------------------------------------------------------------

import os
import tempfile
import random
from typing import List, Dict, Optional, Union
import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from googleapiclient.discovery import build
import re
import concurrent.futures
import librosa
import numpy as np

from transition import apply_transition, list_available_transitions


def _get_api_key(explicit_key: Optional[str] = None) -> Optional[str]:
    if explicit_key:
        return explicit_key
    return os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")


def transcribe_videos(
    video_paths: List[str],
    openai_api_key: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Transcribe each video using OpenAI Whisper API (model='whisper-1').
    Returns: list of dicts like {"path": <path>, "text": <transcript>}
    """
    from openai import OpenAI

    api_key = _get_api_key(openai_api_key)
    if not api_key:
        raise RuntimeError("Missing OpenAI API key. Set API_KEY or OPENAI_API_KEY.")

    client = OpenAI(api_key=api_key)
    results: List[Dict[str, str]] = []

    for path in video_paths:
        try:
            with open(path, "rb") as f:
                tx = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                )
            text = getattr(tx, "text", "") or ""
        except Exception as e:
            print(f"⚠️ Transcription failed for {path}: {repr(e)}")
            text = ""
        results.append({"path": path, "text": text})

    return results


VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm")

def extract_video_links_from_page(url: str) -> list[str]:
    """
    Given a URL to a web page, extract all video file links (.mp4, .mov, etc.).
    Returns a list of absolute URLs.
    """
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        links = set()
        # Find <a href=...> and <video src=...> and <source src=...>
        for tag in soup.find_all(["a", "video", "source"]):
            src = tag.get("href") or tag.get("src")
            if src and any(src.lower().endswith(ext) for ext in VIDEO_EXTENSIONS):
                abs_url = urljoin(url, src)
                links.add(abs_url)
        return list(links)
    except Exception as e:
        print(f"⚠️ Failed to extract video links from {url}: {repr(e)}")
        return []

def extract_gdrive_folder_videos(folder_url: str) -> list[str]:
    """
    Given a Google Drive folder URL, return a list of direct download links for video files in the folder.
    Only works for public folders/files.
    """
    # Extract folder ID from URL
    m = re.search(r"/folders/([a-zA-Z0-9_-]+)", folder_url)
    if not m:
        print(f"⚠️ Could not extract folder ID from {folder_url}")
        return []
    folder_id = m.group(1)
    try:
        service = build("drive", "v3", developerKey=os.getenv("GOOGLE_API_KEY"))
        # Query for video files in the folder
        query = f"'{folder_id}' in parents and mimeType contains 'video/' and trashed = false"
        results = service.files().list(q=query, fields="files(id, name, mimeType)", pageSize=1000).execute()
        files = results.get("files", [])
        links = []
        for f in files:
            # Only allow known video extensions
            if any(f["name"].lower().endswith(ext) for ext in VIDEO_EXTENSIONS):
                # Public direct download link format
                links.append(f"https://drive.google.com/uc?id={f['id']}&export=download")
        return links
    except Exception as e:
        print(f"⚠️ Failed to extract videos from Google Drive folder: {repr(e)}")
        return []

def normalize_clip_paths(clip_paths: list[Union[str, None]]) -> list[str]:
    """
    Given a list of paths/URLs, expand any page URLs into their video links.
    Returns a flat list of video file paths/URLs.
    """
    out = []
    for p in clip_paths:
        if not isinstance(p, str):
            continue
        # Google Drive folder support
        if re.match(r"https?://drive\.google\.com/.*/folders/", p):
            found = extract_gdrive_folder_videos(p)
            out.extend(found)
        elif p.startswith(("http://", "https://")) and not p.lower().endswith(VIDEO_EXTENSIONS):
            # Treat as a page, try to extract video links
            found = extract_video_links_from_page(p)
            out.extend(found)
        else:
            out.append(p)
    return out


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
    cmd = ["ffmpeg", "-y", "-hwaccel", "videotoolbox"]
    for orig, _s, _e in segments:
        cmd += ["-i", orig]

    # ── filter_complex: trim + normalize each clip ────────────────────────────
    filter_parts = []
    clip_durs = []
    for i, (_orig, s, e) in enumerate(segments):
        d = e - s
        clip_durs.append(d)
        filter_parts.append(
            f"[{i}:v]trim=start={s:.4f}:end={e:.4f},"
            f"setpts=PTS-STARTPTS,"
            f"fps={target_fps},"
            f"scale={out_w}:{out_h}:flags=lanczos,"
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
                fc += (
                    f";[{prev_label}][v{i}]xfade=transition={xftype}"
                    f":duration={tdur:.4f}:offset={offset:.4f}[{out_label}]"
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

    def normalize_audio_for_mix(input_path: str) -> Optional[str]:
        if not input_path or not os.path.exists(input_path):
            return None
        out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-vn",
            "-ac", "2",
            "-ar", "44100",
            "-c:a", "pcm_s16le",
            out_path,
        ]
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

        if duration <= target + 0.35:
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

    # ── Weighted screen-time allocation ─────────────────────────────────────
    # Each clip earns screen time in proportion to how much it deserves,
    # based on GPT-4o metadata — not equal fair share.
    #
    # Weight dimensions:
    #   narrative_role  — payoff clips get the most time, broll the least
    #   visual_score    — higher quality = more time (scaled to avoid starving)
    #   emotion         — emotionally heavy clips need longer to land
    #   story_weight    — overlap with user storyline = extra budget
    #   _trim_mult      — tone compression (energetic edits trim more overall)
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

    desired_total = float(target_duration_sec) if target_duration_sec else 34.0
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
            for seg in segments:
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
        role = (m.get("narrative_role") or "development").lower()
        emo  = (m.get("emotion")        or "neutral").lower()
        vs   = float(m.get("visual_score") or 0.5)
        sw   = _story_weight(path)

        w = (
            _ROLE_WEIGHT.get(role, 0.90)
            * (0.35 + 0.65 * max(0.0, min(1.0, vs)))  # quality, not starving low scores
            * _EMOTION_WEIGHT.get(emo, 0.90)
            * (1.0 + 0.25 * sw)                        # story overlap bonus
            * _trim_mult                                # tone-level compression
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
    # CEIL_RATIO must be >= 1/N so that N clips can collectively fill 100% of the
    # budget. With 2 clips, 0.45*2=0.90 — always 10% short. Use max(0.45, 1/N).
    CEIL_RATIO = max(0.45, 1.0 / max(1, _N))

    # Per-clip hard ceiling: actual available footage OR the ratio cap, whichever
    # is smaller. Ensures we never ask a clip for more than it has.
    _ceilings = [max(FLOOR, min(_clip_durs[i], desired_total * CEIL_RATIO))
                 for i in range(_N)]

    # ── Iterative saturation allocation ─────────────────────────────────────
    # Single-pass normalization is broken: rescaling after clamping pushes
    # ceiling-capped clips above their ceiling, then apply_smart_trim silently
    # clips them back, leaking budget. Instead we iterate:
    #   1. Distribute budget proportionally to unsaturated clips.
    #   2. Any clip whose tentative alloc >= ceiling is pinned to that ceiling.
    #   3. Its budget surplus is freed for the remaining unsaturated clips.
    #   4. Repeat until no new pins → assign remainder proportionally.
    _alloc_final = [0.0] * _N
    _saturated   = [False] * _N
    _rem_budget  = desired_total

    for _iter in range(_N + 2):
        _free_w = sum(_raw_weights[i] for i in range(_N) if not _saturated[i]) or 1e-9
        _any_new = False
        for i in range(_N):
            if not _saturated[i]:
                _tentative = _rem_budget * _raw_weights[i] / _free_w
                if _tentative >= _ceilings[i] - 0.01:
                    _alloc_final[i] = _ceilings[i]
                    _saturated[i]   = True
                    _rem_budget    -= _ceilings[i]
                    _any_new        = True
        if not _any_new:
            # No further saturation — assign remaining budget proportionally
            _free_w = sum(_raw_weights[i] for i in range(_N) if not _saturated[i]) or 1e-9
            for i in range(_N):
                if not _saturated[i]:
                    _alloc_final[i] = max(FLOOR, _rem_budget * _raw_weights[i] / _free_w)
            break

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
        trimmed = apply_smart_trim(
            raw, i, len(_expanded),
            alloc_sec         = alloc,
            vision_trim_start = m.get("trim_start_sec"),
            vision_trim_end   = m.get("trim_end_sec"),
            best_moment_sec   = m.get("best_moment_sec"),
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

    # --- Beat-aligned editing if user music provided ---
    beat_times = []
    if custom_music_path and os.path.exists(custom_music_path):
        beat_times = detect_beats(custom_music_path)
        # Only use beats within the total video duration
        total_clip_duration = sum(c.duration for c in clips)
        beat_times = [b for b in beat_times if b < total_clip_duration]

    # --- Adaptive transitions, metadata-driven ---
    available_transitions = list_available_transitions()

    def _pick_transition(path: str) -> str:
        """
        Choose transition blending tone-level bias with per-clip visual metadata.
        Tone defines the character of the edit; clip metadata refines individual cuts.
        """
        meta = _meta.get(path, {})
        shot  = (meta.get("shot_type", "") or "").lower()
        emo   = (meta.get("emotion",   "") or "").lower()
        role  = (meta.get("narrative_role", "") or "").lower()

        # Narrative role overrides everything (same for all tones)
        if role == "hook":
            if _tone_key in ("energetic", "epic"):
                return random.choice(["zoom_in", "slide_left", "slide_right"])
            return random.choice(["zoom_in", "crossfade"])
        if role == "payoff":
            if _tone_key in ("sentimental", "calm"):
                return random.choice(["fadein", "crossfade"])
            return random.choice(["zoom_out", "fadein", "crossfade"])

        # Clip-level emotion/shot considered in the context of tone
        if emo in ("exciting", "dramatic") or shot == "action":
            # Even in a calm edit, action clips get something kinetic
            pool = ["slide_left", "slide_right", "zoom_in"] if _tone_key in ("calm", "sentimental") \
                   else _tone_transition_pool
            return random.choice(pool)
        if shot in ("talking_head", "close_up") or emo in ("calm", "sad", "inspiring"):
            # In energetic edits, even dialogue clips get a slide
            pool = ["crossfade", "fadein"] if _tone_key not in ("energetic", "epic") \
                   else ["slide_left", "slide_right"]
            return random.choice(pool)

        # Default: use tone's preferred pool
        return random.choice(_tone_transition_pool)

    def transition_for_pair(a, b) -> float:
        # At most 20% of the shorter clip
        max_allowed = 0.20 * min(a.duration, b.duration)
        safe = min(IDEAL_TRANSITION, max_allowed)
        return safe if safe >= 0.12 else 0.0

    # If we have enough beats, align cuts to them
    cut_points = []
    if beat_times and len(beat_times) >= len(clips):
        # Use beat times as cut points for each clip
        cut_points = beat_times[:len(clips)]
        # Optionally, trim/extend clips to match beat intervals
        for i, c in enumerate(clips):
            if i < len(cut_points)-1:
                start = cut_points[i]
                end = cut_points[i+1]
                beat_len = end - start
                if beat_len > 0 and c.duration > beat_len:
                    clips[i] = apply_subclip(c, 0, beat_len)
                    # Update trim log: keep the chosen start, shorten the window
                    if i in _trim_log:
                        _s0, _ = _trim_log[i]
                        _trim_log[i] = (_s0, _s0 + beat_len)

    # Safe to close raw clips now — beat alignment (the last reader-dependent
    # operation on trimmed subclips) is complete.
    # NOTE: _deferred_close raws are kept alive through the entire lazy pipeline
    # (beat align → fades → transitions → concat → write). Closed in finally below.

    # First clip fade
    first_fade = min(IDEAL_TRANSITION, 0.25 * clips[0].duration)
    first_fade = first_fade if first_fade >= 0.12 else 0.0
    final_clips: List["VideoFileClip"] = [apply_fadein(clips[0], first_fade)]

    for i in range(1, len(clips)):
        prev = final_clips[-1]
        nxt = clips[i]
        tdur = transition_for_pair(prev, nxt)

        if tdur > 0 and available_transitions:
            transition = _pick_transition(clip_paths[i] if i < len(clip_paths) else "")
            try:
                transitioned = apply_transition(prev, nxt, transition, tdur)
                final_clips[-1] = apply_without_mask(apply_end(transitioned, transitioned.duration))
                _transition_log.append({"transition": transition, "duration": tdur, "applied": True})
                fade_next = min(tdur, 0.25 * nxt.duration)
                fade_next = fade_next if fade_next >= 0.12 else 0.0
                final_clips.append(apply_without_mask(apply_fadein(nxt, fade_next)))
                continue
            except Exception as e:
                print(f"⚠️ Transition failed ({transition}) at clip {i}: {repr(e)}")

        # fallback: no transition (or tiny fade)
        _transition_log.append({"transition": "crossfade", "duration": 0.0, "applied": False})
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

    # Optional opening card
    if show_opening_card and (tone or "").lower() == "cinematic":
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
        normalized_custom = normalize_audio_for_mix(custom_music_path)
        music_path = normalized_custom or custom_music_path
    else:
        music_path = _get_music_for_tone(tone)
    bg_music_clip = None
    if music_path and os.path.exists(music_path):
        try:
            bg_music_clip = AudioFileClip(music_path)
            bg_music_clip = apply_audio_volume(bg_music_clip, _music_vol)

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
    try:
        if bg_music_clip and final.audio is not None:
            if mix_original_audio:
                mixed = CompositeAudioClip([
                    apply_audio_volume(final.audio, 1.0),
                    apply_audio_volume(bg_music_clip, 0.20),
                ])
                final = apply_set_audio(final, mixed)
            else:
                final = apply_set_audio(final, bg_music_clip)
        elif bg_music_clip and final.audio is None:
            final = apply_set_audio(final, bg_music_clip)
    except Exception as e:
        print(f"⚠️ Audio mix failed, keeping existing audio: {repr(e)}")

    # Final hard target enforcement (trim or extend so output matches requested length).
    final = enforce_target_duration(final, target_duration_sec)

    # Write output
    temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    try:
        try:
            # Try hardware-accelerated encoding on macOS (h264_videotoolbox); no 'preset' param
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

    # ── Hard duration guarantee ───────────────────────────────────────────────
    # After MoviePy writes the proxy, verify the actual output duration with
    # ffprobe. If it is shorter than 90% of target (e.g. because GPT-4o trim
    # windows were narrower than the allocated budget), pad with a freeze of the
    # last frame so the video is ALWAYS the requested length.
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
            print(f"[DUR] proxy={_actual_dur:.2f}s  target={_target_dur:.2f}s")
            if _actual_dur < _target_dur * 0.99:
                _pad_needed = _target_dur - _actual_dur
                print(f"[DUR] Output is {_actual_dur:.1f}s — padding {_pad_needed:.1f}s to reach {_target_dur:.0f}s")
                _padded = tempfile.NamedTemporaryFile(delete=False, suffix="_padded.mp4")
                _padded.close()
                _pad_cmd = [
                    "ffmpeg", "-y",
                    "-i", temp_out.name,
                    "-vf", f"tpad=stop_mode=clone:stop_duration={_pad_needed:.4f}",
                    "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                    "-c:a", "aac",
                    "-t", str(_target_dur),
                    _padded.name,
                ]
                _pad_result = subprocess.run(_pad_cmd, capture_output=True, text=True)
                if _pad_result.returncode == 0 and os.path.getsize(_padded.name) > 4096:
                    os.remove(temp_out.name)
                    # Replace temp_out reference by reassigning via a wrapper object
                    class _NamedRef:
                        def __init__(self, name): self.name = name
                    temp_out = _NamedRef(_padded.name)
                    print(f"[DUR] Padded proxy → {_target_dur:.0f}s")
                else:
                    try:
                        os.remove(_padded.name)
                    except Exception:
                        pass
                    print(f"[DUR] Padding failed (rc={_pad_result.returncode}), keeping short output")
        except Exception as _dur_err:
            print(f"[DUR] Duration check skipped: {_dur_err}")
            try:
                os.remove(f)
            except Exception:
                pass

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
        # 4K re-render succeeded — proxy output is no longer needed
        try:
            os.remove(temp_out.name)
        except Exception:
            pass
        return k4_path

    return temp_out.name


def _get_music_for_tone(tone: str) -> Optional[str]:
    tone = (tone or "").lower()
    music_map = {
        "cinematic": "assets/music/cinematic.mp3",
        "energetic": "assets/music/energetic.mp3",
        "sentimental": "assets/music/sentimental.mp3",
        "epic": "assets/music/epic.mp3",
        "calm": "assets/music/calm.mp3",
    }
    return music_map.get(tone)
