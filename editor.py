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


def generate_video(
    clip_paths: List[str],
    storyline: str,
    transition_duration: float = 0.3,   # ✅ safer default
    tone: str = "Cinematic",
    mix_original_audio: bool = False,
    show_opening_card: bool = True,
    custom_music_path: Optional[str] = None,  # <-- new param
) -> str:
    """
    Builds a final video by stitching clips with adaptive transitions,
    optional title card, and background music.

    This version NEVER over-trims short clips and provides better
    error details if no clips can be opened.
    """

    from moviepy.editor import (
        VideoFileClip,
        concatenate_videoclips,
        CompositeVideoClip,
        CompositeAudioClip,
        TextClip,
        AudioFileClip,
    )

    # --- Tunables ---
    MIN_KEEP_SEC = 0.40
    IDEAL_TRANSITION = max(0.15, float(transition_duration))

    clips: List["VideoFileClip"] = []
    load_failures: List[str] = []
    temp_files: List[str] = []  # Track temp files for cleanup

    def is_url(path: str) -> bool:
        return isinstance(path, str) and path.startswith(("http://", "https://"))

    def download_to_temp(url: str) -> Optional[str]:
        try:
            resp = requests.get(url, stream=True, timeout=30)
            resp.raise_for_status()
            content_type = resp.headers.get('Content-Type', '')
            if not content_type.startswith('video/'):
                print(f"⚠️ Skipping download: {url} (Content-Type: {content_type})")
                return None
            suffix = os.path.splitext(url)[-1] if "." in url else ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                for chunk in resp.iter_content(chunk_size=8192):
                    tmp.write(chunk)
                return tmp.name
        except Exception as e:
            print(f"⚠️ Failed to download {url}: {repr(e)}")
            return None

    # Parallelize downloads and loading
    def load_clip(path: str) -> Optional["VideoFileClip"]:
        orig_path = path
        try:
            if not isinstance(path, str):
                return None
            if is_url(path):
                path = download_to_temp(path)
                if not path:
                    return None
                temp_files.append(path)
            if not os.path.exists(path) or os.path.getsize(path) < 1024:
                return None
            clip = VideoFileClip(path)
            if not clip.duration or clip.duration < MIN_KEEP_SEC:
                clip.close()
                return None
            return clip
        except Exception:
            return None

    # Normalize clip paths
    clip_paths = normalize_clip_paths(clip_paths)

    # Parallel load
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        loaded = list(executor.map(load_clip, clip_paths))
    clips = [c for c in loaded if c is not None]
    load_failures = [p for p, c in zip(clip_paths, loaded) if c is None]

    if not clips:
        # Give you useful info instead of the generic ValueError
        preview = "\n".join(load_failures[:12]) if load_failures else "(no failure details)"
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

    # --- Adaptive transitions, beat-aligned if possible ---
    available_transitions = list_available_transitions()

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
                if end > start and c.duration > (end-start):
                    clips[i] = c.subclip(0, end-start)

    # First clip fade
    first_fade = min(IDEAL_TRANSITION, 0.25 * clips[0].duration)
    first_fade = first_fade if first_fade >= 0.12 else 0.0
    final_clips: List["VideoFileClip"] = [clips[0].fadein(first_fade) if first_fade else clips[0]]

    # Per-cut paddings
    paddings: List[float] = []

    for i in range(1, len(clips)):
        prev = final_clips[-1]
        nxt = clips[i]
        tdur = transition_for_pair(prev, nxt)

        if tdur > 0 and available_transitions:
            transition = random.choice(available_transitions)
            try:
                transitioned = apply_transition(prev, nxt, transition, tdur)
                final_clips[-1] = transitioned.set_end(transitioned.duration)

                fade_next = min(tdur, 0.25 * nxt.duration)
                fade_next = fade_next if fade_next >= 0.12 else 0.0
                final_clips.append(nxt.fadein(fade_next) if fade_next else nxt)

                paddings.append(-tdur)
                continue
            except Exception as e:
                print(f"⚠️ Transition failed ({transition}) at clip {i}: {repr(e)}")

        # fallback: no transition (or tiny fade)
        fade_next = min(0.12, 0.15 * nxt.duration)
        fade_next = fade_next if fade_next >= 0.08 else 0.0
        final_clips.append(nxt.fadein(fade_next) if fade_next else nxt)
        paddings.append(0.0)

    # Use method='chain' if all clips have same size/fps
    same_size = all((c.size == clips[0].size and c.fps == clips[0].fps) for c in clips)
    concat_method = "chain" if same_size else "compose"
    final = concatenate_videoclips(final_clips, method=concat_method, padding=paddings)

    # --- Validate all clips before render ---
    for idx, c in enumerate(clips):
        if not hasattr(c, 'duration') or not hasattr(c, 'fps') or not hasattr(c, 'size'):
            raise ValueError(f"Clip {idx} is missing required attributes (duration, fps, size).")
        if c.duration is None or c.duration < 0.1:
            raise ValueError(f"Clip {idx} has invalid duration: {c.duration}")
        if c.fps is None or c.fps < 1:
            raise ValueError(f"Clip {idx} has invalid fps: {c.fps}")
        if not c.size or not isinstance(c.size, (tuple, list)) or len(c.size) != 2:
            raise ValueError(f"Clip {idx} has invalid size: {c.size}")

    # Optional opening card
    if show_opening_card and (tone or "").lower() == "cinematic":
        overlays = []
        try:
            overlays.append(
                TextClip(
                    "KEEMOGRAPHY PRESENTS",
                    fontsize=70,
                    color="white",
                    font="Arial-Bold",
                )
                .set_duration(3)
                .set_position("center")
                .fadein(0.5)
                .fadeout(0.5)
            )
        except Exception as e:
            print(f"ℹ️ Skipping TextClip overlay (likely missing ImageMagick): {repr(e)}")
            overlays = []

        if overlays:
            final = CompositeVideoClip([final] + overlays)

    # Background music
    music_path = None
    if custom_music_path and os.path.exists(custom_music_path):
        music_path = custom_music_path
    else:
        music_path = _get_music_for_tone(tone)
    bg_music_clip = None
    if music_path and os.path.exists(music_path):
        try:
            bg_music_clip = AudioFileClip(music_path).volumex(0.2)
            bg_music_clip = bg_music_clip.subclip(0, min(bg_music_clip.duration, final.duration))
        except Exception as e:
            print(f"⚠️ Could not load background music: {repr(e)}")
            bg_music_clip = None

    # Audio mixing
    try:
        if bg_music_clip and final.audio is not None:
            if mix_original_audio:
                mixed = CompositeAudioClip([
                    final.audio.volumex(1.0),
                    bg_music_clip.volumex(0.15),
                ])
                final = final.set_audio(mixed)
            else:
                final = final.set_audio(bg_music_clip)
        elif bg_music_clip and final.audio is None:
            final = final.set_audio(bg_music_clip)
    except Exception as e:
        print(f"⚠️ Audio mix failed, keeping existing audio: {repr(e)}")

    # Write output
    temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    try:
        final.write_videofile(
            temp_out.name,
            codec="h264_videotoolbox",  # Use hardware-accelerated encoding on macOS
            audio_codec="aac",
            fps=24,
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
