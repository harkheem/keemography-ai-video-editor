# editor.py
from __future__ import annotations

import os
import random
import tempfile
from typing import List, Tuple

from moviepy.editor import (
    VideoFileClip,
    concatenate_videoclips,
    CompositeVideoClip,
    AudioFileClip,
    CompositeAudioClip,
    vfx,
)

# Use your PIL-based overlay from utils.py (no ImageMagick dependency)
try:
    from utils import add_text_overlay
except Exception:
    add_text_overlay = None  # overlay will be skipped if utils is missing

# --- Transition support (graceful fallback if transition.py is missing) ---
try:
    from transition import apply_transition as _apply_transition_mod, list_available_transitions as _list_transitions_mod
    HAS_TRANSITIONS = True
except Exception:
    HAS_TRANSITIONS = False
    _apply_transition_mod = None
    _list_transitions_mod = None

    def _list_transitions_fallback() -> List[str]:
        return ["crossfade"]

    def _compose_crossfade_segments(clip1: VideoFileClip, clip2: VideoFileClip, duration: float):
        """
        Build a crossfade overlap segment using only MoviePy primitives.
        Returns [pre_clip1, overlap_composite, post_clip2]
        """
        d = min(duration, max(0.1, clip1.duration, clip2.duration))
        pre = clip1.subclip(0, max(clip1.duration - d, 0.0))
        t0 = pre.duration

        a = clip1.subclip(max(clip1.duration - d, 0.0)).crossfadeout(d).set_start(t0)
        b = clip2.subclip(0, d).crossfadein(d).set_start(t0)
        overlap = CompositeVideoClip([a, b]).set_duration(t0 + d)
        post = clip2.subclip(d)
        return [pre, overlap, post]

    def _apply_transition_fallback(clip1: VideoFileClip, clip2: VideoFileClip, transition_type: str, duration: float):
        # We only support crossfade in fallback
        return _compose_crossfade_segments(clip1, clip2, duration)

# --- Whisper transcription (lazy load) ---
try:
    from whisper import load_model
except Exception:
    load_model = None  # If whisper not installed, we'll raise when used.

_model = None


def _get_whisper():
    global _model
    if load_model is None:
        raise RuntimeError(
            "openai-whisper is not installed. Add 'openai-whisper' to requirements.txt."
        )
    if _model is None:
        # Choose "base" for reasonable quality/speed; switch to "tiny" if build timeouts
        _model = load_model("base")
    return _model


def transcribe_videos(video_paths: List[str], openai_api_key: str | None = None) -> List[dict]:
    """
    Transcribe the given list of video files using local Whisper.
    Returns a list of dicts: { 'path': str, 'text': str }
    """
    model = _get_whisper()
    out = []
    for p in video_paths:
        try:
            result = model.transcribe(p)
            out.append({"path": p, "text": (result.get("text") or "").strip()})
        except Exception as e:
            print(f"❌ Transcription failed for {p}: {e}")
    return out


def generate_video(
    clip_paths: List[str],
    storyline: str,
    transition_duration: float = 1.0,
    tone: str = "Cinematic",
    mix_original_audio: bool = False,
    show_opening_card: bool = True,
) -> str:
    """
    Builds a final video by:
      - loading/shortening clips to make room for transitions
      - applying a random transition between clips
      - optionally adding an opening text overlay (PIL-based, no ImageMagick)
      - optionally mixing in background music, with optional ducking of original audio
      - exporting to a temp MP4 and returning its path
    """
    if not clip_paths:
        raise ValueError("No clip paths provided.")

    # Load usable clips
    clips: List[VideoFileClip] = []
    for path in clip_paths:
        try:
            clip = VideoFileClip(path)

            # Prevent overrun at ends: trim a small buffer off the tail for transitions
            trim_buffer = max(transition_duration, 0.5)
            safe_duration = max(clip.duration - trim_buffer, 0.5)
            clip = clip.subclip(0, safe_duration)

            if clip.duration >= 1.0:
                clips.append(clip)
            else:
                print(f"⚠️ Skipping short clip: {path} ({clip.duration:.2f}s)")
        except Exception as e:
            print(f"❌ Skipping corrupted/unreadable clip: {path} | Error: {e}")

    if not clips:
        raise ValueError("No usable clips after loading/validation.")

    # Prepare transitions
    if HAS_TRANSITIONS:
        list_available_transitions = _list_transitions_mod
        apply_transition = _apply_transition_mod
    else:
        list_available_transitions = _list_transitions_fallback
        apply_transition = _apply_transition_fallback

    available_transitions = list_available_transitions() or ["crossfade"]

    # Build a chain using proper overlaps returned by apply_transition (list of segments)
    final_segments: List[VideoFileClip] = [clips[0]]  # start with first clip
    for i in range(1, len(clips)):
        transition_name = random.choice(available_transitions)
        try:
            segments = apply_transition(final_segments[-1], clips[i], transition_name, transition_duration)
            # Replace last element with the segments (pre, overlap, post)
            final_segments = final_segments[:-1] + segments
        except Exception as e:
            print(f"⚠️ Transition '{transition_name}' failed; falling back to cut: {e}")
            final_segments.append(clips[i])

    # Concatenate everything (no negative padding tricks)
    final = concatenate_videoclips(final_segments, method="compose")

    # Optional opening card (PIL overlay to avoid ImageMagick)
    title_text = None
    if show_opening_card and tone and tone.lower() == "cinematic":
        title_text = "KEEMOGRAPHY PRESENTS"

    if title_text and add_text_overlay is not None:
        try:
            # Create a 3s card by overlaying on a solid black base that precedes the main video
            base = final.set_start(3)  # shift main content right by 3 seconds
            black_bg = final.fx(vfx.colorx, 0).subclip(0, 3)  # darken copy as "black" background
            card = add_text_overlay(black_bg, title_text, fontsize=64, color="white")
            final = CompositeVideoClip([base, card]).set_duration(base.duration)
        except Exception as e:
            print(f"ℹ️ Skipping opening card (overlay error): {e}")

    # Background music (with optional ducking of original audio)
    music_path = _get_music_for_tone(tone)
    if music_path:
        try:
            bg = AudioFileClip(music_path).volumex(0.20)  # base level
            bg = bg.subclip(0, min(bg.duration, final.duration))
            if mix_original_audio and final.audio is not None:
                # Simple ducking (lower bg a bit more)
                mix = CompositeAudioClip([final.audio.volumex(1.0), bg.volumex(0.6)])
                final = final.set_audio(mix)
            else:
                final = final.set_audio(bg)
        except Exception as e:
            print(f"⚠️ Could not load or set background music: {e}")

    # Write the final video to a temp file
    temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    try:
        final.write_videofile(
            temp_out.name,
            codec="libx264",
            audio_codec="aac",
            fps=24,
            threads=os.cpu_count() or 2,
            logger=None,  # reduce noisy logs on Streamlit
        )
    finally:
        # Close readers to release file handles (important on Streamlit/Docker)
        try:
            final.close()
        except Exception:
            pass
        for c in clips:
            try:
                c.close()
            except Exception:
                pass

    return temp_out.name


def _get_music_for_tone(tone: str | None) -> str | None:
    """
    Map tone → local asset path. Returns None if the file is missing.
    """
    tone = (tone or "").lower()
    music_map = {
        "cinematic": "assets/music/cinematic.mp3",
        "energetic": "assets/music/energetic.mp3",
        "sentimental": "assets/music/sentimental.mp3",
        "epic": "assets/music/epic.mp3",
        "calm": "assets/music/calm.mp3",
    }
    path = music_map.get(tone)
    return path if path and os.path.exists(path) else None
