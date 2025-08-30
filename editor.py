# editor.py

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
# -------------------------------------------------------------------------

import os
import tempfile
import random
from typing import List, Dict, Optional

from transition import apply_transition, list_available_transitions


# --- OpenAI API key helper (works with .env or Streamlit Secrets) ---
def _get_api_key(explicit_key: Optional[str] = None) -> Optional[str]:
    if explicit_key:
        return explicit_key
    return os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")


# --- Transcription via OpenAI Whisper API (no local torch/whisper needed) ---
def transcribe_videos(
    video_paths: List[str],
    openai_api_key: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Transcribe each video using OpenAI Whisper API (model='whisper-1').
    Returns: list of dicts like {"path": <path>, "text": <transcript>}
    """
    from openai import OpenAI  # lightweight import

    api_key = _get_api_key(openai_api_key)
    if not api_key:
        raise RuntimeError(
            "Missing OpenAI API key. Set API_KEY or OPENAI_API_KEY in env/secrets."
        )

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
            # Don't crash entire job; record empty text and continue.
            print(f"⚠️ Transcription failed for {path}: {e}")
            text = ""
        results.append({"path": path, "text": text})

    return results


# --- Video generation ---
def generate_video(
    clip_paths: List[str],
    storyline: str,
    transition_duration: float = 1.0,
    tone: str = "Cinematic",
    mix_original_audio: bool = False,
    show_opening_card: bool = True,
) -> str:
    """
    Builds a final video by stitching clips with random transitions,
    optional title card, and background music.
    """

    # Lazy-import MoviePy so this module can be imported even if deps
    # weren’t present at process start (bootstrap above will install them).
    from moviepy.editor import (
        VideoFileClip,
        concatenate_videoclips,
        CompositeVideoClip,
        CompositeAudioClip,
        TextClip,
        AudioFileClip,
    )

    clips: List["VideoFileClip"] = []

    # Load usable clips with safety trim for transitions
    for path in clip_paths:
        try:
            clip = VideoFileClip(path)

            # Prevent overflow at clip end during transitions
            trim_buffer = max(transition_duration, 0.5)
            safe_duration = max(clip.duration - trim_buffer, 0.5)
            clip = clip.subclip(0, safe_duration)

            if clip.duration < 1.0:
                print(f"⚠️ Skipping clip {path} (too short: {clip.duration:.2f}s)")
                continue

            clips.append(clip)

        except Exception as e:
            print(f"❌ Skipping corrupted/unreadable clip: {path} | Error: {e}")

    if not clips:
        raise ValueError("No usable clips provided to generate_video().")

    available_transitions = list_available_transitions()

    # Add fade-in to first clip
    final_clips: List["VideoFileClip"] = [clips[0].fadein(transition_duration)]

    # Apply a transition between each pair
    for i in range(1, len(clips)):
        transition = random.choice(available_transitions)
        try:
            transitioned = apply_transition(
                final_clips[-1], clips[i], transition, transition_duration
            )
            final_clips[-1] = transitioned.set_end(transitioned.duration)
            final_clips.append(clips[i].fadein(transition_duration))
        except Exception as e:
            print(f"⚠️ Failed transition for clip {i} ({transition}): {e}")
            final_clips.append(clips[i].fadein(transition_duration))

    # Compose the full video (negative padding to blend crossfades)
    final = concatenate_videoclips(
        final_clips, method="compose", padding=-transition_duration
    )

    # Optional intro overlay
    if show_opening_card and tone.lower() == "cinematic":
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
            # TextClip requires ImageMagick; if unavailable, just skip.
            print(f"ℹ️ Skipping TextClip overlay (likely no ImageMagick): {e}")
            overlays = []

        if overlays:
            final = CompositeVideoClip([final] + overlays)

    # Background music based on tone (if assets exist)
    music_path = _get_music_for_tone(tone)
    bg_music_clip = None
    if music_path and os.path.exists(music_path):
        try:
            bg_music_clip = AudioFileClip(music_path).volumex(0.2)
            bg_music_clip = bg_music_clip.subclip(0, min(bg_music_clip.duration, final.duration))
        except Exception as e:
            print(f"⚠️ Could not load background music: {e}")
            bg_music_clip = None

    # Audio mixing logic
    try:
        if bg_music_clip and final.audio is not None:
            if mix_original_audio:
                # Keep original audio AND music under it (simple ducking by lowering music)
                mixed = CompositeAudioClip([
                    final.audio.volumex(1.0),
                    bg_music_clip.volumex(0.15),  # duck music under voices
                ])
                final = final.set_audio(mixed)
            else:
                # Replace with background music only
                final = final.set_audio(bg_music_clip)
        elif bg_music_clip and final.audio is None:
            final = final.set_audio(bg_music_clip)
        # else: keep whatever audio is there
    except Exception as e:
        print(f"⚠️ Audio mix failed, keeping existing audio: {e}")

    # Write to a temp mp4
    temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    try:
        final.write_videofile(
            temp_out.name,
            codec="libx264",
            audio_codec="aac",
            fps=24,
            threads=2,
            preset="medium",
        )
    except Exception as e:
        print(f"❌ Failed to write video file: {e}")
        raise
    finally:
        # Clean up MoviePy readers
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
