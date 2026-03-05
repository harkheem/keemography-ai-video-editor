# app_utils.py
# Shared utilities extracted from app.py — no Streamlit dependency.
# Used by both the legacy Streamlit app and the new FastAPI backend.

import os
import re
import gc
import subprocess
import tempfile
import time
from typing import List, Optional


WHISPER_MAX_MB = 24  # OpenAI Whisper API limit (24 MB for safety)


# ---------------------------------------------------------------------------
# Video chunking
# ---------------------------------------------------------------------------

def split_video_to_chunks(path: str, max_mb: float = WHISPER_MAX_MB) -> List[str]:
    """
    Splits media into audio-only WAV chunks each under max_mb.
    Returns a list of temp WAV file paths.
    """

    def _run_ffmpeg(cmd):
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ffmpeg failed: {e.stderr or e.stdout}")

    def _probe_duration_seconds(video_path: str) -> float:
        probe_cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path,
        ]
        try:
            result = subprocess.run(probe_cmd, check=True, capture_output=True, text=True)
            return float((result.stdout or "").strip())
        except Exception as e:
            raise RuntimeError(f"Could not determine video duration with ffprobe: {e}")

    def _probe_has_audio(media_path: str) -> bool:
        probe_cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=codec_type",
            "-of", "csv=p=0",
            media_path,
        ]
        try:
            result = subprocess.run(probe_cmd, check=True, capture_output=True, text=True)
            return "audio" in (result.stdout or "").lower()
        except Exception:
            return False

    def _probe_format_name(media_path: str) -> str:
        probe_cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=format_name",
            "-of", "default=noprint_wrappers=1:nokey=1",
            media_path,
        ]
        try:
            result = subprocess.run(probe_cmd, check=True, capture_output=True, text=True)
            return (result.stdout or "").strip().lower()
        except Exception:
            return ""

    duration = _probe_duration_seconds(path)
    if not _probe_has_audio(path):
        return [path]

    max_bytes = int(max_mb * 1024 * 1024)
    target_bytes = int(max_bytes * 0.90)
    est_audio_bytes_per_second = 32000.0  # PCM 16-bit mono 16 kHz WAV ≈ 32 000 B/s
    base_chunk_duration = max(15.0, min(1200.0, target_bytes / est_audio_bytes_per_second))
    out_paths: List[str] = []
    start = 0.0

    while start < duration - 0.01:
        seg_dur = min(base_chunk_duration, duration - start)
        temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name

        attempt = 0
        while True:
            audio_cmd = [
                "ffmpeg", "-y",
                "-ss", f"{start:.3f}",
                "-t", f"{seg_dur:.3f}",
                "-i", path,
                "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
                temp_path,
            ]
            _run_ffmpeg(audio_cmd)

            produced_size = os.path.getsize(temp_path) if os.path.exists(temp_path) else 0
            fmt = _probe_format_name(temp_path)
            is_valid_wav = "wav" in fmt
            if produced_size > 2048 and produced_size <= target_bytes and is_valid_wav:
                break

            attempt += 1
            if attempt >= 6:
                raise RuntimeError(
                    f"Unable to create valid chunk after retries: {temp_path} "
                    f"({produced_size} bytes, format='{fmt}')"
                )
            seg_dur = max(0.20, seg_dur * 0.80)

        out_paths.append(temp_path)
        start += seg_dur

    return out_paths


# ---------------------------------------------------------------------------
# Transcription (with automatic splitting)
# ---------------------------------------------------------------------------

def transcribe_videos_with_split(
    video_paths: List[str],
    openai_api_key: Optional[str] = None,
    progress_callback=None,
) -> List[dict]:
    """
    Transcribe each video, splitting into WAV chunks if necessary.
    Returns list of dicts {path, text}.
    """
    from openai import OpenAI

    client = OpenAI(api_key=openai_api_key)
    results: List[dict] = []
    chunk_map: dict = {}

    for path in video_paths:
        try:
            chunk_map[path] = split_video_to_chunks(path)
        except Exception as e:
            print(f"⚠️ Split failed for {path}: {repr(e)}")
            chunk_map[path] = []

    total_chunks = sum(len(chunks) for chunks in chunk_map.values())
    processed_chunks = 0

    for path in video_paths:
        try:
            chunks = chunk_map.get(path, [])
            chunk_results = []
            for chunk in chunks:
                try:
                    tx = None
                    last_exc = None
                    for attempt in range(2):
                        try:
                            with open(chunk, "rb") as f:
                                tx = client.audio.transcriptions.create(
                                    model="whisper-1", file=f
                                )
                            break
                        except Exception as e:
                            last_exc = e
                            err = str(e).lower()
                            if (
                                ("error parsing the body" in err or "invalid file format" in err)
                                and attempt == 0
                            ):
                                time.sleep(0.5)
                                continue
                            raise
                    if tx is None and last_exc is not None:
                        raise last_exc
                    text = getattr(tx, "text", "") or ""
                    chunk_results.append(text)
                except Exception as e:
                    print(f"⚠️ Transcription failed for chunk {chunk}: {repr(e)}")
                    chunk_results.append("")
                finally:
                    processed_chunks += 1
                    if callable(progress_callback):
                        try:
                            progress_callback(processed_chunks, total_chunks)
                        except Exception:
                            pass
                    if chunk != path:
                        try:
                            os.remove(chunk)
                        except Exception:
                            pass
            results.append({"path": path, "text": " ".join(chunk_results)})
        except Exception as e:
            print(f"⚠️ Transcription failed for {path}: {repr(e)}")
            results.append({"path": path, "text": ""})

    return results


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def normalize_drive_dropbox(url: str) -> str:
    u = url.strip()
    import re
    m = re.search(r"https?://drive\.google\.com/file/d/([a-zA-Z0-9_-]+)", u)
    if m:
        return f"https://drive.google.com/uc?export=download&id={m.group(1)}"
    if "dropbox.com" in u:
        if "?dl=0" in u:
            return u.replace("?dl=0", "?dl=1")
        if "?dl=1" not in u:
            return u + "?dl=1"
    return u


def probe_duration_seconds(path: str) -> float:
    """Return clip duration via ffprobe; 0.0 on failure."""
    if not path or not os.path.exists(path):
        return 0.0
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    try:
        res = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return float((res.stdout or "").strip())
    except Exception:
        return 0.0


def filter_existing(paths: List[str]):
    """Split paths into (good, bad) where good exist and are > 1 KB."""
    good, bad = [], []
    for p in paths:
        exists = os.path.exists(p)
        size = os.path.getsize(p) if exists else 0
        if exists and size > 1024:
            good.append(p)
        else:
            bad.append((p, exists, size))
    return good, bad


def normalize_to_paths(maybe_list) -> List[str]:
    """Convert scoring output (list[str|dict|tuple]) → list[str] of file paths."""
    out = []
    if not maybe_list:
        return out
    for item in maybe_list:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict) and item.get("path"):
            out.append(item["path"])
        elif isinstance(item, (list, tuple)) and item and isinstance(item[0], str):
            out.append(item[0])
    seen: set = set()
    deduped = []
    for p in out:
        if p not in seen:
            seen.add(p)
            deduped.append(p)
    return deduped
