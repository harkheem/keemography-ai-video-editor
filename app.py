# =========================
# app.py  (FULL, CORRECTED)
# =========================

# ---- bootstrap critical deps if the build missed them ----
import sys, subprocess, importlib.util

def ensure(spec: str, import_name: str | None = None):
    name = import_name or spec.split("==")[0].split(">=")[0].split("[")[0]
    if importlib.util.find_spec(name) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", spec])

# minimal set for your editor imports
ensure("moviepy==2.1.1", "moviepy")
ensure("imageio-ffmpeg>=0.5.1", "imageio_ffmpeg")
ensure("imageio>=2.34.0", "imageio")
ensure("Pillow>=10.4.0", "PIL")
ensure("numpy>=2.0.2", "numpy")
# ----------------------------------------------------------

import os
import gc
import shutil
import tempfile
from datetime import datetime, timedelta
import time

import threading
import streamlit as st
from dotenv import load_dotenv
import requests, re
import psutil

from editor import generate_video

# --- Whisper API file size limit (MB) ---
WHISPER_MAX_MB = 24  # OpenAI Whisper API limit (use 24MB for safety)

def split_video_to_chunks(path, max_mb=WHISPER_MAX_MB):
    """
    Splits media into audio-only WAV chunks under max_mb using ffmpeg subprocess.
    Returns a list of temp file paths.
    """
    import os

    def _run_ffmpeg(cmd):
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ffmpeg failed: {e.stderr or e.stdout}")

    def _probe_duration_seconds(video_path: str) -> float:
        probe_cmd = [
            "ffprobe",
            "-v", "error",
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
            "ffprobe",
            "-v", "error",
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
            "ffprobe",
            "-v", "error",
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
    # PCM 16-bit mono 16kHz WAV ≈ 32,000 bytes/sec
    est_audio_bytes_per_second = 32000.0
    base_chunk_duration = max(15.0, min(1200.0, target_bytes / est_audio_bytes_per_second))
    out_paths = []
    start = 0.0

    while start < duration - 0.01:
        seg_dur = min(base_chunk_duration, duration - start)
        temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name

        attempt = 0
        while True:
            audio_cmd = [
                "ffmpeg",
                "-y",
                "-ss", f"{start:.3f}",
                "-t", f"{seg_dur:.3f}",
                "-i", path,
                "-vn",
                "-ac", "1",
                "-ar", "16000",
                "-c:a", "pcm_s16le",
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
                    f"Unable to create valid chunk under size limit after retries: {temp_path} ({produced_size} bytes, format='{fmt}')"
                )
            seg_dur = max(0.20, seg_dur * 0.80)

        out_paths.append(temp_path)
        start += seg_dur

    return out_paths

def transcribe_videos_with_split(video_paths, openai_api_key=None, progress_callback=None):
    """
    Transcribe each video, splitting if needed. Returns list of dicts {path, text}.
    """
    results = []
    chunk_map = {}
    from openai import OpenAI
    client = OpenAI(api_key=openai_api_key)

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
                                    model="whisper-1",
                                    file=f
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

from scoring import score_clips_with_story


# ---------------- ENV / SECRETS ----------------
load_dotenv()

def _get_key():
    return (
        os.getenv("API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or (st.secrets.get("API_KEY") if hasattr(st, "secrets") else None)
        or (st.secrets.get("OPENAI_API_KEY") if hasattr(st, "secrets") else None)
    )

OPENAI_API_KEY = _get_key()


# ---------------- PAGE CONFIG / THEME ----------------
st.set_page_config(
    page_title="Keemography · AI Video Editor",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS / Theme ────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

html, body, .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background: #080808 !important;
    color: #e8e8e8;
}
.block-container { padding-top: 0 !important; padding-bottom: 3rem; max-width: 1100px !important; }
[data-testid="stSidebar"] { display: none !important; }

/* ── Header ────────────────────────────────────────────────────────────── */
.kee-header {
    background: linear-gradient(135deg, #0e0e0e 0%, #150808 100%);
    border-bottom: 1px solid #1e1e1e;
    padding: 26px 0 20px;
    margin: 0 -4rem 2.5rem;
    text-align: center;
}
.kee-title { font-size: 1.55rem; font-weight: 900; letter-spacing: 0.14em;
             text-transform: uppercase; color: #fff; }
.kee-sub   { font-size: 0.72rem; color: #555; letter-spacing: 0.2em;
             text-transform: uppercase; margin-top: 4px; }
.kee-red   { color: #e63939; }

/* ── Section labels ────────────────────────────────────────────────────── */
.step-label { font-size: 0.63rem; font-weight: 700; letter-spacing: 0.22em;
              text-transform: uppercase; color: #e63939; margin-bottom: 4px; }
.sec-title  { font-size: 0.98rem; font-weight: 700; color: #fff; margin-bottom: 12px; }
.kee-hr     { border: none; border-top: 1px solid #181818; margin: 20px 0; }

/* ── Card panels ───────────────────────────────────────────────────────── */
.kee-panel {
    background: rgba(255,255,255,0.026);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 20px 22px;
    margin-bottom: 14px;
}

/* ── Inputs ────────────────────────────────────────────────────────────── */
.stTextArea textarea, .stTextInput > div > input {
    background: #0d0d0d !important; color: #f0f0f0 !important;
    border: 1px solid #242424 !important; border-radius: 10px !important;
    font-size: 0.9rem !important;
}
.stTextArea textarea:focus, .stTextInput > div > input:focus {
    border-color: #e63939 !important;
    box-shadow: 0 0 0 2px rgba(230,57,57,0.14) !important;
}
.stTextArea textarea::placeholder, .stTextInput > div > input::placeholder { color: #3c3c3c !important; }
label[data-testid="stWidgetLabel"] p, div[data-testid="stWidgetLabel"] p {
    color: #888 !important; font-size: 0.79rem !important; font-weight: 500 !important; }

/* ── Selectbox ─────────────────────────────────────────────────────────── */
.stSelectbox div[data-baseweb="select"] > div {
    background: #0d0d0d !important; border: 1px solid #242424 !important;
    border-radius: 10px !important; color: #f0f0f0 !important;
}

/* ── Slider ────────────────────────────────────────────────────────────── */
/* track rail (dark) */
[data-testid="stSlider"] [data-baseweb="slider"] > div > div:first-child {
    background: #252525 !important; border-radius: 4px !important; }
/* filled portion (red) */
[data-testid="stSlider"] [data-baseweb="slider"] > div > div:nth-child(2) {
    background: #e63939 !important; border-radius: 4px !important; }
/* thumb */
[data-testid="stSlider"] [role="slider"] {
    background: #e63939 !important;
    border-color: #e63939 !important;
    box-shadow: 0 0 0 4px rgba(230,57,57,0.22) !important;
    width: 16px !important; height: 16px !important;
}
/* current value label */
[data-testid="stSlider"] p { color: #aaa !important; font-size: 0.8rem !important; }
/* min / max tick labels */
[data-testid="stSlider"] [data-testid="stTickBarMin"],
[data-testid="stSlider"] [data-testid="stTickBarMax"] {
    color: #444 !important; font-size: 0.72rem !important;
}

/* ── Toggle ────────────────────────────────────────────────────────────── */
[data-testid="stToggle"] span[aria-checked="true"] { background: #e63939 !important; }

/* ── File uploader ─────────────────────────────────────────────────────── */
[data-testid="stFileUploaderDropzone"] {
    background: #0a0a0a !important; border: 1.5px dashed #252525 !important;
    border-radius: 14px !important;
}
[data-testid="stFileUploaderDropzone"]:hover { border-color: #e63939 !important; }

/* ── Buttons ───────────────────────────────────────────────────────────── */
.stButton > button {
    background: #111 !important; color: #ddd !important;
    border: 1px solid #252525 !important; border-radius: 10px !important;
    font-weight: 600 !important; font-size: 0.85rem !important; transition: all 0.15s !important;
}
.stButton > button:hover { border-color: #e63939 !important; color: #fff !important; background: #1a0808 !important; }
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #b52828 0%, #e63939 100%) !important;
    border: none !important; color: #fff !important; font-size: 1.05rem !important;
    font-weight: 800 !important; letter-spacing: 0.06em !important;
    border-radius: 13px !important; box-shadow: 0 4px 22px rgba(230,57,57,0.38) !important;
}
.stButton > button[kind="primary"]:hover { box-shadow: 0 6px 30px rgba(230,57,57,0.58) !important; }
.stDownloadButton > button {
    background: linear-gradient(135deg, #0a1f0a 0%, #0e2e0e 100%) !important;
    border: 1px solid #1a5c1a !important; color: #6de06d !important;
    border-radius: 12px !important; font-weight: 700 !important;
    box-shadow: 0 4px 16px rgba(20,100,20,0.25) !important;
}

/* ── Progress bar ──────────────────────────────────────────────────────── */
.stProgress > div > div { background: #181818 !important; border-radius: 100px !important; }
.stProgress > div > div > div {
    background: linear-gradient(90deg, #b52828, #e63939, #ff6b6b) !important;
    border-radius: 100px !important;
}

/* ── Expanders ─────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid #1c1c1c !important; border-radius: 12px !important; background: #0c0c0c !important;
}

/* ── Metrics ───────────────────────────────────────────────────────────── */
[data-testid="stMetricValue"] { color: #e63939 !important; font-weight: 800 !important; }
[data-testid="stMetricLabel"] p { color: #777 !important; font-size: 0.76rem !important; }

/* ── Alerts ────────────────────────────────────────────────────────────── */
.stAlert { border-radius: 12px !important; }

/* ── Caption / small text ──────────────────────────────────────────────── */
.stCaption, .stCaption p { color: #666 !important; font-size: 0.77rem !important; }

/* ── Stage pills ───────────────────────────────────────────────────────── */
.stage-row { display: flex; gap: 8px; margin-bottom: 16px; flex-wrap: wrap; }
.stage-pill { font-size: 0.67rem; font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase; padding: 4px 13px; border-radius: 100px;
    border: 1px solid #222; color: #444; background: #0e0e0e; transition: all 0.3s; }
.stage-pill.active { background: #1a0808; border-color: #e63939; color: #ff7070;
    box-shadow: 0 0 14px rgba(230,57,57,0.2); }
.stage-pill.done { background: #0a160a; border-color: #1a5c1a; color: #5ed85e; }

/* ── Clip analysis cards ───────────────────────────────────────────────── */
.clip-row { display: flex; align-items: center; gap: 9px; padding: 9px 13px;
    border-radius: 11px; background: #0d0d0d; border: 1px solid #1c1c1c; margin-bottom: 7px; }
.clip-badge { font-size: 0.61rem; font-weight: 700; padding: 2px 8px; border-radius: 6px;
    text-transform: uppercase; letter-spacing: 0.05em; flex-shrink: 0; }
.badge-hook        { background: #2a0e00; color: #ff8c2a; border: 1px solid #6b3000; }
.badge-payoff      { background: #18002e; color: #c987ff; border: 1px solid #56228a; }
.badge-turn        { background: #00102e; color: #54b4ff; border: 1px solid #1a446e; }
.badge-development { background: #0a160a; color: #68c868; border: 1px solid #26502a; }
.badge-broll       { background: #161616; color: #888;    border: 1px solid #2e2e2e; }
.badge-emo { font-size: 0.59rem; font-weight: 600; padding: 2px 7px; border-radius: 6px;
    background: #141414; color: #999; border: 1px solid #222; flex-shrink: 0; }
.clip-name  { font-size: 0.8rem; font-weight: 600; color: #ddd; flex: 1; min-width: 0;
              white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.clip-score { font-size: 0.75rem; color: #e63939; font-weight: 700; flex-shrink: 0; }
.clip-trim  { font-size: 0.69rem; color: #555; flex-shrink: 0; }

/* ── Output card ───────────────────────────────────────────────────────── */
.output-card { background: linear-gradient(135deg, #0b1508 0%, #080d08 100%);
    border: 1px solid #1a3a1a; border-radius: 16px; padding: 24px; margin-top: 20px;
    box-shadow: 0 8px 40px rgba(0,0,0,0.5); }
.output-title { font-size: 1.05rem; font-weight: 800; color: #6de06d;
    letter-spacing: 0.05em; margin-bottom: 14px; }

/* ── Settings card ─────────────────────────────────────────────────────── */
.settings-card {
    background: #0d0d0d;
    border: 1px solid #1e1e1e;
    border-radius: 14px;
    padding: 18px 20px;
    margin-bottom: 16px;
}
</style>
""", unsafe_allow_html=True)

# ── App Header ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="kee-header">
  <div class="kee-title">Keem<span class="kee-red">ography</span></div>
  <div class="kee-sub">AI Video Editor</div>
</div>
""", unsafe_allow_html=True)


# ---------------- HELPERS ----------------
def _too_big(file, limit_mb: int) -> bool:
    size = getattr(file, "size", None)
    return bool(size and size > limit_mb * 1024 * 1024)

def _normalize_drive_dropbox(url: str) -> str:
    u = url.strip()
    m = re.search(r"https?://drive\.google\.com/file/d/([a-zA-Z0-9_-]+)", u)
    if m:
        return f"https://drive.google.com/uc?export=download&id={m.group(1)}"
    if "dropbox.com" in u:
        if "?dl=0" in u:
            return u.replace("?dl=0", "?dl=1")
        if "?dl=1" not in u:
            return u + "?dl=1"
    return u

def _normalize_to_paths(maybe_list):
    """
    Forces scoring output into list[str] of file paths.
    Supports: list[str], list[dict{path}], list[tuple(path, ...)]
    """
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
    # de-dupe preserving order
    seen = set()
    deduped = []
    for p in out:
        if p not in seen:
            seen.add(p)
            deduped.append(p)
    return deduped

def _filter_existing(paths):
    good = []
    bad = []
    for p in paths:
        exists = os.path.exists(p)
        size = os.path.getsize(p) if exists else 0
        if exists and size > 1024:
            good.append(p)
        else:
            bad.append((p, exists, size))
    return good, bad

def _probe_duration_seconds(path: str) -> float:
    """Best-effort ffprobe duration read; returns 0.0 on failure."""
    if not path or not os.path.exists(path):
        return 0.0
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    try:
        res = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return float((res.stdout or "").strip())
    except Exception:
        return 0.0


# ---------------- STATE ----------------
if "fetched_paths" not in st.session_state:
    st.session_state.fetched_paths = []


# ── Memory check ──────────────────────────────────────────────────────────
_mem = psutil.virtual_memory()
if _mem.percent > 85:
    st.warning(f"⚠️ Memory at {_mem.percent}% — consider fewer/smaller clips or closing other apps.")

# ══════════════════════════════════════════════════════════════════════════════
#  LEFT  |  RIGHT  two-column master layout
# ══════════════════════════════════════════════════════════════════════════════
_col_left, _col_right = st.columns([0.48, 0.52], gap="large")

# ─────────────────────────────────────────────────────────────────────────────
#  LEFT COLUMN : Story · Footage · Music · Keywords · Generate
# ─────────────────────────────────────────────────────────────────────────────
with _col_left:

    # ── Story ──────────────────────────────────────────────────────────────
    st.markdown('<p class="step-label">① Story</p>', unsafe_allow_html=True)
    storyline = st.text_area(
        "What is this video about?",
        height=110,
        placeholder="e.g. A behind-the-scenes look at a golden-hour photoshoot — raw, emotional, cinematic.",
        help="The AI uses this to rank clips and decide how to arrange them.",
        label_visibility="collapsed",
    )

    st.markdown('<hr class="kee-hr">', unsafe_allow_html=True)

    # ── Footage ────────────────────────────────────────────────────────────
    st.markdown('<p class="step-label">② Footage</p>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload MP4 / MPEG4 clips",
        type=["mp4", "mpeg4"],
        accept_multiple_files=True,
        help="No file-count limit. Up to 2 GB each.",
    )
    urls = st.text_area(
        "Or paste direct video URLs",
        placeholder="https://.../clip1.mp4\nhttps://.../clip2.mp4",
        height=68,
        help="Google Drive share links and Dropbox links are auto-converted to direct downloads.",
    )
    _fc1, _fc2 = st.columns([1, 1])
    fetch_clicked = _fc1.button("⬇ Fetch URLs", use_container_width=True)
    clear_fetched = _fc2.button("🗑 Clear", use_container_width=True)

    if clear_fetched:
        st.session_state.fetched_paths = []
    if fetch_clicked and urls.strip():
        st.session_state.fetched_paths = []
        url_list = [u.strip() for u in re.split(r"[,\n]+", urls) if u.strip()]
        for u in url_list:
            direct = _normalize_drive_dropbox(u)
            try:
                st.caption(f"Fetching {direct[:60]}…")
                with requests.get(direct, stream=True, timeout=1200) as r:
                    r.raise_for_status()
                    total = int(r.headers.get("content-length", 0))
                    _prog = st.progress(0.0)
                    downloaded = 0
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as _ff:
                        for chunk in r.iter_content(chunk_size=1024 * 1024 * 16):
                            if chunk:
                                _ff.write(chunk)
                                downloaded += len(chunk)
                                if total:
                                    _prog.progress(min(1.0, downloaded / total))
                        saved = _ff.name
                st.session_state.fetched_paths.append(saved)
                st.success(f"Saved: {os.path.basename(saved)}")
            except Exception as e:
                st.error(f"Download failed: {e}")

    if st.session_state.fetched_paths:
        with st.expander(f"{len(st.session_state.fetched_paths)} URL clip(s) ready"):
            for p in st.session_state.fetched_paths:
                st.caption(p)

    st.markdown('<hr class="kee-hr">', unsafe_allow_html=True)

    # ── Music ──────────────────────────────────────────────────────────────
    st.markdown('<p class="step-label">③ Music  <span style="color:#555;font-weight:400;font-size:0.75rem;text-transform:none;letter-spacing:normal">(optional)</span></p>', unsafe_allow_html=True)
    user_music_file = st.file_uploader(
        "Upload MP3 / WAV",
        type=["mp3", "wav", "m4a", "aac", "ogg"],
        accept_multiple_files=False,
        key="music_upload",
        help="Used for beat-aligned editing. Max 100 MB. If omitted, a default soundtrack is chosen by tone.",
    )
    user_music_path = None
    if user_music_file:
        if _too_big(user_music_file, 100):
            st.warning("Max 100 MB.")
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(user_music_file.name)[-1]) as tmp:
                user_music_file.seek(0)
                tmp.write(user_music_file.read())
                user_music_path = tmp.name
            st.success(f"✓ {user_music_file.name}")

    st.markdown('<hr class="kee-hr">', unsafe_allow_html=True)

    # ── Keyword filters ────────────────────────────────────────────────────
    with st.expander("🔍 Keyword filters  (optional)"):
        _kc1, _kc2 = st.columns(2)
        user_priority_keywords = _kc1.text_input(
            "Prioritize",
            placeholder="dance, culture, joy",
            help="Clips matching these are ranked higher.",
        )
        user_excluded_keywords = _kc2.text_input(
            "Exclude",
            placeholder="blurry, quiet",
            help="Clips matching these are filtered out.",
        )

    st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)

    # ── Generate button ────────────────────────────────────────────────────
    run = st.button("⬡  Generate Video", type="primary", use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
#  RIGHT COLUMN : Settings · Preview
# ─────────────────────────────────────────────────────────────────────────────
with _col_right:

    # ── Settings ───────────────────────────────────────────────────────────
    st.markdown('<p class="step-label">④ Settings</p>', unsafe_allow_html=True)

    tone = st.selectbox(
        "Tone",
        ["Cinematic", "Energetic", "Sentimental", "Epic", "Calm"],
        help="Controls pacing, clip length, and music selection.",
    )
    target_duration_sec = st.slider(
        "Target duration (seconds)",
        min_value=15, max_value=180, value=45, step=5,
        help="The AI will try to hit this length. Actual output may differ by a few seconds.",
    )
    transition_duration = st.slider(
        "Transition length (seconds)",
        min_value=0.15, max_value=1.5, value=0.3, step=0.05,
        help="Shorter = snappier cut. Longer = smoother crossfade.",
    )

    _tc1, _tc2 = st.columns(2)
    mix_original_audio = _tc1.toggle("Mix original audio", value=False,
        help="Blend the clip's own audio under the music.")
    show_opening_card  = _tc2.toggle("Show opening card",  value=True,
        help="Adds a title card at the start of the video.")

    st.markdown('<hr class="kee-hr">', unsafe_allow_html=True)

    # ── Preview ────────────────────────────────────────────────────────────
    st.markdown('<p class="step-label">⑤ Preview</p>', unsafe_allow_html=True)
    _preview_path = None
    _fetched_for_preview = st.session_state.get("fetched_paths", [])
    if _fetched_for_preview:
        _preview_path = _fetched_for_preview[0]
    elif uploaded_files:
        try:
            _uf0 = uploaded_files[0]
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as _ptmp:
                try: _uf0.seek(0)
                except Exception: pass
                shutil.copyfileobj(_uf0, _ptmp, length=8 * 1024 * 1024)
                _preview_path = _ptmp.name
            try: _uf0.seek(0)
            except Exception: pass
        except Exception:
            _preview_path = None

    if _preview_path:
        st.video(_preview_path)
    else:
        st.markdown(
            '<div style="background:#0a0a0a;border:1px solid #1c1c1c;border-radius:14px;'
            'height:220px;display:flex;align-items:center;justify-content:center;'
            'color:#333;font-size:0.82rem;letter-spacing:0.08em;text-transform:uppercase;">'
            'Upload a clip to preview</div>',
            unsafe_allow_html=True,
        )




# ---------------- ACTION ----------------
if run:

    kept_files = list(uploaded_files or [])
    has_uploads = bool(kept_files)
    has_fetched = bool(st.session_state.fetched_paths)

    if not (has_uploads or has_fetched):
        st.error("Please upload at least one video or fetch from URLs above.")
        st.stop()
    if not storyline or not storyline.strip():
        st.error("Please describe the story you want the final video to tell.")
        st.stop()
    if not OPENAI_API_KEY:
        st.error("Missing API key. Set `API_KEY` or `OPENAI_API_KEY` in your .env or Streamlit Secrets.")
        st.stop()

    def format_time_left(seconds):
        seconds = max(0, int(seconds))
        if seconds < 60:
            return f"{seconds}s left"
        else:
            mins = seconds // 60
            secs = seconds % 60
            return f"{mins}m {secs}s left"

    # ── Countdown ticker ────────────────────────────────────────────────────────
    # Runs in a daemon thread so the timer ticks every second even while Python
    # is blocked inside a long synchronous call (scoring, rendering, etc.).
    _ticker_stop_event: list = [None]  # index-0 holds the active threading.Event

    def _start_countdown(remaining_seconds: float):
        """Start (or restart) the background countdown ticker."""
        ev = _ticker_stop_event[0]
        if ev is not None:
            ev.set()  # stop any currently-running ticker
        ev = threading.Event()
        _ticker_stop_event[0] = ev
        deadline = time.time() + max(1.0, float(remaining_seconds))

        def _tick():
            while not ev.is_set():
                secs_left = max(0, deadline - time.time())
                try:
                    countdown_text.markdown(f"⏳ {format_time_left(secs_left)}")
                except Exception:
                    break
                if secs_left <= 0:
                    break
                ev.wait(1.0)

        t = threading.Thread(target=_tick, daemon=True)
        # Propagate Streamlit's script context so the thread can call st ops
        # without generating "missing ScriptRunContext" warnings.
        try:
            from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
            ctx = get_script_run_ctx()
            if ctx is not None:
                add_script_run_ctx(t, ctx)
        except Exception:
            pass
        t.start()

    def _stop_countdown():
        """Stop the ticker and clear the countdown display."""
        ev = _ticker_stop_event[0]
        if ev is not None:
            ev.set()
            _ticker_stop_event[0] = None
        try:
            countdown_text.markdown("")
        except Exception:
            pass
    # ────────────────────────────────────────────────────────────────────────────

    # Dynamic estimate tuned by input count (still an estimate, but adaptive)
    est_input_count = max(1, len(kept_files) + len(st.session_state.fetched_paths))
    est_total_time = (
        2
        + max(3, len(kept_files) * 1.5)
        + max(12, est_input_count * 6)
        + max(5, est_input_count * 1.2)
        + max(20, est_input_count * 4)
    )
    start_time = time.time()

    progress_bar = st.progress(0)
    countdown_text = st.empty()
    _status_ph = st.empty()

    # We keep two lists so we ONLY delete files we created from uploads (not fetched URLs).
    upload_temp_paths: list[str] = []
    fetched_paths: list[str] = list(st.session_state.fetched_paths)

    def print_mem_usage(msg):
        import psutil
        mem = psutil.virtual_memory()
        print(f"[MEM] {msg}: {mem.percent}% ({mem.used // (1024**2)}MB / {mem.total // (1024**2)}MB)")

    try:
        with st.spinner("Transcribing and editing your video..."):
            print_mem_usage("Start of processing")
            # Save uploads to tmp files
            for i, uf in enumerate(kept_files):
                _status_ph.caption(f"Saving clip {i + 1} of {len(kept_files)}…")
                elapsed = time.time() - start_time
                time_left = est_total_time - elapsed
                countdown_text.markdown(f"⏳ {format_time_left(time_left)}")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                    try:
                        uf.seek(0)
                    except Exception:
                        pass
                    shutil.copyfileobj(uf, tmp, length=8 * 1024 * 1024)  # 8 MB chunks
                    upload_temp_paths.append(tmp.name)
                progress_bar.progress(min(25, int((i + 1) / max(1, len(kept_files)) * 25)))
            print_mem_usage("After saving uploads")
            gc.collect()

            # Combine inputs
            input_paths = upload_temp_paths + fetched_paths

            # Filter any missing/tiny paths BEFORE transcription
            input_paths, bad_inputs = _filter_existing(input_paths)
            if bad_inputs:
                st.warning("Some inputs were missing or too small and were removed.")
                with st.expander("Show skipped inputs"):
                    st.write(bad_inputs[:50])

            if not input_paths:
                st.error("No usable video clips found after validation.")
                st.stop()

            print_mem_usage("Before transcription")
            # Transcribe (with splitting)
            # (transcription stage set by _render_stages)
            transcribe_start = time.time()

            def _on_transcribe_progress(done_chunks: int, total_chunks: int):
                if total_chunks <= 0:
                    return
                elapsed_transcribe = time.time() - transcribe_start
                if done_chunks < 3:
                    # Early samples are noisy because split/transcode overhead skews timing.
                    seeded_avg_per_chunk = 3.0
                    remaining_transcribe = max(0.0, (total_chunks - done_chunks) * seeded_avg_per_chunk)
                else:
                    avg_per_chunk = elapsed_transcribe / max(1, done_chunks)
                    avg_per_chunk = min(12.0, max(0.4, avg_per_chunk))
                    remaining_transcribe = max(0.0, (total_chunks - done_chunks) * avg_per_chunk)
                post_transcribe_est = max(18.0, est_input_count * 3.0)
                total_left = remaining_transcribe + post_transcribe_est
                total_left = min(total_left, 3600.0)
                countdown_text.markdown(f"⏳ {format_time_left(total_left)}")
                progress_bar.progress(min(60, 25 + int((done_chunks / total_chunks) * 35)))

            transcriptions = transcribe_videos_with_split(
                input_paths,
                openai_api_key=OPENAI_API_KEY,
                progress_callback=_on_transcribe_progress,
            )
            transcribe_elapsed = max(0.1, time.time() - transcribe_start)
            progress_bar.progress(45)
            print_mem_usage("After transcription")
            gc.collect()

            if not transcriptions:
                st.error("Transcription failed for all clips.")
                st.stop()

            # Score
            # (analyzing stage set by _render_stages)
            score_start = time.time()
            score_estimate = max(5.0, 0.20 * transcribe_elapsed + (0.8 * len(input_paths)))
            render_baseline_estimate = max(18.0, 0.85 * transcribe_elapsed + (2.0 * len(input_paths)))
            _start_countdown(score_estimate + render_baseline_estimate)  # ticks while scoring blocks
            scored = score_clips_with_story(
                transcriptions,
                storyline,
                priority_keywords=[kw.strip() for kw in user_priority_keywords.split(",") if kw.strip()],
                exclude_keywords=[kw.strip() for kw in user_excluded_keywords.split(",") if kw.strip()],
                tone=tone,
                target_duration_sec=target_duration_sec,
                openai_api_key=OPENAI_API_KEY,
            )
            score_elapsed = max(0.1, time.time() - score_start)
            _stop_countdown()  # stop scoring ticker before re-estimating render time
            print_mem_usage("After scoring")
            gc.collect()

            # Build clip_metadata dict (path → {role, shot_type, emotion, visual_score, trim})
            # scored is List[Dict] when visual analysis succeeded
            clip_metadata: dict = {}
            if scored and isinstance(scored[0], dict):
                for item in scored:
                    p = item.get("path", "")
                    if p:
                        clip_metadata[p] = {
                            "narrative_role":  item.get("narrative_role", "development"),
                            "shot_type":       item.get("shot_type", "unknown"),
                            "emotion":         item.get("emotion", "neutral"),
                            "visual_score":    item.get("visual_score", 0.5),
                            "description":     item.get("description", ""),
                            # Vision-guided trim recommendation from GPT-4o
                            "best_moment_sec": item.get("best_moment_sec"),
                            "trim_start_sec":  item.get("trim_start_sec"),
                            "trim_end_sec":    item.get("trim_end_sec"),
                        }

            # FORCE to list[str]
            relevant_paths = _normalize_to_paths(scored)

            # If scoring returns nothing usable, fallback to original order
            if not relevant_paths:
                st.warning("No relevant clips detected by scoring; using original order.")
                relevant_paths = [t.get("path") for t in transcriptions if t.get("path")]

            # Filter again (critical)
            relevant_paths, bad_ranked = _filter_existing(relevant_paths)
            if not relevant_paths:
                st.error("Scoring produced no usable file paths. Falling back to original order failed too.")
                with st.expander("Show ranked paths that failed validation"):
                    st.write(bad_ranked[:100])
                st.stop()

            usable_input_duration = sum(_probe_duration_seconds(p) for p in input_paths)
            selected_input_duration = sum(_probe_duration_seconds(p) for p in relevant_paths)

            with st.expander("🔍 Visual Analysis per Clip", expanded=False):
                if clip_metadata:
                    for p, m in clip_metadata.items():
                        trim_info = ""
                        if m.get("trim_start_sec") is not None and m.get("trim_end_sec") is not None:
                            trim_info = f" | cut `{m['trim_start_sec']:.1f}s`→`{m['trim_end_sec']:.1f}s`"
                        elif m.get("best_moment_sec") is not None:
                            trim_info = f" | peak @ `{m['best_moment_sec']:.1f}s`"
                        st.markdown(
                            f"**{os.path.basename(p)}**   "
                            f"`{m['narrative_role']}` | `{m['shot_type']}` | `{m['emotion']}` | "
                            f"visual score: `{m['visual_score']:.2f}`{trim_info}"
                        )
                        if m.get("description"):
                            st.caption(m["description"])
                else:
                    st.info("Visual analysis unavailable (no API key or analysis skipped).")

            with st.expander("🧪 Pre-Render Duration Debug", expanded=False):
                st.write({
                    "target_duration_sec": int(target_duration_sec),
                    "usable_input_clip_count": len(input_paths),
                    "selected_clip_count": len(relevant_paths),
                    "usable_input_total_sec": round(float(usable_input_duration), 2),
                    "selected_input_total_sec": round(float(selected_input_duration), 2),
                })

            progress_bar.progress(70)
            render_estimate = max(15.0, (0.75 * transcribe_elapsed) + (2.25 * score_elapsed) + (2.2 * len(relevant_paths)))
            _start_countdown(render_estimate)  # ticks every second while generate_video() blocks

            # Render final video
            # (rendering stage set by _render_stages)
            final_video_path = generate_video(
                relevant_paths,
                storyline,
                transition_duration=transition_duration,
                tone=tone,
                target_duration_sec=target_duration_sec,
                mix_original_audio=mix_original_audio,
                show_opening_card=show_opening_card,
                custom_music_path=user_music_path,
                clip_metadata=clip_metadata,   # <-- visual metadata for smart editing
            )
            progress_bar.progress(95)
            _stop_countdown()  # render is done — clear the ticker immediately

            final_output_duration = _probe_duration_seconds(final_video_path)
            with st.expander("🧪 Post-Render Duration Debug", expanded=False):
                st.write({
                    "target_duration_sec": int(target_duration_sec),
                    "final_output_sec": round(float(final_output_duration), 2),
                    "duration_delta_sec": round(float(final_output_duration - float(target_duration_sec)), 2),
                })

            print_mem_usage("Before video download button")
            # Avoid reading large video into memory for download
            filename = f"final_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            st.success("✅ Video created successfully!")
            st.video(final_video_path)
            with open(final_video_path, "rb") as f:
                video_bytes = f.read()
            st.download_button(
                "📥 Download MP4",
                data=video_bytes,
                file_name=filename,
                mime="video/mp4",
            )
            del video_bytes
            progress_bar.progress(100)
            _status_ph.empty()
            _stop_countdown()  # ensure cleared on success
            gc.collect()
            print_mem_usage("End of processing")

    except Exception as e:
        _stop_countdown()  # always clear on error
        st.error("Something went wrong while creating your video.")
        with st.expander("Show error details"):
            st.exception(e)

    finally:
        # Cleanup ONLY upload temps we created here
        for p in upload_temp_paths:
            try:
                os.remove(p)
            except Exception:
                pass
