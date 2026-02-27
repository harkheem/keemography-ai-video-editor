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
    page_title="KEEMOGRAPHY AI VIDEO EDITOR",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Enhanced Custom CSS: red/black cinematic theme ---
st.markdown("""
<style>
body, .stApp {
    background: radial-gradient(circle at top right, #2a0000 0%, #130000 35%, #050505 100%);
    color: #f6f6f6;
}

.block-container {
    padding-top: 1.5rem;
    max-width: 1200px;
}

h1, h2, h3, h4 {
    color: #ffffff;
    letter-spacing: 0.2px;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a0000 0%, #0a0a0a 100%);
    border-right: 1px solid #3a0f0f;
}

[data-testid="stSidebar"] * {
    color: #f4eeee;
}

.stButton>button, .stDownloadButton>button {
    background: linear-gradient(90deg, #8a0000 0%, #d41414 100%);
    color: #ffffff;
    border: 1px solid #ff4a4a;
    border-radius: 10px;
    font-weight: 700;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35);
    padding: 0.6em 1.2em;
}

.stButton>button:hover, .stDownloadButton>button:hover {
    background: linear-gradient(90deg, #b30000 0%, #ff2525 100%);
    border-color: #ff6a6a;
    color: #ffffff;
}

.stTextInput>div>input, .stTextArea textarea {
    background: #111111 !important;
    color: #ffffff !important;
    border-radius: 10px;
    border: 1px solid #4a1a1a !important;
}

.stTextInput>div>input:focus, .stTextArea textarea:focus {
    border-color: #c72424 !important;
    box-shadow: 0 0 0 1px #c72424 !important;
}

.stSelectbox div[data-baseweb="select"] > div,
.stMultiSelect div[data-baseweb="select"] > div {
    background: #111111;
    border: 1px solid #4a1a1a;
    border-radius: 10px;
}

.stProgress > div > div {
    background: linear-gradient(90deg, #8a0000, #ff2d2d);
}

.stAlert {
    border-radius: 10px;
    border: 1px solid #4a1a1a;
}

[data-testid="stFileUploaderDropzone"] {
    background: #0f0f0f;
    border: 1px dashed #7a1b1b;
    border-radius: 12px;
}

.uploadedFile {
    background: #121212;
    border: 1px solid #3d1717;
    border-radius: 10px;
    padding: 8px;
    margin-bottom: 6px;
}

.timeline {
    background: linear-gradient(180deg, #120808 0%, #0d0d0d 100%);
    border: 1px solid #3a1111;
    border-radius: 12px;
    padding: 16px;
    margin-top: 16px;
}

.timeline-bar {
    height: 18px;
    border-radius: 6px;
    margin-bottom: 8px;
}

.timeline-bar.video {
    background: linear-gradient(90deg, #7c0000, #d41414);
}

.timeline-bar.audio {
    background: linear-gradient(90deg, #370000, #8a0000);
}

[data-testid="stMetricValue"] {
    color: #ff5a5a;
}
</style>
""", unsafe_allow_html=True)

st.title("🎬 KEEMOGRAPHY AI VIDEO EDITOR")


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


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("⚙️ Project Settings")
    st.markdown("Set your video style and preferences.")
    tone = st.selectbox("🎨 Tone", ["Cinematic", "Energetic", "Sentimental", "Epic", "Calm"])
    target_duration_sec = st.slider("🎯 Target Video Length (sec)", 15, 180, 45, 5)
    transition_duration = st.slider("⏱️ Transition Duration (sec)", 0.15, 1.5, 0.3, 0.05)
    mix_original_audio = st.toggle("🎚️ Mix Original Audio", value=False)
    show_opening_card = st.toggle("🎬 Show Opening Card", value=True)
    st.caption("💡 Tip: Use short transitions for fast-paced edits.")
    # Memory usage display
    mem = psutil.virtual_memory()
    st.write(f"🧠 Memory usage: {mem.percent}% ({mem.used // (1024**2)}MB / {mem.total // (1024**2)}MB)")
    if mem.percent > 85:
        st.warning("System memory is critically high! Try fewer/smaller clips or close other apps.")


# ===================== EDITOR LAYOUT =====================
left, right = st.columns([0.38, 0.62], gap="large")

# -------- LEFT --------
with left:
    st.subheader("📝 Tell Your Story")
    storyline = st.text_area("Describe your video story", height=140, placeholder="A cat is sitting on a window sill. The rain is falling outside.")
    st.subheader("📁 Add Clips")
    uploaded_files = st.file_uploader("Upload MP4/MPEG4 files (no file count limit, 2GB each)", type=["mp4", "mpeg4"], accept_multiple_files=True)
    st.caption("Or paste direct video URLs (comma/newline separated):")
    urls = st.text_area("Paste URLs", placeholder="https://.../video.mp4")
    fetch_clicked = st.button("⬇️ Fetch from URLs")
    clear_fetched = st.button("🧹 Clear fetched")

    # --- New: User music upload ---
    st.subheader("🎵 Add Your Own Music (optional)")
    user_music_file = st.file_uploader("Upload MP3/WAV music", type=["mp3", "wav", "m4a", "aac", "ogg"], accept_multiple_files=False, key="music_upload")
    user_music_path = None
    if user_music_file:
        if _too_big(user_music_file, 100):
            st.warning("Music file is too large (max 100MB). Please upload a smaller file.")
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(user_music_file.name)[-1]) as tmp:
                user_music_file.seek(0)
                tmp.write(user_music_file.read())
                user_music_path = tmp.name
            st.success(f"Music uploaded: {user_music_file.name}")
            st.caption("Your music will be used for beat-aligned editing. If the file is invalid or too short, default music will be used.")
    else:
        st.caption("If you don't upload music, a default soundtrack will be used based on your selected tone.")

    if clear_fetched:
        st.session_state.fetched_paths = []
        st.info("Cleared fetched files list.")

    if fetch_clicked and urls.strip():
        st.session_state.fetched_paths = []
        url_list = [u.strip() for u in re.split(r"[,\n]+", urls) if u.strip()]
        for u in url_list:
            direct = _normalize_drive_dropbox(u)
            try:
                st.write(f"⬇️ Fetching {direct} ...")
                with requests.get(direct, stream=True, timeout=1200) as r:
                    r.raise_for_status()
                    total = int(r.headers.get("content-length", 0))
                    prog = st.progress(0.0)
                    downloaded = 0
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
                        for chunk in r.iter_content(chunk_size=1024 * 1024 * 16):  # 16MB
                            if chunk:
                                f.write(chunk)
                                downloaded += len(chunk)
                                if total:
                                    prog.progress(min(1.0, downloaded / total))
                        saved = f.name
                st.session_state.fetched_paths.append(saved)
                st.success(f"✅ Saved: {saved}")
            except Exception as e:
                st.error(f"Download failed for {u}: {e}")

    if st.session_state.fetched_paths:
        st.markdown("**Fetched files:**")
        for p in st.session_state.fetched_paths:
            st.code(p, language="text")

    st.markdown('<div style="margin-top:1em"></div>', unsafe_allow_html=True)
    run = st.button("🚀 Generate Video", type="primary", use_container_width=True)

# -------- RIGHT --------
with right:
    st.subheader("🔎 Preview & Timeline")
    preview_path = None
    fetched_paths_list = st.session_state.get("fetched_paths", [])
    if fetched_paths_list:
        preview_path = fetched_paths_list[0]
    elif uploaded_files:
        try:
            uf = uploaded_files[0]
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as _tmp:
                try:
                    uf.seek(0)
                except Exception:
                    pass
                shutil.copyfileobj(uf, _tmp, length=8 * 1024 * 1024)  # 8 MB chunks
                preview_path = _tmp.name
            try:
                uf.seek(0)
            except Exception:
                pass
        except Exception:
            preview_path = None

    if preview_path:
        st.video(preview_path)
    else:
        st.image(
            "https://picsum.photos/960/540?blur=2",
            caption="Preview appears here after you add a clip",
            use_container_width=True,
        )

    # Timeline visualization
    st.markdown('<div class="timeline">', unsafe_allow_html=True)
    st.markdown("**Timeline**")
    st.markdown('<div class="timeline-bar video" style="width:80%"></div>', unsafe_allow_html=True)
    st.markdown('<div class="timeline-bar audio" style="width:60%"></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ---------------- EXTRA INPUTS ----------------
user_priority_keywords = st.text_input(
    "Optional: Keywords to prioritize (comma-separated)",
    placeholder="e.g. dance, culture, joy",
)
user_excluded_keywords = st.text_input(
    "Optional: Keywords to exclude (comma-separated)",
    placeholder="e.g. blurry, quiet",
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

    progress_text = st.empty()
    progress_bar = st.progress(0)
    countdown_text = st.empty()

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
                progress_text.write(f"📥 Saving upload {i + 1} of {len(kept_files)}...")
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
            progress_text.write("📝 Transcribing clips...")
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
            progress_text.write("🧠 Scoring clips based on your story and preferences...")
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
            progress_text.write("🏞️ Generating final video...")
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
            progress_text.write("✅ Done!")
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
