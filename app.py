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
import tempfile
from datetime import datetime, timedelta
import time

import streamlit as st
from dotenv import load_dotenv
import requests, re
import psutil

# IMPORTANT: reload editor during dev so Streamlit doesn't keep an old version
import importlib
import editor as editor_mod
importlib.reload(editor_mod)
from editor import generate_video, transcribe_videos

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

# --- Enhanced Custom CSS for beautiful UI/UX ---
st.markdown("""
<style>
body, .stApp { background: #181825; color: #e9e7ff; }
.block-container { padding-top: 1.5rem; max-width: 1200px; }
h1, h2, h3, h4 { color: #fff; }
.stSidebar { background: #232336; }
.stButton>button, .stDownloadButton>button {
    background: linear-gradient(90deg, #7c4dff 0%, #49c3b1 100%);
    color: #fff; border: none; border-radius: 8px; font-weight: 600;
    box-shadow: 0 2px 8px #0002; padding: 0.6em 1.2em;
}
.stButton>button:hover, .stDownloadButton>button:hover {
    background: linear-gradient(90deg, #49c3b1 0%, #7c4dff 100%);
}
.stTextInput>div>input, .stTextArea textarea {
    background: #232336 !important; color: #fff !important; border-radius: 8px;
    border: 1px solid #343447 !important;
}
.stProgress>div>div { background: linear-gradient(90deg, #7c4dff, #49c3b1); }
.stAlert { border-radius: 8px; }
.uploadedFile { background: #232336; border-radius: 8px; padding: 8px; margin-bottom: 6px; }
.timeline { background: #232336; border-radius: 12px; padding: 16px; margin-top: 16px; }
.timeline-bar { height: 18px; border-radius: 6px; margin-bottom: 8px; }
.timeline-bar.video { background: #49c3b1; }
.timeline-bar.audio { background: #7c4dff; }
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


# ---------------- STATE ----------------
if "fetched_paths" not in st.session_state:
    st.session_state.fetched_paths = []


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("⚙️ Project Settings")
    st.markdown("Set your video style and preferences.")
    tone = st.selectbox("🎨 Tone", ["Cinematic", "Energetic", "Sentimental", "Epic", "Calm"])
    transition_duration = st.slider("⏱️ Transition Duration (sec)", 0.15, 1.5, 0.3, 0.05)
    mix_original_audio = st.toggle("🎚️ Mix Original Audio", value=False)
    show_opening_card = st.toggle("🎬 Show Opening Card", value=True)
    size_limit_mb = st.number_input("📦 Max File Size (MB)", 50, 2000, 200, 50)
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
    uploaded_files = st.file_uploader("Upload MP4/MPEG4 files (max 5, 200MB each)", type=["mp4", "mpeg4"], accept_multiple_files=True)
    if uploaded_files and len(uploaded_files) > 5:
        st.error("You can only upload up to 5 video files at once.")
        uploaded_files = uploaded_files[:5]
    for uf in uploaded_files or []:
        if hasattr(uf, "size") and uf.size > 200 * 1024 * 1024:
            st.warning(f"File {uf.name} is over 200MB and may cause memory issues.")
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
                _tmp.write(uf.read())
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
    kept_files = [f for f in (uploaded_files or []) if not _too_big(f, size_limit_mb)]
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
        if seconds < 60:
            return f"{int(seconds)}s left"
        else:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}m {secs}s left"

    # Estimate total steps and time (rough estimate)
    EST_TOTAL_STEPS = 5
    EST_STEP_TIMES = [3, 8, 10, 12, 20]  # seconds per step (tune as needed)
    est_total_time = sum(EST_STEP_TIMES)
    start_time = time.time()

    progress_text = st.empty()
    progress_bar = st.progress(0)
    countdown_text = st.empty()

    # We keep two lists so we ONLY delete files we created from uploads (not fetched URLs).
    upload_temp_paths: list[str] = []
    fetched_paths: list[str] = list(st.session_state.fetched_paths)

    try:
        with st.spinner("Transcribing and editing your video..."):
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
                    tmp.write(uf.read())
                    upload_temp_paths.append(tmp.name)

                progress_bar.progress(min(25, int((i + 1) / max(1, len(kept_files)) * 25)))

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

            # Transcribe
            progress_text.write("📝 Transcribing clips...")
            elapsed = time.time() - start_time
            time_left = est_total_time - elapsed
            countdown_text.markdown(f"⏳ {format_time_left(time_left)}")
            transcriptions = transcribe_videos(input_paths, openai_api_key=OPENAI_API_KEY)
            progress_bar.progress(45)

            if not transcriptions:
                st.error("Transcription failed for all clips.")
                st.stop()

            # Score
            progress_text.write("🧠 Scoring clips based on your story and preferences...")
            elapsed = time.time() - start_time
            time_left = est_total_time - elapsed
            countdown_text.markdown(f"⏳ {format_time_left(time_left)}")
            scored = score_clips_with_story(
                transcriptions,
                storyline,
                priority_keywords=[kw.strip() for kw in user_priority_keywords.split(",") if kw.strip()],
                exclude_keywords=[kw.strip() for kw in user_excluded_keywords.split(",") if kw.strip()],
                openai_api_key=OPENAI_API_KEY,
            )

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

            progress_bar.progress(65)
            elapsed = time.time() - start_time
            time_left = est_total_time - elapsed
            countdown_text.markdown(f"⏳ {format_time_left(time_left)}")

            # Render final video (wire the toggles)
            progress_text.write("🎞️ Generating final video...")
            final_video_path = generate_video(
                relevant_paths,
                storyline,
                transition_duration=transition_duration,
                tone=tone,
                mix_original_audio=mix_original_audio,
                show_opening_card=show_opening_card,
                custom_music_path=user_music_path,  # <-- pass user music
            )
            progress_bar.progress(95)
            elapsed = time.time() - start_time
            time_left = est_total_time - elapsed
            countdown_text.markdown(f"⏳ {format_time_left(time_left)}")

            with open(final_video_path, "rb") as f:
                video_bytes = f.read()

            filename = f"final_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"

            st.success("✅ Video created successfully!")
            st.video(final_video_path)
            st.download_button(
                "📥 Download MP4",
                data=video_bytes,
                file_name=filename,
                mime="video/mp4",
            )
            progress_bar.progress(100)
            progress_text.write("✅ Done!")
            countdown_text.markdown("")

    except Exception as e:
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
