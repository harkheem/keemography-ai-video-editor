# app.py

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

# now your normal imports:
import os
import tempfile
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

# NEW: for server-side fetch
import requests, re

from editor import generate_video, transcribe_videos
from scoring import score_clips_with_story
from utils import create_temp_file  # optional helper; safe to keep even if unused


# ---------------- ENV / SECRETS ----------------
load_dotenv()

def _get_key():
    # Works both locally (.env) and on Streamlit Cloud (Secrets)
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
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .stApp {
            background-image: linear-gradient(135deg, #1c003b, #2a004d);
            background-size: cover;
            color: white;
        }
        h1, h2, h3, h4 { color: #f4e8ff; text-align: center; }
        .stTextArea textarea, .stTextInput input {
            background-color: #2a004d !important;
            color: white !important;
        }
        .block-container { max-width: 900px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎬 KEEMOGRAPHY AI VIDEO EDITOR")


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("⚙️ Options")
    tone = st.selectbox(
        "Choose the tone",
        ["Cinematic", "Energetic", "Sentimental", "Epic", "Calm"],
        index=0,
    )
    # These toggles are cosmetic for now (editor.py doesn’t use them).
    mix_original_audio = st.toggle("Mix original audio with music (ducking)", value=False)
    show_opening_card = st.toggle("Show opening title card", value=True)

    transition_duration = st.slider("Transition duration (sec)", 0.3, 2.5, 1.0, 0.1)
    size_limit_mb = st.number_input(
        "Max per-file size (MB)", min_value=50, max_value=1000, value=500, step=50
    )
    st.caption("Tip: On first cloud run, models are downloaded by the API; allow a moment.")


# ---------------- MAIN INPUTS ----------------
uploaded_files = st.file_uploader(
    "Upload multiple video clips (MP4/MPEG4)",
    type=["mp4", "mpeg4"],
    accept_multiple_files=True,
)

# ========== SERVER-SIDE FETCH (2GB+) ==========
if "fetched_paths" not in st.session_state:
    st.session_state.fetched_paths = []

def _normalize_drive_dropbox(url: str) -> str:
    """Convert common share links to direct-download links when possible."""
    u = url.strip()
    # Google Drive: https://drive.google.com/file/d/<ID>/view -> https://drive.google.com/uc?export=download&id=<ID>
    m = re.search(r"https?://drive\.google\.com/file/d/([a-zA-Z0-9_-]+)", u)
    if m:
        return f"https://drive.google.com/uc?export=download&id={m.group(1)}"
    # Dropbox share -> direct dl
    if "dropbox.com" in u:
        if "?dl=0" in u:
            return u.replace("?dl=0", "?dl=1")
        if "?dl=1" not in u:
            return u + "?dl=1"
    return u

st.markdown("**Or paste direct MP4 URLs for large files (2GB+):**")
urls = st.text_area(
    "One or more URLs (comma or newline separated)",
    placeholder="https://example.com/video1.mp4\nhttps://example.com/video2.mp4",
)

cols_fetch = st.columns([1,1,2])
with cols_fetch[0]:
    fetch_clicked = st.button("⬇️ Fetch from URLs")
with cols_fetch[1]:
    clear_fetched = st.button("🧹 Clear fetched")

if clear_fetched:
    st.session_state.fetched_paths = []
    st.info("Cleared fetched files list.")

if fetch_clicked and urls.strip():
    st.session_state.fetched_paths = []  # reset on every fetch
    url_list = [u.strip() for u in re.split(r"[,\n]+", urls) if u.strip()]
    for u in url_list:
        direct = _normalize_drive_dropbox(u)
        try:
            st.write(f"⬇️ Fetching {direct} ...")
            with requests.get(direct, stream=True, timeout=1200) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                size_mb = total / (1024 * 1024) if total else None
                if size_mb:
                    st.info(f"Downloading ~{size_mb:.1f} MB")
                prog = st.progress(0.0)
                downloaded = 0
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024 * 50):  # 50MB
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total:
                                prog.progress(min(1.0, downloaded / total))
                    saved = f.name
            st.session_state.fetched_paths.append(saved)
            st.success(f"✅ Saved to {saved}")
        except Exception as e:
            st.error(f"Download failed for {u}: {e}")

if st.session_state.fetched_paths:
    st.caption("Fetched files ready:")
    for p in st.session_state.fetched_paths:
        st.code(p)

# ==============================================

storyline = st.text_area(
    "What story do you want the final video to tell?",
    placeholder="e.g. A Nigerian-American birthday party filled with culture, dance, and joy..."
)

user_priority_keywords = st.text_input(
    "Optional: Keywords to prioritize (comma-separated)",
    placeholder="e.g. dance, culture, joy",
)

user_excluded_keywords = st.text_input(
    "Optional: Keywords to exclude (comma-separated)",
    placeholder="e.g. blurry, quiet",
)


def _too_big(file, limit_mb: int) -> bool:
    size = getattr(file, "size", None)
    return bool(size and size > limit_mb * 1024 * 1024)


if uploaded_files:
    too_big = [f.name for f in uploaded_files if _too_big(f, size_limit_mb)]
    if too_big:
        st.warning("These files exceed your size limit and will be skipped: " + ", ".join(too_big))


# ---------------- ACTION ----------------
run = st.button("✨ Generate Video", type="primary")

if run:
    # Basic validation
    has_uploads = bool(uploaded_files) and any(not _too_big(f, size_limit_mb) for f in uploaded_files)
    has_fetched = bool(st.session_state.fetched_paths)

    if not (has_uploads or has_fetched):
        st.error("Please upload at least one video or fetch from URLs above.")
        st.stop()
    if not storyline.strip():
        st.error("Please describe the story you want the final video to tell.")
        st.stop()
    if not OPENAI_API_KEY:
        st.error(
            "Missing API key. Set `API_KEY` or `OPENAI_API_KEY` in your .env (local) "
            "or in Streamlit Cloud Secrets."
        )
        st.stop()

    progress_text = st.empty()
    progress_bar = st.progress(0)
    temp_video_paths = []

    try:
        with st.spinner("Transcribing and editing your video..."):
            # Save uploads to tmp files (skip oversized)
            kept_files = [f for f in (uploaded_files or []) if not _too_big(f, size_limit_mb)]
            for i, uf in enumerate(kept_files):
                progress_text.write(f"📥 Saving clip {i + 1} of {len(kept_files)}...")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                    tmp.write(uf.read())
                    temp_video_paths.append(tmp.name)
                # Up to 25% during save
                if kept_files:
                    progress_bar.progress(min(25, int((i + 1) / max(1, len(kept_files)) * 25)))

            # Add any server-fetched big files
            temp_video_paths.extend(st.session_state.fetched_paths)

            if not temp_video_paths:
                st.error("No usable video clips found. Please upload or fetch at least one clip.")
                st.stop()

            # Transcribe via OpenAI Whisper API (inside editor.transcribe_videos)
            progress_text.write("📝 Transcribing clips...")
            transcriptions = transcribe_videos(temp_video_paths, openai_api_key=OPENAI_API_KEY)
            progress_bar.progress(45)

            if not transcriptions:
                st.error("Transcription failed for all clips. Check your files and try again.")
                st.stop()

            # Score by story using OpenAI Embeddings (scoring.score_clips_with_story)
            progress_text.write("🧠 Scoring clips based on your story and preferences...")
            relevant_clips = score_clips_with_story(
                transcriptions,
                storyline,
                priority_keywords=[kw.strip() for kw in user_priority_keywords.split(",") if kw.strip()],
                exclude_keywords=[kw.strip() for kw in user_excluded_keywords.split(",") if kw.strip()],
                api_key=OPENAI_API_KEY,
            )
            if not relevant_clips:
                st.warning("No relevant clips detected from transcription; using original order.")
                relevant_clips = [t["path"] for t in transcriptions]
            progress_bar.progress(65)

            # Render final video (generate_video from editor.py)
            progress_text.write("🎞️ Generating final video with transitions and effects...")
            final_video_path = generate_video(
                relevant_clips,
                storyline,
                transition_duration=transition_duration,
                tone=tone,
                # NOTE: not passing mix_original_audio / show_opening_card here because
                # the provided editor.py signature doesn’t include them.
            )
            progress_bar.progress(95)

            # Prepare a download button
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

    except Exception as e:
        st.error("Something went wrong while creating your video.")
        with st.expander("Show error details"):
            st.exception(e)

    finally:
        # Cleanup temp inputs (keep output; Streamlit will clean on restart)
        for p in temp_video_paths:
            try:
                os.remove(p)
            except Exception:
                pass
