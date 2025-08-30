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

# for server-side fetch
import requests, re

from editor import generate_video, transcribe_videos
from scoring import score_clips_with_story
from utils import create_temp_file  # optional helper


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

# Modern dark editor styling
st.markdown("""
<style>
:root { --bg:#13131a; --panel:#1b1b25; --panel-2:#20202b; --accent:#7c4dff; --text:#e9e7ff; --text-dim:#b7b4d6; }
.stApp { background: var(--bg); }
.block-container { padding-top: 1.2rem; max-width: 1120px; }
h1,h2,h3,h4 { color: var(--text); }
.side-card, .editor-card { background: var(--panel); border:1px solid #2a2a38; border-radius:14px; padding:16px; }
.preview { background: var(--panel-2); border-radius:12px; padding:12px; border:1px solid #2a2a38; }
.toolbar { display:flex; flex-direction:column; gap:10px; }
.toolbtn {
  width:44px; height:44px; border-radius:10px; background:#2a2a38;
  display:flex; align-items:center; justify-content:center; color:var(--text-dim);
  border:1px solid #343447;
}
.controls { display:flex; align-items:center; gap:10px; padding:10px 0; color:var(--text-dim); }
.ctrlbtn { width:40px; height:40px; border-radius:10px; background:#2a2a38; border:1px solid #343447;
  display:flex; align-items:center; justify-content:center; }
.scrubber { height:6px; border-radius:6px; background:#2a2a38; position:relative; }
.scrubber > div { position:absolute; top:0; left:0; height:100%; background:#494a72; border-radius:6px; }
.tl { background: var(--panel); border:1px solid #2a2a38; border-radius:12px; padding:12px; }
.tl-row { display:grid; grid-template-columns:120px 1fr; gap:12px; align-items:center; margin-top:10px; }
.tl-label { color:var(--text-dim); font-weight:600; font-size:0.9rem; }
.tl-track { position:relative; height:36px; background:#242438; border-radius:8px; overflow:hidden; }
.tl-clip  { position:absolute; top:6px; height:24px; border-radius:6px; }
.tl-clip.video { background:#49c3b1; opacity:.85; }
.tl-clip.audio { background:#8a63ff; opacity:.85; }
.stTextArea textarea, .stTextInput input {
  background:#20202b !important; color:var(--text) !important; border:1px solid #2a2a38 !important;
}
.generate-btn button { background:var(--accent) !important; color:white !important; border:none !important; }
</style>
""", unsafe_allow_html=True)

st.title("AI VIDEO EDITOR")


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("⚙️ Options")
    tone = st.selectbox(
        "Choose the tone",
        ["Cinematic", "Energetic", "Sentimental", "Epic", "Calm"],
        index=0,
    )
    mix_original_audio = st.toggle("Mix original audio with music (ducking)", value=False)
    show_opening_card = st.toggle("Show opening title card", value=True)

    transition_duration = st.slider("Transition duration (sec)", 0.3, 2.5, 1.0, 0.1)
    size_limit_mb = st.number_input(
        "Max per-file size (MB)", min_value=50, max_value=2000, value=500, step=50
    )
    st.caption("Tip: On first cloud run, models are downloaded by the API; allow a moment.")


# ---------------- STATE ----------------
if "fetched_paths" not in st.session_state:
    st.session_state.fetched_paths = []

def _too_big(file, limit_mb: int) -> bool:
    size = getattr(file, "size", None)
    return bool(size and size > limit_mb * 1024 * 1024)


# ===================== EDITOR LAYOUT =====================
left, right = st.columns([0.36, 0.64], gap="large")

# -------- LEFT: Text panel + inputs --------
with left:
    st.markdown("### TEXT TO VIDEO")
    storyline = st.text_area(
        " ",
        height=180,
        placeholder="A cat is sitting on a window sill. The rain is falling outside.",
        label_visibility="collapsed",
    )

    uploaded_files = st.file_uploader(
        "Add clips (MP4/MPEG4)",
        type=["mp4", "mpeg4"],
        accept_multiple_files=True,
    )

    # Warn about oversized local uploads (these will be skipped)
    if uploaded_files:
        too_big = [f.name for f in uploaded_files if _too_big(f, size_limit_mb)]
        if too_big:
            st.warning("These files exceed your size limit and will be skipped: " + ", ".join(too_big))

    # ======== SERVER-SIDE FETCH (2GB+) ========
    st.caption("Or paste direct MP4 URLs for large files (2GB+).")
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

    urls = st.text_area(
        "One or more URLs (comma or newline separated)",
        placeholder="https://example.com/video1.mp4\nhttps://example.com/video2.mp4",
    )
    colA, colB = st.columns([1,1])
    with colA:
        fetch_clicked = st.button("⬇️ Fetch from URLs")
    with colB:
        clear_fetched = st.button("🧹 Clear fetched")
    if clear_fetched:
        st.session_state.fetched_paths = []
        st.info("Cleared fetched files list.")

    if fetch_clicked and urls.strip():
        st.session_state.fetched_paths = []  # reset each fetch
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

    st.markdown('<div class="generate-btn">', unsafe_allow_html=True)
    run = st.button("GENERATE", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# -------- RIGHT: Preview + toolbar + timeline --------
with right:
    st.markdown("### PREVIEW")
    preview_path = None

    # Prefer the first fetched file for preview (it’s already on disk)
    if st.session_state.get("fetched_paths"):
        preview_path = st.session_state["fetched_paths"][0]

    # Otherwise preview the first uploaded file (save a temp copy but reset pointer)
    if not preview_path and uploaded_files:
        try:
            uf = uploaded_files[0]
            pos = uf.tell() if hasattr(uf, "tell") else None
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as _tmp:
                data = uf.read()
                _tmp.write(data)
                preview_path = _tmp.name
            # reset the file pointer so later saving still works
            try:
                uf.seek(0)
            except Exception:
                pass
        except Exception:
            preview_path = None

    st.markdown('<div class="preview">', unsafe_allow_html=True)
    if preview_path:
        st.video(preview_path)
    else:
        st.image(
            "https://picsum.photos/960/540?blur=2",
            caption="Preview appears here after you add a clip",
            use_column_width=True,
        )

    # visual-only transport controls + scrubber
    st.markdown("""
    <div class="controls">
        <div class="ctrlbtn">⏮</div>
        <div class="ctrlbtn">⏪</div>
        <div class="ctrlbtn">▶️</div>
        <div class="ctrlbtn">⏩</div>
        <div class="ctrlbtn">⏭</div>
        <div style="flex:1"></div>
        <div>🔉</div>
    </div>
    <div class="scrubber"><div style="width: 35%;"></div></div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    tcol1, tcol2 = st.columns([1,9])
    with tcol1:
        st.markdown("""
        <div class="toolbar">
            <div class="toolbtn" title="Cuts">✂️</div>
            <div class="toolbtn" title="Speed">⏱️</div>
            <div class="toolbtn" title="Levels">🎚️</div>
            <div class="toolbtn" title="Transitions">🔀</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### Timeline")
    st.markdown("""
    <div class="tl">
      <div class="tl-row">
        <div class="tl-label">VIDEO 1</div>
        <div class="tl-track">
          <div class="tl-clip video" style="left:2%; width:80%;"></div>
        </div>
      </div>
      <div class="tl-row">
        <div class="tl-label">AUDIO 1</div>
        <div class="tl-track">
          <div class="tl-clip audio" style="left:2%; width:60%;"></div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------- EXTRA INPUTS UNDER LAYOUT ----------------
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
    # Basic validation
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
        st.error("Missing API key. Set `API_KEY` or `OPENAI_API_KEY` in your .env or Streamlit Cloud Secrets.")
        st.stop()

    progress_text = st.empty()
    progress_bar = st.progress(0)
    temp_video_paths: list[str] = []

    try:
        with st.spinner("Transcribing and editing your video..."):
            # Save uploads to tmp files (skip oversized)
            for i, uf in enumerate(kept_files):
                progress_text.write(f"📥 Saving clip {i + 1} of {len(kept_files)}...")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                    # Ensure pointer is at start in case we previewed
                    try:
                        uf.seek(0)
                    except Exception:
                        pass
                    tmp.write(uf.read())
                    temp_video_paths.append(tmp.name)
                if kept_files:
                    progress_bar.progress(min(25, int((i + 1) / max(1, len(kept_files)) * 25)))

            # Add any server-fetched big files
            temp_video_paths.extend(st.session_state.fetched_paths)

            if not temp_video_paths:
                st.error("No usable video clips found. Please upload or fetch at least one clip.")
                st.stop()

            # Transcribe via OpenAI Whisper API
            progress_text.write("📝 Transcribing clips...")
            transcriptions = transcribe_videos(temp_video_paths, openai_api_key=OPENAI_API_KEY)
            progress_bar.progress(45)

            if not transcriptions:
                st.error("Transcription failed for all clips. Check your files and try again.")
                st.stop()

            # Score by story using OpenAI Embeddings
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

            # Render final video
            progress_text.write("🎞️ Generating final video with transitions and effects...")
            final_video_path = generate_video(
                relevant_clips,
                storyline,
                transition_duration=transition_duration,
                tone=tone,
                # mix_original_audio and show_opening_card exist in editor.py but
                # weren't in the original call; wire them if you decide to use:
                # mix_original_audio=mix_original_audio,
                # show_opening_card=show_opening_card,
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
