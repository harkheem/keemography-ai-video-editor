import os
import tempfile
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

from editor import generate_video, transcribe_videos
from scoring import score_clips_with_story
from utils import create_temp_file  # optional helper

# ---------- ENV ----------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ---------- STREAMLIT PAGE CONFIG ----------
st.set_page_config(
    page_title="KEEMOGRAPHY AI VIDEO EDITOR",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ---------- THEME / STYLES ----------
st.markdown(
    """
    <style>
        .stApp {
            background-image: linear-gradient(135deg, #1c003b, #2a004d);
            background-size: cover;
            color: white;
        }
        h1, h2, h3, h4 {
            color: #f4e8ff;
            text-align: center;
        }
        .stTextArea textarea, .stTextInput input, .stSelectbox [data-baseweb="select"] {
            background-color: #2a004d !important;
            color: white !important;
        }
        .block-container { max-width: 900px; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🎬 KEEMOGRAPHY AI VIDEO EDITOR")

# ---------- SIDEBAR OPTIONS ----------
with st.sidebar:
    st.header("⚙️ Options")
    tone = st.selectbox(
        "Choose the tone",
        ["Cinematic", "Energetic", "Sentimental", "Epic", "Calm"],
        index=0
    )
    mix_original_audio = st.toggle("Mix original audio with music (ducking)", value=False)
    show_opening_card = st.toggle("Show opening title card", value=True)
    transition_duration = st.slider("Transition duration (sec)", 0.3, 2.5, 1.0, 0.1)
    size_limit_mb = st.number_input("Max per-file size (MB)", min_value=50, max_value=1000, value=500, step=50)

    st.caption("Tip: Large models will download on first run; be patient on initial deploy.")

# ---------- MAIN FORM ----------
uploaded_files = st.file_uploader(
    "Upload multiple video clips (MP4/MPEG4)",
    type=["mp4", "mpeg4"],
    accept_multiple_files=True
)

storyline = st.text_area(
    "What story do you want the final video to tell?",
    placeholder="e.g. A Nigerian-American birthday party filled with culture, dance, and joy..."
)

user_priority_keywords = st.text_input(
    "Optional: Keywords to prioritize (comma-separated)",
    placeholder="e.g. dance, culture, joy"
)

user_excluded_keywords = st.text_input(
    "Optional: Keywords to exclude (comma-separated)",
    placeholder="e.g. blurry, quiet"
)

def _too_big(f, limit_mb: int) -> bool:
    size = getattr(f, "size", None)
    return bool(size and size > limit_mb * 1024 * 1024)

if uploaded_files:
    too_big = [f.name for f in uploaded_files if _too_big(f, size_limit_mb)]
    if too_big:
        st.warning(
            "These files exceed your size limit and will be skipped: " + ", ".join(too_big)
        )

# ---------- ACTION ----------
run = st.button("✨ Generate Video", type="primary")

if run:
    if not uploaded_files:
        st.error("Please upload at least one video.")
        st.stop()
    if not storyline.strip():
        st.error("Please describe the story you want the final video to tell.")
        st.stop()

    # Progress UI
    progress_text = st.empty()
    progress_bar = st.progress(0)

    temp_video_paths = []
    try:
        with st.spinner("Transcribing and editing your video... (first run may download models)"):
            # Save uploads to tmp files
            kept_files = [f for f in uploaded_files if not _too_big(f, size_limit_mb)]
            if not kept_files:
                st.error("All files were above the size limit. Increase the limit or upload smaller files.")
                st.stop()

            for i, uf in enumerate(kept_files):
                progress_text.write(f"📥 Saving clip {i + 1} of {len(kept_files)}...")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                    tmp.write(uf.read())
                    temp_video_paths.append(tmp.name)
                progress_bar.progress(min(25, int((i + 1) / max(1, len(kept_files)) * 25)))

            # Transcribe
            progress_text.write("📝 Transcribing clips...")
            transcriptions = transcribe_videos(temp_video_paths, openai_api_key=OPENAI_API_KEY)
            progress_bar.progress(45)

            if not transcriptions:
                st.error("Transcription failed for all clips. Check your files and try again.")
                st.stop()

            # Score & choose top clips
            progress_text.write("🧠 Scoring clips based on your story and preferences...")
            relevant_clips = score_clips_with_story(
                transcriptions,
                storyline,
                priority_keywords=[kw.strip() for kw in user_priority_keywords.split(",") if kw.strip()],
                exclude_keywords=[kw.strip() for kw in user_excluded_keywords.split(",") if kw.strip()]
            )
            if not relevant_clips:
                st.warning("No relevant clips detected from transcription; using original order.")
                relevant_clips = [t["path"] for t in transcriptions]
            progress_bar.progress(65)

            # Generate final video
            progress_text.write("🎞️ Generating final video with transitions and effects...")
            final_video_path = generate_video(
                relevant_clips,
                storyline,
                transition_duration=transition_duration,
                tone=tone,
                mix_original_audio=mix_original_audio,
                show_opening_card=show_opening_card,
            )
            progress_bar.progress(95)

            # Read for download button
            with open(final_video_path, "rb") as f:
                video_bytes = f.read()

            filename = f"final_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"

            st.success("✅ Video created successfully!")
            st.video(final_video_path)
            st.download_button(
                "📥 Download MP4",
                data=video_bytes,
                file_name=filename,
                mime="video/mp4"
            )
            progress_bar.progress(100)
            progress_text.write("✅ Done!")

    except Exception as e:
        st.error("Something went wrong while creating your video.")
        with st.expander("Show error details"):
            st.exception(e)

    finally:
        # Cleanup temp inputs (keep output for preview; it's removed on server restart)
        for p in temp_video_paths:
            try:
                os.remove(p)
            except Exception:
                pass
