# KEEMOGRAPHY AI Video Editor MVP

Streamlit app that turns a rough set of video clips into a story-driven short edit.

It uses OpenAI Whisper for transcription, embeddings for clip relevance ranking, and MoviePy for timeline rendering with transitions and optional background music.

## Features

- Story-first editing: rank clips by semantic similarity to your storyline prompt.
- Automatic transcription: transcribe uploaded/fetched clips with Whisper.
- URL ingest: fetch direct MP4 links (Dropbox and Google Drive link normalization included).
- Timeline rendering: stitch clips with adaptive transitions.
- Audio options: keep original audio, replace with music, or mix (ducked music).
- Download-ready export: generates final MP4 for preview and download in-app.

## Tech Stack

- Python
- Streamlit
- OpenAI API (`whisper-1`, `text-embedding-3-small`)
- MoviePy + FFmpeg
- NumPy, Pillow, Requests

## Project Structure

```text
ai_video_editor_mvp/
  app.py              # Streamlit UI and end-to-end flow
  editor.py           # Transcription + video assembly/render
  scoring.py          # Embedding-based clip scoring
  transition.py       # Transition library and effects
  utils.py            # Optional helpers (scene detect, overlays, audio)
  requirements.txt    # Python dependencies
  packages.txt        # System packages for deployment (ffmpeg, fonts)
  runtime.txt         # Target Python runtime for deployment
  .env                # Local secrets (not committed)
```

## Requirements

- Python 3.11+ (deployment target is `3.12` per `runtime.txt`)
- FFmpeg installed and available in `PATH`
- OpenAI API key

For local macOS setup, install FFmpeg if needed:

```bash
brew install ffmpeg
```

## Setup

From the project folder:

```bash
cd ai_video_editor_mvp
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Environment Variables

Create `.env` in `ai_video_editor_mvp/`:

```env
OPENAI_API_KEY=your_openai_api_key
# or
API_KEY=your_openai_api_key
```

Notes:

- At least one of `OPENAI_API_KEY` or `API_KEY` is required.
- Current code path does not require `GOOGLE_API_KEY`; URL fetching works with direct downloadable links.

## Run the App

```bash
cd ai_video_editor_mvp
streamlit run app.py
```

Then open the local Streamlit URL shown in terminal.

## How to Use

1. Enter a storyline in the `TELL YOUR STORY` panel.
2. Add clips by upload and/or fetch direct URLs.
3. Choose tone, transition duration, and audio options.
4. (Optional) Add priority/exclusion keywords.
5. Click `GENERATE` and download the resulting MP4.

## Background Music Assets (Optional but Recommended)

If you want tone-based music, add files at:

```text
assets/music/cinematic.mp3
assets/music/energetic.mp3
assets/music/sentimental.mp3
assets/music/epic.mp3
assets/music/calm.mp3
```

If missing, video export still works; it just skips background music.

## Troubleshooting

- `Missing API key`:
  - Confirm `.env` contains `OPENAI_API_KEY` or `API_KEY`.
- `No usable clips provided`:
  - Ensure files exist, are valid videos, and are not tiny/corrupt.
- Slow first run:
  - `app.py`/`editor.py` can bootstrap missing packages on startup.
- `TextClip`/font issues:
  - The app already falls back safely if text overlay dependencies are unavailable.
- URL fetch failures:
  - Verify link is directly downloadable and publicly accessible.

## Security Notes

- Never commit `.env` or secrets.
- Rotate API keys if they are exposed.
- Public URL downloads should only be from trusted sources.

## Deployment Notes

- `runtime.txt` and `packages.txt` are ready for PaaS workflows (for example, Streamlit Community Cloud-style builds).
- Ensure environment variables are configured in your deployment secrets panel.

