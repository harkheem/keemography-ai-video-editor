# KEEMOGRAPHY AI Video Editor

Turn a rough set of video clips into a story-driven short edit — powered by GPT-4o vision, Whisper transcription, and embedding-based clip ranking.

## Architecture

```
┌──────────────────────────────────┐
│  React + Vite + Tailwind (SPA)   │  3-step wizard UI
└────────────────┬─────────────────┘
                 │ REST + SSE
┌────────────────▼─────────────────┐
│  FastAPI backend (uvicorn)       │  /api/* endpoints
│  ├── backend/main.py             │  job queue, file management
│  ├── app_utils.py                │  chunking, transcription helpers
│  ├── scoring.py                  │  GPT-4o vision + embeddings
│  ├── editor.py                   │  MoviePy + FFmpeg rendering
│  └── transition.py               │  crossfade / xfade library
└──────────────────────────────────┘
```

The FastAPI server also statically serves the built React frontend, so a single container handles everything.

A legacy `app.py` Streamlit interface is still present but is not the production entrypoint.

## Features

- **Story-first editing** — rank and arrange clips by semantic similarity to your narrative prompt using OpenAI embeddings.
- **GPT-4o visual analysis** — each clip is scored for narrative role (hook / payoff / turn / development / b-roll), shot type, emotion, and a visual score; a trim recommendation is generated from sampled frames.
- **Whisper transcription** — clips are transcribed with automatic chunking for files over 24 MB.
- **Beat-aligned editing** — librosa analyzes the music track for beats and transients; cuts snap to the beat grid.
- **AI music trim** — GPT-4o suggests the best start/end window of an uploaded track.
- **URL ingest** — paste direct MP4, Google Drive share, or Dropbox links; the backend normalizes and downloads them.
- **Real-time progress** — SSE stream pushes stage/percentage updates to the frontend during the job.
- **Tone presets** — Cinematic / Energetic / Sentimental / Epic / Calm adjusts pacing, clip length, and default music selection.
- **Audio options** — keep original audio, replace with music, or mix (ducked background music).
- **Opening title card** — optional title frame at the start of the export.
- **Keyword filters** — boost clips matching priority keywords; drop clips matching exclusion keywords.
- **4K support** — xfade transitions with timebase normalization for high-resolution footage.
- **Docker / Railway ready** — single `Dockerfile` builds and serves the full stack.

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 18, Vite, Tailwind CSS |
| Backend | Python 3.11, FastAPI, uvicorn |
| AI | OpenAI API — `whisper-1`, `text-embedding-3-small`, `gpt-4o` |
| Video | MoviePy 2, FFmpeg, imageio-ffmpeg |
| Audio | librosa, scipy, soundfile |
| Deploy | Docker, Railway |

## Project Structure

```
ai_video_editor_mvp/
├── backend/
│   ├── main.py           # FastAPI app — routes, job queue, SSE
│   └── music/            # Default tone soundtrack assets
├── frontend/
│   ├── src/
│   │   ├── App.jsx       # 3-step wizard (Clips → Settings → Generate)
│   │   └── api.js        # Typed fetch helpers for all /api/* endpoints
│   ├── package.json
│   └── vite.config.js
├── app_utils.py          # Shared helpers: chunking, transcription, path utils
├── editor.py             # Timeline assembly and FFmpeg render pipeline
├── scoring.py            # GPT-4o vision + embedding-based clip scoring
├── transition.py         # Crossfade / xfade transition effects
├── app.py                # Legacy Streamlit UI (not used in production)
├── requirements.txt      # Python dependencies
├── Dockerfile            # Single-container build (Python + Node + ffmpeg)
├── .env                  # Local secrets (not committed)
└── runtime.txt           # Python version pin
```

## Requirements

- Python 3.11+
- Node.js 18+ (for local frontend development)
- FFmpeg installed and available in `PATH`
- OpenAI API key

```bash
# macOS
brew install ffmpeg node
```

## Local Development

### 1. Python environment

```bash
python3.11 -m venv .venv311
source .venv311/bin/activate
pip install -r requirements.txt
```

### 2. Environment variables

Create `.env` in the project root:

```env
OPENAI_API_KEY=sk-...
# Optional: GOOGLE_API_KEY is not required
```

### 3. Start the backend

```bash
uvicorn backend.main:app --reload --port 8000
```

### 4. Start the frontend (dev mode with HMR)

```bash
cd frontend
npm install
npm run dev        # serves at http://localhost:5173, proxies /api → :8000
```

Open `http://localhost:5173`.

## Production Build (Docker)

```bash
docker build -t keemography .
docker run -p 8000:8000 -e OPENAI_API_KEY=sk-... keemography
```

Open `http://localhost:8000`. The React build is served statically by FastAPI.

## Railway Deploy

1. Connect the repo to a Railway project.
2. Set `OPENAI_API_KEY` in the Railway environment variables panel.
3. Railway detects the `Dockerfile` and builds automatically. The `$PORT` variable is injected at runtime.

## How to Use

1. **Step 1 — Clips**: upload MP4 / MOV / AVI / MKV files or paste direct video URLs (Google Drive and Dropbox links are auto-converted).
2. **Step 2 — Settings**: write your storyline prompt, choose tone, target duration, transition length, and optionally upload a music track. Expand keyword filters to prioritize or exclude specific content.
3. **Step 3 — Generate**: review the summary and click **Generate Video**. A progress bar with stage pills (Upload → Transcribe → Analyze → Render → Done) updates in real time via SSE. When done, preview and download the MP4.

## Background Music Assets

Default per-tone soundtracks are loaded from `backend/music/`. The render still works without them — it just skips the background music layer.

Expected filenames: `cinematic.mp3`, `energetic.mp3`, `sentimental.mp3`, `epic.mp3`, `calm.mp3`.

## API Reference

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/upload` | Upload video clips (multipart) |
| `POST` | `/api/upload-music` | Upload and analyze a music track |
| `POST` | `/api/fetch-url` | Download a clip from a URL |
| `POST` | `/api/generate` | Start a render job; returns `job_id` |
| `GET` | `/api/status/{job_id}` | Poll job status and clip analysis |
| `GET` | `/api/events/{job_id}` | SSE stream of progress events |
| `GET` | `/api/download/{job_id}` | Download the finished MP4 |
| `GET` | `/api/health` | Health check |

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Missing API key` | Set `OPENAI_API_KEY` in `.env` or the Railway secrets panel. |
| `No usable clips` | Verify files are valid video, not zero-byte, and not corrupted. |
| URL fetch fails | The link must be directly downloadable and publicly accessible. |
| Slow on first run | Pip may bootstrap missing packages at startup; subsequent runs are faster. |
| 4K xfade artifacts | Check FFmpeg version ≥ 5.0; the pipeline normalizes timebase before concat. |

## Security Notes

- Never commit `.env` or API keys.
- Uploaded files are stored in a server-side temp directory and swept after job completion.
- URL downloads should only reference trusted, publicly accessible sources.
- Rotate your OpenAI key if it is ever exposed.
