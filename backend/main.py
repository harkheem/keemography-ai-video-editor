# backend/main.py
# FastAPI backend for Keemography AI Video Editor
# Run with: uvicorn backend.main:app --reload --port 8000
# (from the project root, with project root in PYTHONPATH)

import gc
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# ── Add project root to path so editor / scoring / app_utils are importable ──
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from app_utils import (
    filter_existing,
    normalize_drive_dropbox,
    normalize_to_paths,
    probe_duration_seconds,
    transcribe_videos_with_split,
)
from editor import generate_video
from scoring import score_clips_with_story

# ── Load env ─────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_ROOT, ".env"))
except ImportError:
    pass

# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Keemography API", version="2.0.0")

# In production (Railway) the frontend is served at the same origin as the API,
# so CORS doesn't apply.  ALLOW_ALL_ORIGINS=true is a safety valve for custom
# domains or reverse-proxy setups.
_cors_origins: list[str] | str
if os.getenv("ALLOW_ALL_ORIGINS", "").lower() in ("1", "true", "yes"):
    _cors_origins = ["*"]
else:
    _extra = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
    _cors_origins = [
        "http://localhost:5173",   # Vite dev
        "http://localhost:5174",   # Vite dev (alt port)
        "http://localhost:3000",   # CRA / preview
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        *_extra,
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory job store ───────────────────────────────────────────────────────
# { job_id: { status, progress, message, result_path, filename, error, events[] } }
_jobs: Dict[str, Dict[str, Any]] = {}
_jobs_lock = threading.Lock()


def _get_env_key() -> Optional[str]:
    return os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")


def _emit(job_id: str, progress: int, message: str) -> None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job:
            job["progress"] = progress
            job["message"] = message
            job["events"].append(json.dumps({"progress": progress, "message": message}))


# =============================================================================
#  UPLOAD ENDPOINTS
# =============================================================================

# Max size per multipart field (for small direct uploads, e.g. music)
_MAX_UPLOAD_BYTES = 20 * 1024 * 1024 * 1024

# In-flight chunked uploads: upload_id -> {path, fh, total, received, filename, size}
_chunk_uploads: Dict[str, Any] = {}
_chunk_lock = threading.Lock()


@app.post("/api/upload/chunk")
async def upload_chunk(request: Request):
    """
    Receive one chunk of a multi-part file upload.
    Each chunk is at most ~55 MB so it completes well within proxy timeouts.
    Fields:  upload_id, chunk_index (int), total_chunks (int), filename, chunk (file)
    """
    from starlette.requests import ClientDisconnect
    try:
        form = await request.form(
            max_files=1,
            max_fields=10,
            max_part_size=60 * 1024 * 1024,  # 60 MB ceiling per chunk
        )
    except ClientDisconnect:
        raise HTTPException(408, "Connection dropped during chunk upload — please retry.")

    upload_id = str(form.get("upload_id") or "")
    chunk_index = int(form.get("chunk_index", 0))
    total_chunks = int(form.get("total_chunks", 1))
    filename = str(form.get("filename") or "clip.mp4")
    chunk_field = form.get("chunk")

    if not upload_id or chunk_field is None or not hasattr(chunk_field, "read"):
        raise HTTPException(400, "Missing upload_id or chunk field")

    chunk_data = await chunk_field.read()

    with _chunk_lock:
        if upload_id not in _chunk_uploads:
            suffix = os.path.splitext(filename)[-1] or ".mp4"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            _chunk_uploads[upload_id] = {
                "path": tmp.name,
                "fh": tmp,
                "total": total_chunks,
                "received": set(),
                "filename": filename,
                "size": 0,
            }
        entry = _chunk_uploads[upload_id]
        entry["fh"].write(chunk_data)
        entry["received"].add(chunk_index)
        entry["size"] += len(chunk_data)

    return {"upload_id": upload_id, "chunk": chunk_index}


@app.post("/api/upload/finalize")
async def finalize_upload(body: dict):
    """Close and register a completed chunked upload; return file info."""
    upload_id = str(body.get("upload_id") or "")
    with _chunk_lock:
        entry = _chunk_uploads.pop(upload_id, None)
    if not entry:
        raise HTTPException(404, "Upload ID not found or already finalised")
    try:
        entry["fh"].close()
    except Exception:
        pass
    if len(entry["received"]) != entry["total"]:
        raise HTTPException(400, f"Incomplete upload: got {len(entry['received'])}/{entry['total']} chunks")
    return {"name": entry["filename"], "path": entry["path"], "size": entry["size"]}


@app.post("/api/upload/clips")
async def upload_clips(request: Request):
    """Legacy single-shot upload (small files / music).  Large files use /upload/chunk."""
    from starlette.requests import ClientDisconnect
    try:
        form = await request.form(
            max_files=500,
            max_fields=500,
            max_part_size=_MAX_UPLOAD_BYTES,
        )
    except ClientDisconnect:
        raise HTTPException(408, "Connection dropped during upload — please use chunked upload for large files.")
    saved = []
    for field_name, f in form.multi_items():
        if not hasattr(f, "filename"):  # skip plain text fields
            continue
        suffix = os.path.splitext(f.filename or "clip")[-1] or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            size = 0
            chunk_size = 8 * 1024 * 1024  # 8 MB read buffer
            while True:
                chunk = await f.read(chunk_size)
                if not chunk:
                    break
                tmp.write(chunk)
                size += len(chunk)
            saved.append({"name": f.filename, "path": tmp.name, "size": size})
    if not saved:
        raise HTTPException(400, "No video files received")
    return {"clips": saved}


@app.post("/api/upload/music")
async def upload_music(request: Request):
    """Save an uploaded music file to a temp path; return the path."""
    form = await request.form(
        max_files=1,
        max_fields=5,
        max_part_size=_MAX_UPLOAD_BYTES,
    )
    f = form.get("file")
    if f is None or not hasattr(f, "filename"):
        raise HTTPException(400, "No music file received")
    suffix = os.path.splitext(f.filename or "music")[-1] or ".mp3"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        size = 0
        chunk_size = 4 * 1024 * 1024  # 4 MB chunks
        while True:
            chunk = await f.read(chunk_size)
            if not chunk:
                break
            tmp.write(chunk)
            size += len(chunk)
        path = tmp.name
    return {"name": f.filename, "path": path, "size": size}


@app.post("/api/fetch-url")
async def fetch_url(body: dict):
    """Download a video from a URL (Drive / Dropbox / direct) and return the temp path."""
    url = (body.get("url") or "").strip()
    if not url:
        raise HTTPException(400, "No URL provided")
    direct = normalize_drive_dropbox(url)
    try:
        with requests.get(direct, stream=True, timeout=1200) as r:
            r.raise_for_status()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                for chunk in r.iter_content(chunk_size=16 * 1024 * 1024):
                    if chunk:
                        tmp.write(chunk)
                path = tmp.name
        return {"name": os.path.basename(url), "path": path, "size": os.path.getsize(path)}
    except Exception as e:
        raise HTTPException(500, f"Download failed: {e}")


# =============================================================================
#  GENERATION JOB
# =============================================================================

def _run_job(job_id: str, params: dict) -> None:
    """Background thread: transcribe → score → render."""
    with _jobs_lock:
        _jobs[job_id]["status"] = "running"

    clip_paths: List[str] = params["clip_paths"]
    storyline: str = params["storyline"]
    tone: str = params.get("tone", "Cinematic")
    target_duration_sec: int = int(params.get("target_duration_sec", 45))
    transition_duration: float = float(params.get("transition_duration", 0.3))
    mix_original_audio: bool = bool(params.get("mix_original_audio", False))
    show_opening_card: bool = bool(params.get("show_opening_card", True))
    music_path: Optional[str] = params.get("music_path") or None
    priority_keywords: List[str] = params.get("priority_keywords", [])
    exclude_keywords: List[str] = params.get("exclude_keywords", [])
    api_key: Optional[str] = params.get("api_key") or _get_env_key()

    try:
        # ── Validate ──────────────────────────────────────────────────────
        _emit(job_id, 5, "Validating clips…")
        good, bad = filter_existing(clip_paths)
        if bad:
            print(f"[JOB {job_id}] Skipping {len(bad)} invalid path(s): {bad[:5]}")
        if not good:
            raise ValueError("No usable clips found after validation.")

        # ── Transcribe ────────────────────────────────────────────────────
        _emit(job_id, 12, f"Transcribing {len(good)} clip(s) with Whisper…")

        def _on_progress(done: int, total: int):
            pct = 12 + int((done / max(1, total)) * 33)
            _emit(job_id, pct, f"Transcribing… ({done}/{total} chunks)")

        transcriptions = transcribe_videos_with_split(
            good,
            openai_api_key=api_key,
            progress_callback=_on_progress,
        )
        if not transcriptions:
            raise ValueError("Transcription failed for all clips.")

        # ── Score / visual analysis ───────────────────────────────────────
        n_clips = len(transcriptions)
        _emit(job_id, 47, f"Analyzing {n_clips} clip(s) with AI vision…")

        def _vision_progress(done: int, total: int, clip_name: str) -> None:
            pct = 47 + int(19 * done / max(1, total))  # 47% → 66%
            _emit(job_id, pct, f"Analyzing clip {done}/{total}: {clip_name}")

        scored = score_clips_with_story(
            transcriptions,
            storyline,
            priority_keywords=priority_keywords,
            exclude_keywords=exclude_keywords,
            tone=tone,
            target_duration_sec=target_duration_sec,
            openai_api_key=api_key,
            progress_callback=_vision_progress,
        )
        gc.collect()

        clip_metadata: dict = {}
        relevant_paths: List[str] = []
        if scored and isinstance(scored[0], dict):
            for item in scored:
                p = item.get("path", "")
                if p:
                    relevant_paths.append(p)
                    clip_metadata[p] = {
                        "narrative_role":  item.get("narrative_role", "development"),
                        "shot_type":       item.get("shot_type", "unknown"),
                        "emotion":         item.get("emotion", "neutral"),
                        "visual_score":    item.get("visual_score", 0.5),
                        "description":     item.get("description", ""),
                        "best_moment_sec": item.get("best_moment_sec"),
                        "trim_start_sec":  item.get("trim_start_sec"),
                        "trim_end_sec":    item.get("trim_end_sec"),
                    }

        if not relevant_paths:
            relevant_paths = [t["path"] for t in transcriptions if t.get("path") and os.path.exists(t["path"])]

        relevant_paths, _ = filter_existing(relevant_paths)
        if not relevant_paths:
            raise ValueError("Scoring produced no valid clip paths.")

        # ── Render ────────────────────────────────────────────────────────
        _emit(job_id, 68, "Rendering video with MoviePy…")
        final_path = generate_video(
            relevant_paths,
            storyline,
            transition_duration=transition_duration,
            tone=tone,
            target_duration_sec=target_duration_sec,
            mix_original_audio=mix_original_audio,
            show_opening_card=show_opening_card,
            custom_music_path=music_path,
            clip_metadata=clip_metadata,
        )
        gc.collect()

        # ── Done ──────────────────────────────────────────────────────────
        _emit(job_id, 100, "Done!")
        filename = f"keemography_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        with _jobs_lock:
            _jobs[job_id]["status"] = "done"
            _jobs[job_id]["result_path"] = final_path
            _jobs[job_id]["filename"] = filename
            # Emit visual metadata for the frontend analysis panel
            _jobs[job_id]["clip_analysis"] = [
                {
                    "name": os.path.basename(p),
                    **clip_metadata.get(p, {}),
                }
                for p in relevant_paths
            ]

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[JOB {job_id}] ERROR:\n{tb}")
        with _jobs_lock:
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["error"] = str(e)
            _jobs[job_id]["events"].append(
                json.dumps({"progress": -1, "message": f"Error: {e}"})
            )


@app.post("/api/generate")
async def start_generate(
    clip_paths: str = Form(...),               # JSON array of strings
    storyline: str = Form(...),
    tone: str = Form("Cinematic"),
    target_duration_sec: int = Form(45),
    transition_duration: float = Form(0.3),
    mix_original_audio: bool = Form(False),
    show_opening_card: bool = Form(True),
    music_path: Optional[str] = Form(None),
    priority_keywords: str = Form(""),
    exclude_keywords: str = Form(""),
    api_key: Optional[str] = Form(None),
):
    """Start a generation job; return job_id immediately."""
    try:
        paths: List[str] = json.loads(clip_paths)
    except json.JSONDecodeError:
        raise HTTPException(400, "clip_paths must be a valid JSON array")

    if not paths:
        raise HTTPException(400, "At least one clip path is required")
    if not storyline.strip():
        raise HTTPException(400, "storyline is required")

    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = {
            "status": "pending",
            "progress": 0,
            "message": "Queued…",
            "result_path": None,
            "filename": None,
            "error": None,
            "events": [],
            "clip_analysis": [],
        }

    params = {
        "clip_paths": paths,
        "storyline": storyline,
        "tone": tone,
        "target_duration_sec": target_duration_sec,
        "transition_duration": transition_duration,
        "mix_original_audio": mix_original_audio,
        "show_opening_card": show_opening_card,
        "music_path": music_path,
        "priority_keywords": [k.strip() for k in priority_keywords.split(",") if k.strip()],
        "exclude_keywords": [k.strip() for k in exclude_keywords.split(",") if k.strip()],
        "api_key": api_key,
    }

    t = threading.Thread(target=_run_job, args=(job_id, params), daemon=True)
    t.start()
    return {"job_id": job_id}


# =============================================================================
#  STATUS / EVENTS / DOWNLOAD
# =============================================================================

@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return {
        "status":       job["status"],
        "progress":     job["progress"],
        "message":      job["message"],
        "error":        job.get("error"),
        "has_result":   bool(job.get("result_path")),
        "clip_analysis": job.get("clip_analysis", []),
    }


@app.get("/api/events/{job_id}")
async def job_events(job_id: str):
    """Server-Sent Events stream for live progress updates."""

    async def _generate():
        import asyncio
        last_idx = 0
        last_ping = 0.0
        try:
            while True:
                with _jobs_lock:
                    job = _jobs.get(job_id)
                if not job:
                    yield f"data: {json.dumps({'error': 'not found'})}\n\n"
                    break

                events = job["events"]
                if len(events) > last_idx:
                    for ev in events[last_idx:]:
                        yield f"data: {ev}\n\n"
                    last_idx = len(events)
                    last_ping = asyncio.get_event_loop().time()

                if job["status"] in ("done", "error"):
                    yield f"data: {json.dumps({'done': True, 'status': job['status'], 'error': job.get('error')})}\n\n"
                    break

                # Send a keepalive comment every 15 s so proxies don't drop the connection
                now = asyncio.get_event_loop().time()
                if now - last_ping >= 15:
                    yield ": keepalive\n\n"
                    last_ping = now

                await asyncio.sleep(0.4)
        except (BrokenPipeError, GeneratorExit, ConnectionResetError):
            pass

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/download/{job_id}")
async def download_result(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job["status"] != "done":
        raise HTTPException(400, f"Job not complete (status: {job['status']})")
    result_path = job.get("result_path")
    if not result_path or not os.path.exists(result_path):
        raise HTTPException(404, "Output file not found on disk")
    return FileResponse(
        result_path,
        media_type="video/mp4",
        filename=job.get("filename", "output.mp4"),
    )


@app.delete("/api/job/{job_id}")
async def delete_job(job_id: str):
    """Clean up a job and its output file."""
    with _jobs_lock:
        job = _jobs.pop(job_id, None)
    if not job:
        raise HTTPException(404, "Job not found")
    result_path = job.get("result_path")
    if result_path and os.path.exists(result_path):
        try:
            os.remove(result_path)
        except Exception:
            pass
    return {"deleted": job_id}


# =============================================================================
#  HEALTH
# =============================================================================

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "api_key_set": bool(_get_env_key()),
        "python": sys.version,
    }


# =============================================================================
#  Serve React build (production)
#  Catch-all GET route instead of app.mount("/", StaticFiles(...)) so that
#  POST/PUT/DELETE requests always reach their API handlers first.
#  app.mount("/", StaticFiles(...)) intercepts ALL methods (returns 405 for
#  non-GET), which breaks POST endpoints like /api/upload/chunk.
# =============================================================================
_FRONTEND_DIST = os.path.join(_ROOT, "frontend", "dist")

if os.path.isdir(_FRONTEND_DIST):
    # Serve static assets (js/css/img) for exact file matches
    app.mount("/assets", StaticFiles(directory=os.path.join(_FRONTEND_DIST, "assets")), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        """Serve the React SPA for all non-API GET requests."""
        # Never serve the SPA for /api/* — let those 404 naturally
        if full_path.startswith("api/") or full_path.startswith("api"):
            raise HTTPException(404, "Not found")
        candidate = os.path.join(_FRONTEND_DIST, full_path)
        if full_path and os.path.isfile(candidate):
            return FileResponse(candidate)
        # SPA fallback — serve index.html for all client-side routes
        return FileResponse(os.path.join(_FRONTEND_DIST, "index.html"))
