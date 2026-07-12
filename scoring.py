
# scoring.py
from typing import List, Dict, Optional, Sequence, Tuple
import os
import re
import json
import math
import numpy as np
import concurrent.futures

def _get_api_key(explicit_key: Optional[str] = None) -> Optional[str]:
    if explicit_key:
        return explicit_key
    return os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def _embed_texts(texts: Sequence[str], api_key: Optional[str]) -> np.ndarray:
    from openai import OpenAI  # lightweight
    client = OpenAI(api_key=api_key)
    clean_texts = []
    for text in texts:
        if text is None:
            clean_texts.append(" ")
            continue
        cleaned = str(text).strip()
        clean_texts.append(cleaned if cleaned else " ")
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=clean_texts,
    )
    # shape: (n, 1536)
    vecs = [np.array(d.embedding, dtype=np.float32) for d in resp.data]
    return np.vstack(vecs)

def _safe_embed_texts(texts: Sequence[str], api_key: Optional[str]) -> Optional[np.ndarray]:
    try:
        return _embed_texts(texts, api_key)
    except Exception as _e:
        print(f"⚠️ [SCORE] Embeddings unavailable ({repr(_e)}) — falling back to keyword matching")
        return None

# ---------------------------------------------------------------------------
# Visual analysis helpers (GPT-4o vision)
# ---------------------------------------------------------------------------

def _sample_frames_ffmpeg(
    video_path: str, max_frames: int = 6
) -> List[Tuple[str, float]]:
    """
    Smart frame sampling in three layers, now returning (base64, timestamp_sec)
    tuples so callers can tell GPT-4o *when* each frame occurs in the clip.

    1. Scene-change keyframes + showinfo timestamps — ffmpeg marks frames where
       the image changes dramatically and stderr gives their exact pts_time.
    2. Thumbnail frame — sharpest / most detail-rich frame (at ~midpoint ts).
    3. Evenly-spaced fallback with exact known timestamps.

    Returns up to max_frames (b64, timestamp_sec) tuples.
    """
    import subprocess
    import tempfile
    import glob
    import base64

    if not video_path or not os.path.exists(video_path):
        return []

    # ── probe duration ────────────────────────────────────────────────────
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True, check=True,
        )
        duration = float((probe.stdout or "0").strip() or "0")
    except Exception:
        return []
    if duration < 0.5:
        return []

    frames_with_ts: List[Tuple[str, float]] = []
    tmpdir = tempfile.mkdtemp()

    # ── layer 1: scene-change keyframes with timestamps via showinfo ──────
    # showinfo writes pts_time to stderr; parse it to recover exact timestamps.
    try:
        proc = subprocess.run(
            [
                "ffmpeg", "-y", "-i", video_path,
                "-vf", "select='gt(scene,0.30)',showinfo,scale=320:-1",
                "-vsync", "0", "-q:v", "4",
                "-frames:v", str(max_frames),
                os.path.join(tmpdir, "scene_%04d.jpg"),
            ],
            capture_output=True, text=True,
        )
        # Parse pts_time values from showinfo stderr lines.
        # Example line: [Parsed_showinfo_1 @ ...] n:   0 pts: ... pts_time:1.2833
        scene_ts: List[float] = [
            float(m.group(1))
            for m in re.finditer(r"pts_time:(\d+\.?\d*)", proc.stderr or "")
        ]
        scene_files = sorted(glob.glob(os.path.join(tmpdir, "scene_*.jpg")))
        for idx, fpath in enumerate(scene_files):
            try:
                if os.path.getsize(fpath) > 200:
                    with open(fpath, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode("utf-8")
                    # Use parsed timestamp if available, else linear estimate
                    ts = scene_ts[idx] if idx < len(scene_ts) else duration * (idx + 0.5) / max(1, len(scene_files))
                    frames_with_ts.append((b64, round(ts, 3)))
            except Exception:
                pass
            finally:
                try:
                    os.remove(fpath)
                except Exception:
                    pass
    except Exception:
        pass

    # ── layer 2: thumbnail (sharpest frame, approx middle of clip) ────────
    try:
        thumb_path = os.path.join(tmpdir, "thumb.jpg")
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", video_path,
                "-vf", "thumbnail=n=30,scale=320:-1",
                "-frames:v", "1", "-q:v", "3",
                thumb_path,
            ],
            capture_output=True, check=True,
        )
        if os.path.exists(thumb_path) and os.path.getsize(thumb_path) > 200:
            with open(thumb_path, "rb") as f:
                thumb_b64 = base64.b64encode(f.read()).decode("utf-8")
            thumb_ts = round(duration * 0.5, 3)
            existing_b64 = {fw[0] for fw in frames_with_ts}
            if thumb_b64 not in existing_b64:
                frames_with_ts.insert(0, (thumb_b64, thumb_ts))
        try:
            os.remove(thumb_path)
        except Exception:
            pass
    except Exception:
        pass

    # ── layer 3: evenly-spaced fallback (exact timestamps) ───────────────
    if len(frames_with_ts) < 2:
        frames_with_ts = []
        n_even = min(max_frames, 5)
        for i in range(n_even):
            t = duration * (i + 0.5) / n_even
            tmp_f = os.path.join(tmpdir, f"even_{i:04d}.jpg")
            try:
                subprocess.run(
                    ["ffmpeg", "-y", "-ss", f"{t:.3f}", "-i", video_path,
                     "-vframes", "1", "-q:v", "5", "-vf", "scale=320:-1", tmp_f],
                    capture_output=True, check=True,
                )
                if os.path.exists(tmp_f) and os.path.getsize(tmp_f) > 100:
                    with open(tmp_f, "rb") as f:
                        frames_with_ts.append((base64.b64encode(f.read()).decode("utf-8"), round(t, 3)))
            except Exception:
                pass
            finally:
                try:
                    os.remove(tmp_f)
                except Exception:
                    pass

    # cleanup tmpdir
    try:
        os.rmdir(tmpdir)
    except Exception:
        pass

    # Sort by timestamp so GPT-4o sees frames in temporal order.
    frames_with_ts.sort(key=lambda x: x[1])
    return frames_with_ts[:max_frames]


_VISUAL_DEFAULT: Dict = {
    "description": "",
    "shot_type": "unknown",
    "emotion": "neutral",
    "visual_score": 0.5,
    "visual_quality": 0.5,     # aesthetics only (sharpness, composition, light)
    "story_relevance": 0.5,    # semantic fit to the storyline, per vision model
    "narrative_role": "development",
    "best_moment_sec": None,
    "emotional_peak_sec": None,  # grounded timestamp of visible/audible emotion peak
    "trim_start_sec": None,
    "trim_end_sec": None,
    "motion": None,            # mean inter-frame luma difference (sensory)
    "audio_energy": None,      # mean RMS energy of the clip's own audio
    "face_ratio": None,        # fraction of sampled frames containing a face
    "vis_sig": None,           # 24-dim colour histogram for scene similarity
    "segments": [],   # populated for clips > 20 s with 2+ distinct good windows
}


def _visual_default() -> Dict:
    """Fresh default meta — dict(_VISUAL_DEFAULT) shared one mutable segments
    list across every fallback clip; this returns an independent copy."""
    d = dict(_VISUAL_DEFAULT)
    d["segments"] = []
    return d


def _audio_energy_profile(video_path: str) -> Optional[Dict]:
    """
    Extract audio energy profile from a clip using librosa.
    Returns a dict with peak_energy_sec, avg_energy, and a label (high/medium/low).
    Used to pass audio context to vision models alongside frames.
    """
    try:
        import subprocess
        import tempfile
        import librosa  # type: ignore

        # Extract mono audio to a temp wav
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_wav = tmp.name
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "22050",
             "-c:a", "pcm_s16le", tmp_wav],
            capture_output=True, check=True,
        )
        y, sr = librosa.load(tmp_wav, sr=22050, mono=True)
        try:
            os.remove(tmp_wav)
        except Exception:
            pass

        rms = librosa.feature.rms(y=y)[0]
        times = librosa.frames_to_time(range(len(rms)), sr=sr)
        avg_energy = float(np.mean(rms))
        peak_idx = int(np.argmax(rms))
        peak_sec = float(times[peak_idx])

        if avg_energy > 0.05:
            label = "high"
        elif avg_energy > 0.015:
            label = "medium"
        else:
            label = "low"

        return {
            "peak_energy_sec": round(peak_sec, 2),
            "avg_energy": round(avg_energy, 4),
            "energy_label": label,
        }
    except Exception:
        return None


def _motion_profile(video_path: str) -> Optional[float]:
    """Mean inter-frame luma difference (0-1) — how much the image moves.

    One ffmpeg pass: sample 2 fps at 160px, tblend difference between
    consecutive frames, signalstats reports the mean luma (YAVG) of each
    difference frame. High = action/camera movement, low = static shot.
    """
    import subprocess
    try:
        proc = subprocess.run(
            ["ffmpeg", "-i", video_path,
             "-vf", "fps=2,scale=160:-2,tblend=all_mode=difference,signalstats,"
                    "metadata=print:key=lavfi.signalstats.YAVG:file=-",
             "-f", "null", "-"],
            capture_output=True, text=True, timeout=60,
        )
        vals = [float(m.group(1))
                for m in re.finditer(r"lavfi\.signalstats\.YAVG=(\d+\.?\d*)", proc.stdout or "")]
        if len(vals) < 2:
            return None
        # First frame diffs against black — skip it. Normalise: YAVG of ~25+
        # on a difference frame is already vigorous motion.
        return round(min(1.0, float(np.mean(vals[1:])) / 25.0), 4)
    except Exception:
        return None


def _extract_tiny_frames(video_path: str, duration: float, n: int = 3,
                         w: int = 48, h: int = 27) -> List[np.ndarray]:
    """n small RGB frames, evenly spaced — shared by face + signature probes."""
    import subprocess
    frames = []
    for i in range(n):
        t = max(0.0, duration * (i + 0.5) / n)
        try:
            proc = subprocess.run(
                ["ffmpeg", "-ss", f"{t:.2f}", "-i", video_path,
                 "-vframes", "1", "-vf", f"scale={w}:{h}",
                 "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                capture_output=True, timeout=30,
            )
            if len(proc.stdout) == w * h * 3:
                frames.append(np.frombuffer(proc.stdout, dtype=np.uint8).reshape(h, w, 3))
        except Exception:
            pass
    return frames


def _face_profile(video_path: str, duration: float) -> Optional[float]:
    """Fraction of sampled frames containing a detectable face (0-1).

    Uses OpenCV's Haar cascade when available; returns None (signal unused)
    when opencv isn't installed — scoring degrades gracefully.
    """
    try:
        import cv2  # type: ignore
    except ImportError:
        return None
    try:
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        if cascade.empty():
            return None
        frames = _extract_tiny_frames(video_path, duration, n=4, w=256, h=144)
        if not frames:
            return None
        hits = 0
        for f in frames:
            gray = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
            gray = cv2.equalizeHist(gray)
            # Strict-ish params: Haar happily hallucinates faces in busy
            # texture; requiring more neighbor votes + a meaningful minimum
            # size keeps the signal usable at its small scoring weight.
            faces = cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=7,
                                             minSize=(28, 28))
            if len(faces) > 0:
                hits += 1
        return round(hits / len(frames), 3)
    except Exception:
        return None


def _visual_signature(video_path: str, duration: float) -> Optional[List[float]]:
    """24-dim colour histogram (8 bins × RGB) averaged over 3 tiny frames.

    Cheap scene fingerprint: two clips of the same location/lighting have
    cosine similarity near 1.0 even when their transcripts are empty."""
    frames = _extract_tiny_frames(video_path, duration, n=3)
    if not frames:
        return None
    hists = []
    for f in frames:
        h = [np.histogram(f[:, :, c], bins=8, range=(0, 255))[0] for c in range(3)]
        hists.append(np.concatenate(h).astype(np.float32))
    sig = np.mean(hists, axis=0)
    norm = float(np.linalg.norm(sig))
    if norm <= 0:
        return None
    return [round(float(v), 5) for v in (sig / norm)]


def _describe_clip_visually_gemini(
    video_path: str,
    clip_duration: float,
    google_api_key: str,
    story: str,
    tone: str,
    story_beats: Optional[List[str]] = None,
    audio_profile: Optional[Dict] = None,
    sensory: Optional[Dict] = None,   # {motion: 0-1, face_ratio: 0-1} local probes
) -> Dict:
    """
    Analyze a video clip using Gemini 2.5 Flash with native video input —
    the PRIMARY analysis path. Gemini receives the actual video file (not
    just frames), so it sees motion, timing, pacing, and hears the audio;
    its timestamps are grounded in the real stream rather than interpolated
    between sampled stills.
    """
    try:
        import google.generativeai as genai  # type: ignore
        genai.configure(api_key=google_api_key)

        model = genai.GenerativeModel("gemini-2.5-flash")

        # Upload video to Gemini Files API
        video_file = genai.upload_file(video_path)

        # Wait for processing
        import time
        while video_file.state.name == "PROCESSING":
            time.sleep(1)
            video_file = genai.get_file(video_file.name)

        if video_file.state.name != "ACTIVE":
            print(f"⚠️ [GEMINI] {os.path.basename(video_path)} not ACTIVE after upload: {video_file.state.name}")
            return _visual_default()

        beats_text = ", ".join(story_beats[:4]) if story_beats else "hook, development, payoff"
        audio_hint = ""
        if audio_profile:
            audio_hint = (
                f"Audio analysis: energy={audio_profile['energy_label']}, "
                f"peak at {audio_profile['peak_energy_sec']}s. "
                "Use peak_energy_sec as a strong signal for best_moment_sec when relevant.\n"
            )
        sensory_hint = ""
        if sensory:
            _parts = []
            if sensory.get("motion") is not None:
                _parts.append(f"measured motion level {sensory['motion']:.2f}/1.0")
            if sensory.get("face_ratio") is not None:
                _parts.append(f"faces visible in {int(sensory['face_ratio'] * 100)}% of sampled frames")
            if _parts:
                sensory_hint = "Local sensor readings: " + ", ".join(_parts) + ".\n"
        prompt = (
            f"You are a professional cinematographer analyzing a {round(clip_duration, 1)}s raw video clip for a short-form edit.\n"
            f"Story context: {(story or '')[:300]}\n"
            f"Narrative beats to serve: {beats_text}\n"
            f"Edit tone: {tone or 'cinematic'}\n"
            f"{audio_hint}{sensory_hint}\n"
            "Watch the full video including motion, pacing, audio tone, and visual quality.\n"
            "ALL timestamps must be grounded in the actual video timeline: seconds from clip "
            f"start, within [0.0, {round(clip_duration, 1)}]. Never estimate outside that range.\n"
            "Return ONLY strict JSON with these exact fields:\n"
            '{"description": "1-2 sentence visual summary", '
            '"shot_type": "close_up|medium|wide|extreme_wide|action|talking_head|product|landscape|broll", '
            '"emotion": "exciting|calm|tense|happy|sad|inspiring|neutral|dramatic", '
            '"visual_quality": 0.0-1.0, '
            '"story_relevance": 0.0-1.0, '
            '"narrative_role": "hook|development|turn|payoff|broll", '
            '"best_moment_sec": "float — the single most cinematic moment you actually saw", '
            '"emotional_peak_sec": "float — when emotion visibly/audibly peaks (laughter, tears, cheer, kiss); null if none", '
            '"trim_start_sec": "float — recommended in-point at a natural entry (action start, look up, door opens)", '
            '"trim_end_sec": "float — recommended out-point at a natural exit (action completes, beat lands)", '
            '"segments": "ONLY when the clip is longer than 20s AND contains 2-3 genuinely distinct high-quality sections '
            'separated by 3s+ gaps: array of {start_sec, end_sec, narrative_role, emotion, visual_score 0-1, description}, '
            'sorted by visual_score descending. Otherwise []"}'
        )

        response = model.generate_content([video_file, prompt])

        # Clean up the uploaded file to avoid storage buildup
        try:
            genai.delete_file(video_file.name)
        except Exception:
            pass

        raw = (response.text or "").strip()
        raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()
        parsed = json.loads(raw)

        vq = max(0.0, min(1.0, float(parsed.get("visual_quality", 0.5))))
        sr = max(0.0, min(1.0, float(parsed.get("story_relevance", 0.5))))

        def _safe_float(key: str) -> Optional[float]:
            v = parsed.get(key)
            if v is None:
                return None
            try:
                return float(v)
            except Exception:
                return None

        bm = _safe_float("best_moment_sec")
        ep = _safe_float("emotional_peak_sec")
        ts_start = _safe_float("trim_start_sec")
        ts_end = _safe_float("trim_end_sec")

        if bm is not None:
            bm = max(0.0, min(clip_duration, bm))
        if ep is not None:
            ep = max(0.0, min(clip_duration, ep))
        if ts_start is not None and ts_end is not None:
            ts_start = max(0.0, ts_start)
            ts_end = min(clip_duration, ts_end)
            if ts_end - ts_start < 1.0:
                ts_start = ts_end = None
        else:
            ts_start = ts_end = None

        parsed_segs = parsed.get("segments") or []
        validated_segs = []
        if isinstance(parsed_segs, list):
            for seg in parsed_segs[:3]:
                if not isinstance(seg, dict):
                    continue
                try:
                    s_start = float(seg["start_sec"])
                    s_end = float(seg["end_sec"])
                except (KeyError, TypeError, ValueError):
                    continue
                s_start = max(0.0, min(clip_duration, s_start))
                s_end = max(0.0, min(clip_duration, s_end))
                if s_end - s_start < 1.5:
                    continue
                validated_segs.append({
                    "start_sec": round(s_start, 3),
                    "end_sec": round(s_end, 3),
                    "best_moment_sec": round((s_start + s_end) / 2, 3),
                    "narrative_role": str(seg.get("narrative_role", "development")),
                    "emotion": str(seg.get("emotion", "neutral")),
                    "visual_score": max(0.0, min(1.0, float(seg.get("visual_score", 0.5)))),
                    "description": str(seg.get("description", "")),
                })

        return {
            "description": str(parsed.get("description", "")),
            "shot_type": str(parsed.get("shot_type", "unknown")),
            "emotion": str(parsed.get("emotion", "neutral")),
            "visual_score": round(0.40 * vq + 0.60 * sr, 4),
            "visual_quality": round(vq, 4),
            "story_relevance": round(sr, 4),
            "narrative_role": str(parsed.get("narrative_role", "development")),
            "best_moment_sec": bm,
            "emotional_peak_sec": ep,
            "trim_start_sec": ts_start,
            "trim_end_sec": ts_end,
            "segments": validated_segs,
        }
    except Exception as _e:
        # The legacy SDK embeds the API key in error URLs — redact it so it
        # never lands in server/Railway logs.
        _msg = repr(_e)
        if google_api_key:
            _msg = _msg.replace(google_api_key, "***REDACTED***")
        print(f"⚠️ [GEMINI] Analysis failed for {os.path.basename(video_path)}: {_msg}")
        return _visual_default()


def _describe_clip_visually(
    frames_with_ts: List[Tuple[str, float]],   # (base64, timestamp_sec)
    clip_duration: float,
    api_key: str,
    story: str,
    tone: str,
    story_beats: Optional[List[str]] = None,
    audio_profile: Optional[Dict] = None,
    sensory: Optional[Dict] = None,   # {motion: 0-1, face_ratio: 0-1} local probes
) -> Dict:
    """
    Send timestamped frames to GPT-4o vision.
    Returns shot analysis PLUS vision-guided trim recommendation:
      best_moment_sec  — single most cinematic/relevant moment
      trim_start_sec   — recommended trim window start
      trim_end_sec     — recommended trim window end
    """
    if not frames_with_ts or not api_key:
        print(f"⚠️ [GPT4V] No frames or no API key — skipping vision analysis, using defaults")
        return _visual_default()
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        # Build a human-readable frame index for GPT to reference
        frame_index_desc = ", ".join(
            f"frame_{i+1} @ {ts:.2f}s" for i, (_, ts) in enumerate(frames_with_ts[:5])
        )

        content: List[Dict] = [
            {
                "type": "text",
                "text": json.dumps({
                    "task": (
                        "Analyze these timestamped video frames. "
                        "Return ONLY strict JSON — no markdown, no commentary."
                    ),
                    "clip_duration_sec": round(clip_duration, 2),
                    "frame_index": frame_index_desc,
                    "story_context": (story or "")[:300],
                    "story_beats": (story_beats or [])[:4],
                    "tone": tone or "cinematic",
                    "audio_profile": audio_profile or {},
                    "sensory_profile": sensory or {},
                    "return_fields": {
                        "description": "1-2 sentence visual description of what is happening",
                        "shot_type": "one of: close_up | medium | wide | extreme_wide | action | talking_head | product | landscape | broll",
                        "emotion": "one of: exciting | calm | tense | happy | sad | inspiring | neutral | dramatic",
                        "visual_quality": "float 0.0-1.0: sharpness, composition, lighting quality",
                        "story_relevance": "float 0.0-1.0: how visually relevant is this clip to story_context",
                        "narrative_role": "one of: hook | development | turn | payoff | broll",
                        "best_moment_sec": (
                            "float: timestamp (in seconds) of the single most cinematic / "
                            "emotionally peak moment visible in the frames. Must be within clip_duration_sec."
                        ),
                        "emotional_peak_sec": (
                            "float or null: timestamp where emotion visibly peaks in the frames "
                            "(laughter, tears, embrace, celebration). Must be within clip_duration_sec; "
                            "null when no clear emotional peak is visible."
                        ),
                        "trim_start_sec": (
                            "float: recommended trim start (seconds from clip start). "
                            "Center your recommendation around best_moment_sec. "
                            "Must be >= 0 and < trim_end_sec."
                        ),
                        "trim_end_sec": (
                            "float: recommended trim end (seconds from clip start). "
                            "Aim for a natural ending — a beat, cut point, or composition change. "
                            "Must be <= clip_duration_sec."
                        ),
                        "segments": (
                            "ONLY include when clip_duration_sec > 20 AND there are 2 or 3 genuinely "
                            "distinct, high-quality sections separated by gaps of 3+ seconds. "
                            "Array of objects, each: {start_sec: float, end_sec: float, "
                            "narrative_role: hook|development|turn|payoff|broll, "
                            "emotion: exciting|calm|tense|happy|sad|inspiring|neutral|dramatic, "
                            "visual_score: float 0-1, description: str}. "
                            "Sort by visual_score descending. "
                            "Omit key entirely (or use []) if clip has only one good section."
                        ),
                    },
                }),
            }
        ]
        for b64, _ in frames_with_ts[:5]:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"},
            })

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a professional cinematographer and video editor. "
                        "Given timestamped frames from a raw clip, identify the peak visual moment "
                        "and recommend the ideal trim window for a short-form edit. "
                        "Return strict JSON only. No markdown fences. No text outside JSON."
                    ),
                },
                {"role": "user", "content": content},
            ],
            max_tokens=500,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        raw = (response.choices[0].message.content or "").strip()
        raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()
        parsed = json.loads(raw)

        vq = max(0.0, min(1.0, float(parsed.get("visual_quality", 0.5))))
        sr = max(0.0, min(1.0, float(parsed.get("story_relevance", 0.5))))

        # Validate and clamp vision trim values
        def _safe_float(key: str) -> Optional[float]:
            v = parsed.get(key)
            if v is None:
                return None
            try:
                return float(v)
            except Exception:
                return None

        bm = _safe_float("best_moment_sec")
        ep = _safe_float("emotional_peak_sec")
        ts_start = _safe_float("trim_start_sec")
        ts_end   = _safe_float("trim_end_sec")

        # Clamp to valid range
        if bm is not None:
            bm = max(0.0, min(clip_duration, bm))
        if ep is not None:
            ep = max(0.0, min(clip_duration, ep))
        if ts_start is not None and ts_end is not None:
            ts_start = max(0.0, ts_start)
            ts_end   = min(clip_duration, ts_end)
            # Reject if the window is too small or inverted
            if ts_end - ts_start < 1.0:
                ts_start = ts_end = None
        else:
            ts_start = ts_end = None

        # Parse multi-segment annotations (only present for long clips)
        parsed_segs = parsed.get("segments") or []
        validated_segs = []
        if isinstance(parsed_segs, list):
            for seg in parsed_segs[:3]:   # cap at 3 segments per clip
                if not isinstance(seg, dict):
                    continue
                try:
                    s_start = float(seg["start_sec"])
                    s_end   = float(seg["end_sec"])
                except (KeyError, TypeError, ValueError):
                    continue
                s_start = max(0.0, min(clip_duration, s_start))
                s_end   = max(0.0, min(clip_duration, s_end))
                if s_end - s_start < 1.5:
                    continue  # too short to be useful
                validated_segs.append({
                    "start_sec":       round(s_start, 3),
                    "end_sec":         round(s_end,   3),
                    "best_moment_sec": round((s_start + s_end) / 2, 3),
                    "narrative_role":  str(seg.get("narrative_role", "development")),
                    "emotion":         str(seg.get("emotion",        "neutral")),
                    "visual_score":    max(0.0, min(1.0, float(seg.get("visual_score", 0.5)))),
                    "description":     str(seg.get("description", "")),
                })

        return {
            "description":    str(parsed.get("description", "")),
            "shot_type":      str(parsed.get("shot_type", "unknown")),
            "emotion":        str(parsed.get("emotion", "neutral")),
            "visual_score":   round(0.40 * vq + 0.60 * sr, 4),
            "visual_quality":  round(vq, 4),
            "story_relevance": round(sr, 4),
            "narrative_role": str(parsed.get("narrative_role", "development")),
            "best_moment_sec": bm,
            "emotional_peak_sec": ep,
            "trim_start_sec":  ts_start,
            "trim_end_sec":    ts_end,
            "segments":        validated_segs,
        }
    except Exception as _e:
        print(f"⚠️ [GPT4V] Analysis failed: {repr(_e)}")
        return _visual_default()


def _score_clips_visually(
    transcriptions: List[Dict],
    story: str,
    tone: str,
    api_key: Optional[str],
    max_workers: int = 2,  # 2 instead of 4 — each worker holds frame data in RAM
    story_beats: Optional[List[str]] = None,
    progress_callback: Optional[callable] = None,  # (done: int, total: int, clip_name: str)
    vision_mode: str = "auto",  # "auto" | "gemini" | "gpt4o" (A/B comparison uses forced modes)
) -> Tuple[Dict[str, Dict], int]:  # returns (visual_meta, failed_count)
    """
    Run visual analysis concurrently for all clips.

    PRIMARY path: Gemini 2.5 Flash native video (set GOOGLE_API_KEY) — sees
    motion + hears audio, returns grounded timestamps.
    FALLBACK: GPT-4o frame sampling (OPENAI_API_KEY only).
    vision_mode forces one path for A/B comparison runs.
    Returns a dict keyed by clip path.
    Calls progress_callback(done, total, clip_name) after each clip finishes.
    """
    google_api_key = os.getenv("GOOGLE_API_KEY")

    def _probe_duration(path: str) -> float:
        import subprocess
        try:
            out = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", path],
                capture_output=True, text=True, check=True,
            )
            return float((out.stdout or "0").strip() or "0")
        except Exception:
            return 0.0

    def _analyse(t: Dict) -> Tuple[str, Dict]:
        path = t.get("path", "")
        if not path or not os.path.exists(path):
            return path, _visual_default()
        clip_dur = _probe_duration(path)

        # Local sensory probes run FIRST so both vision models receive them as
        # hints, and their numeric values feed rel_scores directly afterwards.
        audio_profile = _audio_energy_profile(path)
        motion = _motion_profile(path)
        face_ratio = _face_profile(path, clip_dur)
        vis_sig = _visual_signature(path, clip_dur)
        sensory_hints = {"motion": motion, "face_ratio": face_ratio}

        def _attach_sensory(res: Dict) -> Dict:
            res["duration_sec"] = clip_dur
            res["audio_energy"] = (audio_profile or {}).get("avg_energy")
            res["motion"] = motion
            res["face_ratio"] = face_ratio
            res["vis_sig"] = vis_sig
            return res

        # PRIMARY: Gemini native video analysis (sees motion + audio, grounded
        # timestamps). Skipped when vision_mode forces the GPT-4o path.
        if google_api_key and vision_mode in ("auto", "gemini"):
            result = _describe_clip_visually_gemini(
                path, clip_dur, google_api_key, story, tone,
                story_beats=story_beats,
                audio_profile=audio_profile,
                sensory=sensory_hints,
            )
            # Only fall back to GPT-4o if Gemini returned the default (failed)
            if result != _VISUAL_DEFAULT:
                return path, _attach_sensory(result)

        # FALLBACK: GPT-4o frame sampling — more frames for longer clips
        adaptive_frames = 10 if clip_dur > 15 else 6
        frames_with_ts = _sample_frames_ffmpeg(path, max_frames=adaptive_frames)
        result = _describe_clip_visually(
            frames_with_ts, clip_dur, api_key or "", story, tone,
            story_beats=story_beats,
            audio_profile=audio_profile,
            sensory=sensory_hints,
        )
        return path, _attach_sensory(result)

    results: Dict[str, Dict] = {}
    total = len(transcriptions)
    done = 0
    failed_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_analyse, t): t for t in transcriptions}
        for fut in concurrent.futures.as_completed(futures):
            try:
                path, meta = fut.result()
                # Count clips that fell back to defaults (AI analysis didn't run).
                # (meta now carries duration_sec, so compare fields, not the whole dict.)
                if meta.get("visual_score") == 0.5 and not meta.get("description"):
                    failed_count += 1
            except Exception as _thread_exc:
                t = futures[fut]
                path = t.get("path", "")
                print(f"⚠️ [VISION] Thread failed for {os.path.basename(path)}: {repr(_thread_exc)}")
                meta = _visual_default()
                failed_count += 1
            if path:
                results[path] = meta
            done += 1
            if callable(progress_callback):
                clip_name = os.path.basename(path) if path else f"clip {done}"
                try:
                    progress_callback(done, total, clip_name)
                except Exception:
                    pass
    return results, failed_count


# ---------------------------------------------------------------------------
# Shot continuity checker + auto-repair
# ---------------------------------------------------------------------------

# Cinematographic shot-size hierarchy: 0=widest, 5=tightest
_SHOT_RANK: Dict[str, int] = {
    "extreme_wide": 0,
    "landscape":    1,
    "wide":         2,
    "medium":       3,
    "broll":        3,  # broll is size-neutral
    "product":      3,
    "action":       3,  # action is size-neutral
    "talking_head": 4,
    "close_up":     5,
    "unknown":      3,
}

# Which pairs make for a jarring cut (beyond the hierarchy jump)
_JARRING_PAIRS = {
    ("close_up",     "close_up"),      # same tight framing, different subjects = jump cut
    ("extreme_wide", "close_up"),      # spatial disorientation
    ("close_up",     "extreme_wide"),
    ("talking_head",  "talking_head"), # 180° rule risk if not same convo
}


def _is_jarring_cut(shot_a: str, shot_b: str) -> bool:
    """Return True if cutting directly from shot_a to shot_b violates shot grammar."""
    a = (shot_a or "unknown").lower()
    b = (shot_b or "unknown").lower()
    if (a, b) in _JARRING_PAIRS or (b, a) in _JARRING_PAIRS:
        return True
    rank_a = _SHOT_RANK.get(a, 3)
    rank_b = _SHOT_RANK.get(b, 3)
    return abs(rank_a - rank_b) >= 3


def _fix_shot_continuity(
    ordered: List[Dict],
    full_pool: List[Dict],
    max_fixes: int = 4,
    cost_of: Optional[callable] = None,   # clip dict → planned screen-time (sec)
    budget_sec: Optional[float] = None,   # planned duration cap for the whole edit
    protect_ends: bool = False,           # never swap out the closing clip (arc pass chose it)
) -> List[Dict]:
    """
    Walk the ordered clip list and repair jarring consecutive cuts.

    Strategy:
    A) Insert a broll/neutral clip from the unused pool between the jarring pair
       — only if the insert still fits within budget_sec (duration-budget-neutral).
    B) If no broll is available (or budget is exhausted), swap the second clip in
       the pair for the highest-scored unused clip whose shot_type is compatible
       with the first. Swaps trade one clip for another, so the budget is safe.
    C) If neither fix is possible, leave the pair unchanged (don't make it worse).

    Only performs up to max_fixes repairs to avoid thrashing the edit.
    """
    if len(ordered) < 2:
        return ordered

    used_paths = {d["path"] for d in ordered}
    # Build a pool of clips not already in the edit, sorted by visual score desc
    unused = sorted(
        [d for d in full_pool if d.get("path") and d["path"] not in used_paths],
        key=lambda d: d.get("visual_score", 0.5),
        reverse=True,
    )

    result = list(ordered)
    planned_total = sum(cost_of(d) for d in result) if cost_of else 0.0
    fixes = 0
    i = 0
    while i < len(result) - 1 and fixes < max_fixes:
        shot_a = result[i].get("shot_type", "unknown")
        shot_b = result[i + 1].get("shot_type", "unknown")

        if not _is_jarring_cut(shot_a, shot_b):
            i += 1
            continue

        # Strategy A: find an unused broll/neutral buffer clip.
        # Skipped entirely when the planned duration budget is already spent —
        # an insert would push the edit past the user's target length.
        buffer = None
        _budget_allows_insert = True
        if cost_of and budget_sec is not None:
            _budget_allows_insert = planned_total < budget_sec
        if _budget_allows_insert:
            for idx, candidate in enumerate(unused):
                cshot = (candidate.get("shot_type") or "unknown").lower()
                if cshot in ("broll", "medium", "wide", "action", "product"):
                    if not _is_jarring_cut(shot_a, cshot) and not _is_jarring_cut(cshot, shot_b):
                        buffer = unused.pop(idx)
                        break

        if buffer:
            buffer["narrative_role"] = "broll"  # mark as buffer
            result.insert(i + 1, buffer)
            if cost_of:
                planned_total += cost_of(buffer)
            fixes += 1
            i += 2  # skip the inserted clip and the next pair
            continue

        # Strategy B: swap result[i+1] for a compatible unused clip at same beat.
        # When the arc pass chose the closer deliberately, leave it alone.
        if protect_ends and i + 1 == len(result) - 1:
            i += 1
            continue
        target_rank = _SHOT_RANK.get(shot_a.lower(), 3)
        swap = None
        for idx, candidate in enumerate(unused):
            cshot = (candidate.get("shot_type") or "unknown").lower()
            crank = _SHOT_RANK.get(cshot, 3)
            if abs(crank - target_rank) < 3 and (shot_a.lower(), cshot) not in _JARRING_PAIRS:
                swap = unused.pop(idx)
                break

        if swap:
            old = result[i + 1]
            # preserve narrative_role of the original clip
            swap["narrative_role"] = old.get("narrative_role", swap.get("narrative_role", "development"))
            result[i + 1] = swap
            # put the swapped-out clip back into unused for potential later use
            unused.append(old)
            if cost_of:
                planned_total += cost_of(swap) - cost_of(old)
            fixes += 1

        i += 1

    return result


# ---------------------------------------------------------------------------

def _split_story_into_segments(story: str) -> List[str]:
    parts = re.split(r"[\n\.!?;:]+", story or "")
    segments = [p.strip() for p in parts if p and len(p.strip()) > 3]
    if not segments:
        base = (story or "").strip() or "general video"
        return [base, base, base]
    # Single-sentence story: synthesize 3 variants so beat matching has surface area
    if len(segments) == 1:
        return [segments[0], segments[0], segments[0]]
    return segments[:8]

def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9']+", (text or "").lower())

def _auto_keywords(story: str) -> set[str]:
    stop = {
        "the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "for", "with",
        "is", "are", "was", "were", "be", "been", "being", "that", "this", "it", "as",
        "at", "by", "from", "i", "you", "we", "they", "he", "she", "them", "our", "your"
    }
    toks = [t for t in _tokenize(story) if len(t) >= 4 and t not in stop]
    if not toks:
        return set()
    freq = {}
    for token in toks:
        freq[token] = freq.get(token, 0) + 1
    return {k for k, _ in sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:12]}

def _text_quality(text: str) -> float:
    t = (text or "").strip()
    if not t:
        return 0.0
    chars = len(t)
    alpha = sum(ch.isalpha() for ch in t)
    tokens = _tokenize(t)
    uniq = len(set(tokens))
    alpha_ratio = alpha / max(1, chars)
    length_score = min(1.0, chars / 80.0)
    diversity = min(1.0, uniq / max(1, len(tokens)))
    return float(max(0.0, min(1.0, 0.5 * length_score + 0.3 * alpha_ratio + 0.2 * diversity)))

def _lexical_overlap(text: str, keywords: set[str]) -> float:
    if not keywords:
        return 0.0
    tokens = set(_tokenize(text))
    if not tokens:
        return 0.0
    matches = len(tokens.intersection(keywords))
    return matches / max(1, len(keywords))

def _mmr_select(
    indices: List[int],
    rel_scores: np.ndarray,
    embeds: np.ndarray,
    k: int,
    lam: float = 0.78,
    shot_types: Optional[List[str]] = None,
    costs: Optional[List[float]] = None,
    budget_sec: Optional[float] = None,
    min_count: int = 1,
    vis_sigs: Optional[List[Optional[List[float]]]] = None,
    text_empty: Optional[List[bool]] = None,
) -> List[int]:
    """Maximal Marginal Relevance with optional shot-type diversity penalty
    and an optional duration budget.

    shot_types, when provided, must be indexed 1:1 with rel_scores/embeds.
    Each additional clip of the same shot_type loses 0.18 score after the
    first occurrence — forcing visual variety at selection time.

    vis_sigs/text_empty: silent clips embed identically (their transcripts
    are all empty), so text redundancy is blind to them. When either clip of
    a pair has no transcript, redundancy is measured on the colour-histogram
    visual signature instead — same-scene b-roll finally reads as redundant.

    costs/budget_sec, when provided, turn selection into a greedy knapsack:
    clips are added in MMR order until their summed planned screen time
    reaches budget_sec (the clip that crosses the budget is still included,
    so the pool slightly over-fills — the render allocator trims down, never
    pads up). k remains an absolute upper bound on count.
    """
    if not indices:
        return []

    def _pair_redundancy(i: int, j: int) -> float:
        if (
            text_empty is not None
            and vis_sigs is not None
            and (text_empty[i] or text_empty[j])
        ):
            si, sj = vis_sigs[i], vis_sigs[j]
            if si is not None and sj is not None:
                return _cosine_sim(
                    np.asarray(si, dtype=np.float32), np.asarray(sj, dtype=np.float32)
                )
        return _cosine_sim(embeds[i], embeds[j])

    chosen = [max(indices, key=lambda i: rel_scores[i])]
    selected_shots: List[Optional[str]] = [
        shot_types[chosen[0]] if shot_types else None
    ]
    spent = float(costs[chosen[0]]) if costs is not None else 0.0
    remaining = set(indices) - set(chosen)
    while remaining and len(chosen) < k:
        if (
            budget_sec is not None
            and costs is not None
            and spent >= budget_sec
            and len(chosen) >= min_count
        ):
            break
        best_i = None
        best_score = -1e9
        for i in remaining:
            redundancy = max(_pair_redundancy(i, j) for j in chosen) if chosen else 0.0
            score = lam * float(rel_scores[i]) - (1.0 - lam) * float(redundancy)
            # Shot-type diversity penalty per duplicate beyond first occurrence.
            # Prevents selecting 3 wide shots when a close-up exists (even if
            # slightly less relevant) — how a human editor enforces variety.
            if shot_types:
                cand_shot = shot_types[i]
                already = sum(1 for s in selected_shots if s == cand_shot)
                score -= 0.18 * max(0, already - 1)
            if score > best_score:
                best_score = score
                best_i = i
        chosen.append(best_i)
        selected_shots.append(shot_types[best_i] if shot_types else None)
        if costs is not None:
            spent += float(costs[best_i])
        remaining.remove(best_i)
    return chosen

# Desired emotional arc per tone — used in the editorial rerank prompt so GPT-4o
# reasons about sequence shape, not just semantic similarity.
_ARC_DESCRIPTIONS: Dict[str, str] = {
    "energetic":   "start high-energy → build excitement → dramatic peak → strong close",
    "epic":        "powerful open → tension builds → dramatic crescendo → triumphant close",
    "cinematic":   "intriguing open → develop story beats → emotional turn → inspiring payoff",
    "sentimental": "gentle open → personal moments → emotional depth → uplifting resolution",
    "calm":        "peaceful establish → gentle journey → quiet reflection → serene close",
}

def _llm_arc_pass(
    story: str,
    transcriptions: List[Dict[str, str]],
    candidate_indices: List[int],
    api_key: Optional[str],
    tone: Optional[str] = None,
    visual_meta: Optional[Dict[str, Dict]] = None,
    costs: Optional[List[float]] = None,
    budget_sec: Optional[float] = None,
    rel_scores: Optional[np.ndarray] = None,
    avg_shot_len: Optional[float] = None,
) -> Optional[Tuple[List[int], Dict[int, float], Dict[int, str]]]:
    """Single structured editorial pass (replaces the old rerank →
    LLM-continuity-check → hook/payoff-pin chain, whose passes overwrote
    each other).

    GPT-4o receives every clip WITH its planned screen time and the duration
    budget, assigns story-arc slots (hook → build → climax → resolution),
    orders the sequence, may DROP weak clips, assigns each kept clip a
    relative screen_time_weight — its own editorial judgment of how much
    time a clip deserves given the whole set, rather than a fixed per-role
    weight — and picks the transition effect for the cut INTO each clip
    (it already reasons about emotional direction between neighbours, so
    it's better placed than a weighted-random pool to decide whether that
    cut should be a hard cut, a dissolve, or something kinetic).
    Hook/payoff placement and shot-variety are constraints inside this one
    prompt rather than blind post-swaps. Returns
    (ordered indices, {clip_id: weight}, {clip_id: transition_into_this_clip}),
    or None on failure (callers fall back to deterministic ordering +
    pinning + fixed role/emotion weights + heuristic transition picking)."""
    if not api_key or not candidate_indices:
        return None

    try:
        from openai import OpenAI
        from transition import list_available_transitions

        client = OpenAI(api_key=api_key)
        tone_label = (tone or "cinematic").strip().lower()
        arc_desc = _ARC_DESCRIPTIONS.get(tone_label, "hook → development → emotional peak → payoff")
        transition_vocab = list_available_transitions()
        transition_guide = {
            "crossfade":   "neutral dissolve — always safe, use when nothing else fits",
            "fadein":      "rises from black — soft opens, new chapter, gentle emotional beat",
            "fadeout":     "settles to black — soft closes, resolution, a breath before the next beat",
            "zoom_in":     "pushes in — arriving energy, building excitement, tightening focus",
            "zoom_out":    "pulls back — release after a climax, revealing scale, resolution",
            "slide_left":  "kinetic, directional — momentum, forward progress",
            "slide_right": "kinetic, directional — momentum, a turn or reversal",
            "slide_up":    "kinetic, rising — energy building, excitement",
            "slide_down":  "kinetic, falling — grounding, settling down",
        }

        # How many clips a 60s-target Energetic edit actually needs is very
        # different from a 60s-target Calm one (2.6s vs 4.8s average shot).
        # A flat "keep at least N" floor ignores that entirely, so a model
        # that judges most candidates "weak" can prune all the way down to a
        # tiny handful — leaving editor.py to stretch each survivor across
        # many seconds of screen time, which reads as "barely any cutting."
        _min_keep_floor = min(
            len(candidate_indices),
            max(4, math.ceil(budget_sec / avg_shot_len)) if (budget_sec and avg_shot_len) else 4,
        )

        clip_lines = []
        for idx in candidate_indices:
            t = transcriptions[idx]
            path = t.get("path", "")
            vm = (visual_meta or {}).get(path, {})
            text = re.sub(r"\s+", " ", str(t.get("text", "") or "").strip())
            clip_lines.append({
                "id": idx,
                "role": vm.get("narrative_role") or "development",
                "shot": vm.get("shot_type") or "unknown",
                "emotion": vm.get("emotion") or "neutral",
                "quality": round(float(vm.get("visual_quality") or vm.get("visual_score") or 0.5), 2),
                "motion": round(float(vm.get("motion") or 0.0), 2),
                "seconds": round(float(costs[idx]), 1) if costs is not None else None,
                "peak_sec": vm.get("emotional_peak_sec"),
                "desc": (vm.get("description") or "")[:80],
                "speech": text[:100],
            })

        payload = {
            "task": (
                "You are the lead editor assembling a short-form video. "
                "Arrange these clips into a story arc; drop weak clips if needed."
            ),
            "tone": tone_label,
            "story": (story or "")[:300],
            "arc": arc_desc,
            "duration_budget_sec": round(float(budget_sec), 1) if budget_sec else None,
            "clips": clip_lines,
            "rules": [
                "OPEN with the strongest hook: action, striking visual, or emotional grab — never static/generic.",
                "Assign every kept clip an arc slot: hook | build | climax | resolution.",
                "Place the climax (highest emotion/motion) about 60-75% of the way through the sequence.",
                "CLOSE with resolution/payoff — emotional close-up or wide establishing shot.",
                "Never place the same shot type back-to-back when another order exists.",
                "Avoid jarring cuts: no close_up ↔ extreme_wide jumps; move emotion gradually.",
                f"Each clip's 'seconds' is its planned screen time. If the kept clips' summed seconds exceed "
                f"duration_budget_sec, DROP the weakest (low quality, redundant scenes) — but keep at least "
                f"{_min_keep_floor} clips. That floor is deliberately set so the remaining clips can fill "
                f"duration_budget_sec without any single clip needing to run far longer than its planned "
                f"'seconds' — don't drop below it even if several clips seem weak or similar.",
                "For every KEPT clip, also assign a screen_time_weight (0.4-2.0): how much MORE or LESS time it "
                "deserves relative to an average clip, given this specific story and the other clips in this set. "
                "The climax/payoff moment usually deserves the most (1.4-2.0); pure connective b-roll the least "
                "(0.4-0.7). Base this on what actually matters for THIS edit, not a fixed rule per role.",
                "For every KEPT clip EXCEPT the first, also pick the transition used for the cut INTO it, from "
                "transition_vocab, based on the emotional/energy change from the PRECEDING clip in your final "
                "order to this one, and the tone. Don't use the same transition on two consecutive cuts unless "
                "nothing else fits.",
            ],
            "transition_vocab": transition_guide,
            "format": {
                "order": ["kept clip ids in final sequence"],
                "dropped": ["removed clip ids"],
                "weights": {"clip id (as string)": "screen_time_weight float"},
                "transitions": {"clip id (as string), omit the first clip in order": "transition name from transition_vocab"},
            },
        }

        _budget_label = (f"{payload['duration_budget_sec']}s"
                         if payload["duration_budget_sec"] else "none")
        print(f"[ARC] Structured arc pass on {len(candidate_indices)} clips "
              f"(tone={tone_label}, budget={_budget_label})")

        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.0,
            messages=[
                {"role": "system", "content": "You are a professional video editor. Output strict JSON only."},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            max_tokens=750,
            response_format={"type": "json_object"},
        )

        parsed = json.loads(response.choices[0].message.content or "{}")
        raw_order = parsed.get("order", []) if isinstance(parsed, dict) else []
        raw_dropped = parsed.get("dropped", []) if isinstance(parsed, dict) else []
        raw_weights = parsed.get("weights", {}) if isinstance(parsed, dict) else {}
        raw_transitions = parsed.get("transitions", {}) if isinstance(parsed, dict) else {}

        allowed = set(candidate_indices)
        weights: Dict[int, float] = {}
        if isinstance(raw_weights, dict):
            for k, v in raw_weights.items():
                try:
                    cid, w = int(k), float(v)
                except (TypeError, ValueError):
                    continue
                if cid in allowed and w > 0:
                    weights[cid] = max(0.2, min(2.5, w))
        transitions: Dict[int, str] = {}
        if isinstance(raw_transitions, dict):
            for k, v in raw_transitions.items():
                try:
                    cid = int(k)
                except (TypeError, ValueError):
                    continue
                ttype = str(v or "").strip().lower()
                if cid in allowed and ttype in transition_vocab:
                    transitions[cid] = ttype
        dropped: set = set()
        for item in raw_dropped:
            try:
                d = int(item)
            except Exception:
                continue
            if d in allowed:
                dropped.add(d)

        seen: set = set()
        cleaned: List[int] = []
        for item in raw_order:
            try:
                cid = int(item)
            except Exception:
                continue
            if cid in allowed and cid not in seen and cid not in dropped:
                seen.add(cid)
                cleaned.append(cid)

        # Clips the LLM neither kept nor dropped: keep them (never lose clips silently)
        for idx in candidate_indices:
            if idx not in seen and idx not in dropped:
                cleaned.append(idx)
                seen.add(idx)

        # Don't let the LLM gut the edit below a usable minimum — but respect
        # its explicit drops: only resurrect dropped clips when the sequence
        # would otherwise fall under the budget-aware floor computed above.
        min_keep = _min_keep_floor
        if len(cleaned) < min_keep:
            for idx in candidate_indices:
                if idx not in seen:
                    cleaned.append(idx)
                    seen.add(idx)
                if len(cleaned) >= min_keep:
                    break

        # Budget backstop: if the LLM kept too much, trim the weakest middle
        # clips (opener and closer are protected).
        if costs is not None and budget_sec:
            def _rel(i: int) -> float:
                return float(rel_scores[i]) if rel_scores is not None else 0.0
            total = sum(float(costs[i]) for i in cleaned)
            while total > budget_sec * 1.05 and len(cleaned) > min_keep and len(cleaned) > 2:
                victim = min(cleaned[1:-1], key=_rel, default=None)
                if victim is None:
                    break
                cleaned.remove(victim)
                total -= float(costs[victim])
                dropped.add(victim)

        if dropped:
            print(f"[ARC] Dropped {len(dropped)} clip(s) to fit budget/arc: {sorted(dropped)}")

        return (cleaned, weights, transitions) if cleaned else None
    except Exception as _e:
        print(f"⚠️ [ARC] Structured arc pass failed: {repr(_e)}")
        return None


def _llm_story_plan(story: str, api_key: Optional[str], tone: Optional[str] = None) -> Optional[List[str]]:
    if not api_key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        payload = {
            "task": "Create a concise edit blueprint.",
            "instructions": [
                "Return strict JSON only.",
                "Provide 3 to 6 beats in narrative order.",
                "Each beat should be a short phrase, max 10 words.",
            ],
            "tone": (tone or "cinematic"),
            "story": story,
            "format": {"beats": ["hook", "development", "turn", "payoff"]},
        }
        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.1,
            messages=[
                {"role": "system", "content": "You are a senior narrative video editor. Output strict JSON only."},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            max_tokens=300,
            response_format={"type": "json_object"},
        )
        content = (response.choices[0].message.content or "").strip()
        parsed = json.loads(content)
        beats = parsed.get("beats", []) if isinstance(parsed, dict) else []
        clean = [str(b).strip() for b in beats if str(b).strip()]
        if 3 <= len(clean) <= 8:
            return clean
        return clean[:8] if clean else None
    except Exception:
        return None


def _pin_hook_and_payoff(result: List[Dict]) -> List[Dict]:
    """Ensure the edit opens with the best hook and closes with the best payoff.

    A human editor agonizes over the opening and closing shots. This function
    enforces that invariant after all other ordering passes are complete.
    - Opening: highest hook_score (narrative_role==hook OR exciting/dramatic/inspiring emotion
      AND wide/action shot_type)
    - Closing: highest payoff_score (narrative_role==payoff OR inspiring/happy/dramatic emotion)
    Only swaps if a better candidate exists elsewhere in the list.
    """
    if len(result) < 3:
        return result

    _hook_emotions = {"exciting", "dramatic", "inspiring"}
    _hook_shots    = {"wide", "action", "extreme_wide", "landscape"}
    _payoff_emotions = {"inspiring", "happy", "dramatic", "calm"}
    _payoff_shots    = {"wide", "extreme_wide", "close_up", "landscape"}

    def _hook_score(clip: Dict) -> float:
        s = 0.0
        if clip.get("narrative_role") == "hook":              s += 2.0
        if clip.get("emotion") in _hook_emotions:             s += 1.0
        if clip.get("shot_type") in _hook_shots:              s += 0.5
        s += float(clip.get("visual_score") or 0.5)
        return s

    def _payoff_score(clip: Dict) -> float:
        s = 0.0
        if clip.get("narrative_role") == "payoff":            s += 2.0
        if clip.get("emotion") in _payoff_emotions:           s += 1.0
        if clip.get("shot_type") in _payoff_shots:            s += 0.5
        s += float(clip.get("visual_score") or 0.5)
        return s

    # Pin opening: find best hook anywhere except the last position
    best_hook_idx = max(range(len(result) - 1), key=lambda i: _hook_score(result[i]))
    if best_hook_idx != 0:
        result[0], result[best_hook_idx] = result[best_hook_idx], result[0]

    # Pin closing: find best payoff anywhere except the first position (now pinned)
    best_payoff_idx = max(range(1, len(result)), key=lambda i: _payoff_score(result[i]))
    if best_payoff_idx != len(result) - 1:
        result[-1], result[best_payoff_idx] = result[best_payoff_idx], result[-1]

    print(f"[EDITORIAL] Pinned opening: {result[0].get('emotion','?')} {result[0].get('shot_type','?')} "
          f"({result[0].get('narrative_role','?')})  |  "
          f"closing: {result[-1].get('emotion','?')} {result[-1].get('shot_type','?')} "
          f"({result[-1].get('narrative_role','?')})")
    return result


def _reduce_adjacent_similarity(
    result: List[Dict],
    sim_threshold: float = 0.92,
    max_swaps: int = 3,
) -> List[Dict]:
    """Break up back-to-back near-identical scenes.

    Transcript embeddings can't see that two clips show the same location and
    lighting; the colour-histogram vis_sig can. Any adjacent pair whose
    signatures match above sim_threshold gets the second clip swapped with a
    later clip, provided the swap doesn't create a new same-scene pair.
    Opener and closer stay pinned."""
    if len(result) < 4:
        return result

    def _vis_sim(a: Dict, b: Dict) -> float:
        sa, sb = a.get("vis_sig"), b.get("vis_sig")
        if not sa or not sb:
            return 0.0
        return _cosine_sim(np.asarray(sa, dtype=np.float32),
                           np.asarray(sb, dtype=np.float32))

    def _pair_bad(k: int) -> bool:
        return 0 <= k < len(result) - 1 and _vis_sim(result[k], result[k + 1]) >= sim_threshold

    swaps = 0
    i = 0
    while i < len(result) - 2 and swaps < max_swaps:  # i+1 never touches the closer
        if not _pair_bad(i):
            i += 1
            continue
        for j in range(i + 2, len(result) - 1):  # j never touches the closer either
            result[i + 1], result[j] = result[j], result[i + 1]
            if not (_pair_bad(i) or _pair_bad(i + 1) or _pair_bad(j - 1) or _pair_bad(j)):
                swaps += 1
                print(f"[VARIETY] Swapped positions {i + 1}<->{j} to break a same-scene run")
                break
            result[i + 1], result[j] = result[j], result[i + 1]  # revert
        i += 1
    return result


def score_clips_with_story(
    transcriptions: List[Dict[str, str]],
    story: str,
    priority_keywords: Optional[List[str]] = None,
    exclude_keywords: Optional[List[str]] = None,
    tone: Optional[str] = None,
    target_duration_sec: Optional[int] = None,
    openai_api_key: Optional[str] = None,
    progress_callback: Optional[callable] = None,  # (done, total, clip_name)
    vision_mode: str = "auto",  # "auto" | "gemini" | "gpt4o" — forced by A/B compare
) -> List[Dict]:
    """
    Returns clips ranked by hybrid visual + semantic + lexical relevance,
    with narrative beat assignment.
    Each item in the returned list is a dict:
      {path, narrative_role, shot_type, emotion, visual_score}
    so _normalize_to_paths() in app.py still works unchanged.
    """
    api_key = _get_api_key(openai_api_key)
    if not api_key:
        print("⚠️ [SCORE] No API key — skipping all AI analysis, using defaults for all clips")
        return [{"path": t["path"], "narrative_role": "development",
                 "shot_type": "unknown", "emotion": "neutral", "visual_score": 0.5}
                for t in transcriptions if t.get("path")]

    texts = [str(t.get("text", "") or "") for t in transcriptions]

    story_text = str(story or "").strip() or " "

    # ── Step 1: story plan first so visual analysis can align to beats ────
    # We run planning synchronously — it's fast (gpt-4o-mini) and short.
    # The beats go into every per-clip GPT-4o prompt as context so the model
    # knows which narrative moment each clip should serve.
    _llm_plan = _llm_story_plan(story_text, api_key, tone)
    if not _llm_plan:
        print("⚠️ [SCORE] LLM story plan failed — using punctuation split as fallback")
    story_segments = _llm_plan or _split_story_into_segments(story_text)

    # ── Step 2: visual analysis with story beats as context ───────────────
    visual_meta: Dict[str, Dict]
    visual_meta, _failed_clips = _score_clips_visually(
        transcriptions, story_text, tone or "", api_key,
        story_beats=story_segments,
        progress_callback=progress_callback,
        vision_mode=vision_mode,
    )
    if _failed_clips > 0:
        print(f"⚠️ [SCORE] {_failed_clips}/{len(transcriptions)} clip(s) fell back to default metadata (AI analysis unavailable)")

    # ── Normalize visual_score across batch ───────────────────────────────
    # GPT-4o scores cluster in 0.6-0.85; normalizing to [0.15, 0.95] gives
    # the allocation engine 5× more differentiation between clips.
    _vs_values = [
        float(visual_meta.get(t.get("path", ""), {}).get("visual_score", 0.5))
        for t in transcriptions
    ]
    _vs_min, _vs_max = min(_vs_values), max(_vs_values)
    if _vs_max - _vs_min > 0.05:
        for t in transcriptions:
            p = t.get("path", "")
            if p in visual_meta:
                raw_vs = float(visual_meta[p].get("visual_score", 0.5))
                normalized = 0.15 + 0.80 * (raw_vs - _vs_min) / (_vs_max - _vs_min)
                visual_meta[p]["visual_score"] = round(normalized, 4)

    story_keywords = _auto_keywords(story_text)

    pri = set((priority_keywords or []))
    exc = set((exclude_keywords or []))

    qualities = np.array([_text_quality(text) for text in texts], dtype=np.float32)
    # A clip is valid if it has decent text quality OR strong visual relevance
    valid_idx = [
        i for i, q in enumerate(qualities)
        if q >= 0.12
        or visual_meta.get(transcriptions[i].get("path", ""), {}).get("visual_score", 0.0) >= 0.40
    ]
    if not valid_idx:
        valid_idx = list(range(len(texts)))

    clip_vecs = _safe_embed_texts(texts, api_key)
    segment_vecs = _safe_embed_texts(story_segments, api_key)

    tone_key = (tone or "").strip().lower()
    # Per-tone shot-length band (min, mid, max) seconds. mid is the fallback
    # estimate when a clip's real duration is unknown; min/max clamp how much
    # planned screen time any single clip contributes to the duration budget.
    _SHOT_BANDS = {
        "energetic":   (1.8, 2.6, 4.5),
        "epic":        (2.2, 3.2, 5.5),
        "cinematic":   (2.5, 3.8, 6.5),
        "sentimental": (3.0, 4.2, 7.5),
        "calm":        (3.5, 4.8, 9.0),
    }
    _min_shot, avg_shot_len, _max_shot = _SHOT_BANDS.get(tone_key, _SHOT_BANDS["cinematic"])
    desired_duration = float(target_duration_sec) if target_duration_sec else 45.0
    # Selection is a duration budget, not a clip count. The render allocator
    # trims down but never pads up, so plan ~15% MORE screen time than the
    # target — transition overlap alone consumes ~0.3s per cut, and headroom
    # lets the allocator drop or shorten weak clips while still landing on
    # target.
    planning_budget = desired_duration * 1.15

    def _planned_cost(i: int) -> float:
        """Planned screen time clip i contributes to the edit (seconds)."""
        vm = visual_meta.get(transcriptions[i].get("path", ""), {})
        dur = float(vm.get("duration_sec") or 0.0)
        segs = vm.get("segments") or []
        if len(segs) >= 2:
            # Multi-window clips become independent timeline entries downstream
            total = 0.0
            for s in segs:
                w = float(s.get("end_sec", 0.0)) - float(s.get("start_sec", 0.0))
                total += max(_min_shot, min(_max_shot, w))
            return total
        ts, te = vm.get("trim_start_sec"), vm.get("trim_end_sec")
        usable = (
            float(te) - float(ts)
            if (ts is not None and te is not None and float(te) > float(ts))
            else dur
        )
        if usable <= 0.0:
            usable = avg_shot_len
        return max(_min_shot, min(_max_shot, usable))

    _costs = [_planned_cost(i) for i in range(len(transcriptions))]

    if clip_vecs is None or segment_vecs is None:
        print("⚠️ [SCORE] Embeddings unavailable — falling back to visual score + keyword matching (no semantic story matching)")
        # No embeddings: use visual score + lexical as fallback.
        # Selection is still budget-driven: take ranked clips until their
        # planned screen time fills the duration budget.
        ranked = sorted(
            range(len(texts)),
            key=lambda i: (
                visual_meta.get(transcriptions[i].get("path",""), {}).get("visual_score", 0.5)
                + _lexical_overlap(texts[i], story_keywords)
                + 0.15 * float(qualities[i])
            ),
            reverse=True,
        )
        picked: List[int] = []
        _spent = 0.0
        _min_n = min(4, len(ranked))
        for i in ranked:
            if _spent >= planning_budget and len(picked) >= _min_n:
                break
            picked.append(i)
            _spent += _costs[i]
        return [
            {
                "path": transcriptions[i]["path"],
                **visual_meta.get(transcriptions[i].get("path", ""), _VISUAL_DEFAULT),
            }
            for i in picked
            if transcriptions[i].get("path")
        ]

    seg_sims = np.zeros((len(texts), len(story_segments)), dtype=np.float32)
    for i in range(len(texts)):
        for j in range(len(story_segments)):
            seg_sims[i, j] = _cosine_sim(clip_vecs[i], segment_vecs[j])

    semantic_max = seg_sims.max(axis=1)
    semantic_mean = seg_sims.mean(axis=1)

    rel_scores = np.zeros(len(texts), dtype=np.float32)
    best_segment = seg_sims.argmax(axis=1)

    # ── Sensory signal normalization ──────────────────────────────────────
    # motion / audio_energy live on arbitrary scales; min-max normalize across
    # the batch so they contribute comparably. Unknown values sit at neutral.
    def _norm_signal(key: str) -> List[float]:
        vals = [visual_meta.get(t.get("path", ""), {}).get(key) for t in transcriptions]
        known = [float(v) for v in vals if v is not None]
        if len(known) < 2 or (max(known) - min(known)) < 1e-6:
            return [0.5] * len(vals)
        lo, hi = min(known), max(known)
        return [((float(v) - lo) / (hi - lo)) if v is not None else 0.5 for v in vals]

    _motion_n = _norm_signal("motion")
    _audio_n = _norm_signal("audio_energy")

    for i, t in enumerate(transcriptions):
        path_i = t.get("path", "")
        text_lower = (t.get("text", "") or "").lower()
        has_transcript = len(text_lower.strip()) > 20
        lexical = _lexical_overlap(text_lower, story_keywords)
        pri_bonus = sum(1.0 for k in pri if k.lower() in text_lower) * 0.08
        exc_penalty = sum(1.0 for k in exc if k.lower() in text_lower) * 0.12
        quality_penalty = (1.0 - float(qualities[i])) * 0.15
        vm_i = visual_meta.get(path_i, {})
        # Quality and relevance as separate terms — the old blended visual_score
        # double-counted story relevance on top of the embedding similarity.
        vq = float(vm_i.get("visual_quality") or vm_i.get("visual_score") or 0.5)
        sr = float(vm_i.get("story_relevance") or vm_i.get("visual_score") or 0.5)
        face = float(vm_i.get("face_ratio") or 0.0)

        if has_transcript:
            # Hybrid: semantic + vision + lexical + sensory
            rel_scores[i] = (
                0.30 * float(semantic_max[i])
                + 0.08 * float(semantic_mean[i])
                + 0.10 * float(lexical)
                + 0.16 * sr                    # vision's read on story fit
                + 0.14 * vq                    # pure aesthetics
                + 0.09 * float(_motion_n[i])   # visual energy
                + 0.05 * float(_audio_n[i])    # clip's own audio energy
                + 0.08 * face                  # human interest
                + pri_bonus
                - exc_penalty
                - quality_penalty
            )
        else:
            # No transcript (b-roll, music video): sensory + vision carry it
            rel_scores[i] = (
                0.28 * vq
                + 0.24 * sr
                + 0.20 * float(_motion_n[i])
                + 0.10 * float(_audio_n[i])
                + 0.12 * face
                + 0.06 * float(qualities[i])
                + pri_bonus
                - exc_penalty
            )

    candidate_indices = sorted(valid_idx, key=lambda i: float(rel_scores[i]), reverse=True)
    if not candidate_indices:
        candidate_indices = list(range(len(texts)))

    top_pool = list(candidate_indices)

    # Provide shot_types so MMR penalises visual repetition (-0.10 per extra duplicate).
    _pool_shot_types = [
        visual_meta.get(transcriptions[i].get("path", ""), {}).get("shot_type") or "unknown"
        for i in range(len(transcriptions))
    ]
    # Visual signatures + empty-transcript mask: silent clips all embed the
    # same, so MMR measures their redundancy on colour histograms instead.
    _pool_vis_sigs = [
        visual_meta.get(transcriptions[i].get("path", ""), {}).get("vis_sig")
        for i in range(len(transcriptions))
    ]
    _pool_text_empty = [len((t.get("text") or "").strip()) <= 20 for t in transcriptions]
    # Greedy knapsack: MMR order, stop when planned screen time fills the budget.
    chosen = _mmr_select(
        top_pool, rel_scores, clip_vecs,
        k=len(top_pool),
        shot_types=_pool_shot_types,
        costs=_costs,
        budget_sec=planning_budget,
        min_count=min(4, len(top_pool)),
        vis_sigs=_pool_vis_sigs,
        text_empty=_pool_text_empty,
    )

    # Enforce narrative beat coverage first (human-editor style assembly).
    beat_cover = []
    used = set()
    for beat_idx in range(len(story_segments)):
        ranked_for_beat = sorted(
            top_pool,
            key=lambda i: (float(seg_sims[i, beat_idx]) + 0.25 * float(rel_scores[i])),
            reverse=True,
        )
        for idx in ranked_for_beat:
            if idx not in used and qualities[idx] >= 0.15:
                beat_cover.append(idx)
                used.add(idx)
                break

    # Beat-covering clips get priority, then MMR picks fill the remaining
    # duration budget. Cap by planned screen time, not clip count.
    ordered_full = beat_cover + [idx for idx in chosen if idx not in used]
    ordered = []
    _spent = 0.0
    _min_n = min(4, len(ordered_full))
    for idx in ordered_full:
        if _spent >= planning_budget and len(ordered) >= _min_n:
            break
        ordered.append(idx)
        _spent += _costs[idx]

    if not ordered:
        ordered = sorted(chosen, key=lambda i: (int(best_segment[i]), -float(rel_scores[i])))

    # Single structured editorial pass: GPT-4o sees durations + budget, assigns
    # arc slots (hook → build → climax → resolution), orders, and may drop.
    # Hook/payoff placement and shot variety are constraints in this prompt;
    # the old rerank → continuity-check → pin chain overwrote its own work.
    _arc_result = _llm_arc_pass(
        story=story_text,
        transcriptions=transcriptions,
        candidate_indices=ordered,
        api_key=api_key,
        tone=tone,
        visual_meta=visual_meta,
        costs=_costs,
        budget_sec=planning_budget,
        rel_scores=rel_scores,
        avg_shot_len=avg_shot_len,
    )
    _arc_applied = bool(_arc_result)
    _arc_weights: Dict[int, float] = {}
    _arc_transitions: Dict[int, str] = {}
    if _arc_result:
        ordered, _arc_weights, _arc_transitions = _arc_result

    # Build enriched result list (dicts with path + visual metadata)
    result: List[Dict] = []
    for i in ordered:
        t = transcriptions[i]
        path = t.get("path", "")
        if not path:
            continue
        vm = visual_meta.get(path, {})
        # Use GPT-4o narrative_role when available; otherwise fall back to
        # which story beat this clip best matched.
        role = vm.get("narrative_role") or "development"
        result.append({
            "path":            path,
            "narrative_role":  role,
            "shot_type":       vm.get("shot_type", "unknown"),
            "emotion":         vm.get("emotion", "neutral"),
            "visual_score":    vm.get("visual_score", 0.5),
            # The arc pass's own editorial judgment of relative screen time
            # for THIS clip in THIS edit — None when the arc pass didn't run,
            # so editor.py falls back to its fixed role/emotion weight table.
            "arc_weight":      _arc_weights.get(i),
            # The arc pass's chosen transition effect for the cut INTO this
            # clip (None for the opener, or when the arc pass didn't run) —
            # editor.py falls back to its heuristic transition picker.
            "arc_transition":  _arc_transitions.get(i),
            "description":     vm.get("description", ""),
            # Real source duration (probed during visual analysis)
            "duration_sec":    vm.get("duration_sec"),
            # Visual scene fingerprint — used to break same-scene adjacency
            "vis_sig":         vm.get("vis_sig"),
            # Vision-guided trim: the model's grounded cut recommendations
            "best_moment_sec": vm.get("best_moment_sec"),
            "emotional_peak_sec": vm.get("emotional_peak_sec"),
            "trim_start_sec":  vm.get("trim_start_sec"),
            "trim_end_sec":    vm.get("trim_end_sec"),
            # Multi-window segments for long clips (used by generate_video expansion)
            "segments":        vm.get("segments") or [],
        })

    if result:
        # ── shot continuity repair ────────────────────────────────────────
        # Build the full pool (all transcriptions with metadata) for the fixer
        full_pool = [
            {
                "path": t["path"],
                **visual_meta.get(t.get("path", ""), _VISUAL_DEFAULT),
            }
            for t in transcriptions
            if t.get("path")
        ]

        def _cost_of_clip(d: Dict) -> float:
            """Same planned-cost model as _planned_cost, but for clip dicts."""
            dur = float(d.get("duration_sec") or 0.0)
            segs = d.get("segments") or []
            if len(segs) >= 2:
                return sum(
                    max(_min_shot, min(_max_shot,
                        float(s.get("end_sec", 0.0)) - float(s.get("start_sec", 0.0))))
                    for s in segs
                )
            ts, te = d.get("trim_start_sec"), d.get("trim_end_sec")
            usable = (
                float(te) - float(ts)
                if (ts is not None and te is not None and float(te) > float(ts))
                else dur
            )
            if usable <= 0.0:
                usable = avg_shot_len
            return max(_min_shot, min(_max_shot, usable))

        result = _fix_shot_continuity(
            result, full_pool, max_fixes=4,
            cost_of=_cost_of_clip, budget_sec=planning_budget,
            protect_ends=_arc_applied,
        )
        # Hook/payoff pinning only when the arc pass didn't run — the arc
        # prompt already places the opener and closer deliberately, and blind
        # post-swaps were undoing it.
        if not _arc_applied:
            result = _pin_hook_and_payoff(result)
        # Break up back-to-back near-identical scenes (colour-histogram match)
        result = _reduce_adjacent_similarity(result)
        return result
    # Final fallback: original order with default metadata
    return [
        {"path": t["path"], "narrative_role": "development",
         "shot_type": "unknown", "emotion": "neutral", "visual_score": 0.5, "description": ""}
        for t in transcriptions if t.get("path")
    ]
