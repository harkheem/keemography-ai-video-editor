
# scoring.py
from typing import List, Dict, Optional, Sequence, Tuple
import os
import re
import json
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
    except Exception:
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
    "narrative_role": "development",
    "best_moment_sec": None,
    "trim_start_sec": None,
    "trim_end_sec": None,
}


def _describe_clip_visually(
    frames_with_ts: List[Tuple[str, float]],   # (base64, timestamp_sec)
    clip_duration: float,
    api_key: str,
    story: str,
    tone: str,
    story_beats: Optional[List[str]] = None,
) -> Dict:
    """
    Send timestamped frames to GPT-4o vision.
    Returns shot analysis PLUS vision-guided trim recommendation:
      best_moment_sec  — single most cinematic/relevant moment
      trim_start_sec   — recommended trim window start
      trim_end_sec     — recommended trim window end
    """
    if not frames_with_ts or not api_key:
        return dict(_VISUAL_DEFAULT)
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
                    },
                }),
            }
        ]
        for b64, _ts in frames_with_ts[:5]:
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
            max_tokens=280,
            temperature=0.0,
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
        ts_start = _safe_float("trim_start_sec")
        ts_end   = _safe_float("trim_end_sec")

        # Clamp to valid range
        if bm is not None:
            bm = max(0.0, min(clip_duration, bm))
        if ts_start is not None and ts_end is not None:
            ts_start = max(0.0, ts_start)
            ts_end   = min(clip_duration, ts_end)
            # Reject if the window is too small or inverted
            if ts_end - ts_start < 1.0:
                ts_start = ts_end = None
        else:
            ts_start = ts_end = None

        return {
            "description":    str(parsed.get("description", "")),
            "shot_type":      str(parsed.get("shot_type", "unknown")),
            "emotion":        str(parsed.get("emotion", "neutral")),
            "visual_score":   round(0.40 * vq + 0.60 * sr, 4),
            "narrative_role": str(parsed.get("narrative_role", "development")),
            "best_moment_sec": bm,
            "trim_start_sec":  ts_start,
            "trim_end_sec":    ts_end,
        }
    except Exception:
        return dict(_VISUAL_DEFAULT)


def _score_clips_visually(
    transcriptions: List[Dict],
    story: str,
    tone: str,
    api_key: Optional[str],
    max_workers: int = 2,  # 2 instead of 4 — each worker holds frame data in RAM
    story_beats: Optional[List[str]] = None,
) -> Dict[str, Dict]:
    """
    Run GPT-4o visual analysis concurrently for all clips.
    Returns a dict keyed by clip path.
    Now passes story_beats so GPT-4o can align trim recommendations
    to specific narrative moments.
    """
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
            return path, dict(_VISUAL_DEFAULT)
        clip_dur = _probe_duration(path)
        frames_with_ts = _sample_frames_ffmpeg(path, max_frames=6)
        result = _describe_clip_visually(
            frames_with_ts, clip_dur, api_key or "", story, tone,
            story_beats=story_beats,
        )
        return path, result

    results: Dict[str, Dict] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        for path, meta in ex.map(_analyse, transcriptions):
            if path:
                results[path] = meta
    return results


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
) -> List[Dict]:
    """
    Walk the ordered clip list and repair jarring consecutive cuts.

    Strategy:
    A) Insert a broll/neutral clip from the unused pool between the jarring pair.
    B) If no broll is available, swap the second clip in the pair for the
       highest-scored unused clip whose shot_type is compatible with the first.
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
    fixes = 0
    i = 0
    while i < len(result) - 1 and fixes < max_fixes:
        shot_a = result[i].get("shot_type", "unknown")
        shot_b = result[i + 1].get("shot_type", "unknown")

        if not _is_jarring_cut(shot_a, shot_b):
            i += 1
            continue

        # Strategy A: find an unused broll/neutral buffer clip
        buffer = None
        for idx, candidate in enumerate(unused):
            cshot = (candidate.get("shot_type") or "unknown").lower()
            if cshot in ("broll", "medium", "wide", "action", "product"):
                if not _is_jarring_cut(shot_a, cshot) and not _is_jarring_cut(cshot, shot_b):
                    buffer = unused.pop(idx)
                    break

        if buffer:
            buffer["narrative_role"] = "broll"  # mark as buffer
            result.insert(i + 1, buffer)
            fixes += 1
            i += 2  # skip the inserted clip and the next pair
            continue

        # Strategy B: swap result[i+1] for a compatible unused clip at same beat
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
            fixes += 1

        i += 1

    return result


# ---------------------------------------------------------------------------

def _split_story_into_segments(story: str) -> List[str]:
    parts = re.split(r"[\n\.!?;:]+", story or "")
    segments = [p.strip() for p in parts if p and p.strip()]
    return segments[:8] if segments else [" "]

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

def _mmr_select(indices: List[int], rel_scores: np.ndarray, embeds: np.ndarray, k: int, lam: float = 0.78) -> List[int]:
    if not indices:
        return []
    chosen = [max(indices, key=lambda i: rel_scores[i])]
    remaining = set(indices) - set(chosen)
    while remaining and len(chosen) < k:
        best_i = None
        best_score = -1e9
        for i in remaining:
            redundancy = max(_cosine_sim(embeds[i], embeds[j]) for j in chosen) if chosen else 0.0
            score = lam * float(rel_scores[i]) - (1.0 - lam) * float(redundancy)
            if score > best_score:
                best_score = score
                best_i = i
        chosen.append(best_i)
        remaining.remove(best_i)
    return chosen

def _llm_editorial_rerank(
    story: str,
    transcriptions: List[Dict[str, str]],
    candidate_indices: List[int],
    api_key: Optional[str],
    tone: Optional[str] = None,
) -> Optional[List[int]]:
    if not api_key or not candidate_indices:
        return None

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        tone_label = (tone or "").strip() or "cinematic"

        payload = []
        for idx in candidate_indices:
            text = str(transcriptions[idx].get("text", "") or "").strip()
            text = re.sub(r"\s+", " ", text)
            payload.append({
                "id": idx,
                "transcript": text[:280],
            })

        prompt = {
            "task": "Order candidate clips into the best narrative sequence for a short edit.",
            "constraints": [
                "Prefer coherent story flow: hook -> development -> payoff.",
                "Avoid clips with gibberish/low-information text unless necessary.",
                "Prioritize emotional and semantic relevance to the story.",
                "Return ONLY JSON with key 'order' as list of clip ids.",
            ],
            "tone": tone_label,
            "story": story,
            "candidates": payload,
        }

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.0,
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional video editor. Output strict JSON only.",
                },
                {
                    "role": "user",
                    "content": json.dumps(prompt, ensure_ascii=False),
                },
            ],
            max_tokens=220,
        )

        content = (response.choices[0].message.content or "").strip()
        parsed = json.loads(content)
        raw_order = parsed.get("order", []) if isinstance(parsed, dict) else []

        seen = set()
        cleaned = []
        allowed = set(candidate_indices)
        for item in raw_order:
            try:
                clip_id = int(item)
            except Exception:
                continue
            if clip_id in allowed and clip_id not in seen:
                seen.add(clip_id)
                cleaned.append(clip_id)

        for idx in candidate_indices:
            if idx not in seen:
                cleaned.append(idx)

        return cleaned if cleaned else None
    except Exception:
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
            model="gpt-4o-mini",
            temperature=0.1,
            messages=[
                {"role": "system", "content": "You are a senior narrative video editor. Output strict JSON only."},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            max_tokens=180,
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

def score_clips_with_story(
    transcriptions: List[Dict[str, str]],
    story: str,
    priority_keywords: Optional[List[str]] = None,
    exclude_keywords: Optional[List[str]] = None,
    tone: Optional[str] = None,
    target_duration_sec: Optional[int] = None,
    openai_api_key: Optional[str] = None,
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
        return [{"path": t["path"], "narrative_role": "development",
                 "shot_type": "unknown", "emotion": "neutral", "visual_score": 0.5}
                for t in transcriptions if t.get("path")]

    texts = [str(t.get("text", "") or "") for t in transcriptions]

    story_text = str(story or "").strip() or " "

    # ── Step 1: story plan first so visual analysis can align to beats ────
    # We run planning synchronously — it's fast (gpt-4o-mini) and short.
    # The beats go into every per-clip GPT-4o prompt as context so the model
    # knows which narrative moment each clip should serve.
    story_segments = _llm_story_plan(story_text, api_key, tone) or _split_story_into_segments(story_text)

    # ── Step 2: visual analysis with story beats as context ───────────────
    visual_meta: Dict[str, Dict] = _score_clips_visually(
        transcriptions, story_text, tone or "", api_key,
        story_beats=story_segments,
    )
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
    shot_len_by_tone = {
        "energetic": 2.6,
        "epic": 3.2,
        "cinematic": 3.8,
        "sentimental": 4.2,
        "calm": 4.8,
    }
    avg_shot_len = shot_len_by_tone.get(tone_key, 3.8)
    desired_duration = float(target_duration_sec) if target_duration_sec else 45.0
    desired_clip_count = int(round(desired_duration / max(1.8, avg_shot_len)))
    target_k = max(4, min(len(texts), desired_clip_count))

    if clip_vecs is None or segment_vecs is None:
        # No embeddings: use visual score + lexical as fallback
        ranked = sorted(
            range(len(texts)),
            key=lambda i: (
                visual_meta.get(transcriptions[i].get("path",""), {}).get("visual_score", 0.5)
                + _lexical_overlap(texts[i], story_keywords)
                + 0.15 * float(qualities[i])
            ),
            reverse=True,
        )
        top_k = max(4, min(len(ranked), target_k))
        return [
            {
                "path": transcriptions[i]["path"],
                **visual_meta.get(transcriptions[i].get("path", ""), _VISUAL_DEFAULT),
            }
            for i in ranked[:top_k]
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
    for i, t in enumerate(transcriptions):
        path_i = t.get("path", "")
        text_lower = (t.get("text", "") or "").lower()
        has_transcript = len(text_lower.strip()) > 20
        lexical = _lexical_overlap(text_lower, story_keywords)
        pri_bonus = sum(1.0 for k in pri if k.lower() in text_lower) * 0.08
        exc_penalty = sum(1.0 for k in exc if k.lower() in text_lower) * 0.12
        quality_penalty = (1.0 - float(qualities[i])) * 0.15
        vis = float(visual_meta.get(path_i, {}).get("visual_score", 0.5))

        if has_transcript:
            # Hybrid: semantic + visual + lexical
            rel_scores[i] = (
                0.38 * float(semantic_max[i])
                + 0.12 * float(semantic_mean[i])
                + 0.12 * float(lexical)
                + 0.30 * vis            # visual carries real weight
                + pri_bonus
                - exc_penalty
                - quality_penalty
            )
        else:
            # No transcript (b-roll, music video): visual is primary signal
            rel_scores[i] = (
                0.75 * vis
                + 0.15 * float(qualities[i])
                + pri_bonus
                - exc_penalty
            )

    candidate_indices = sorted(valid_idx, key=lambda i: float(rel_scores[i]), reverse=True)
    if not candidate_indices:
        candidate_indices = list(range(len(texts)))

    top_pool = list(candidate_indices)

    target_k = max(4, min(len(candidate_indices), desired_clip_count))
    chosen = _mmr_select(top_pool, rel_scores, clip_vecs, target_k)

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

    ordered = beat_cover + [idx for idx in chosen if idx not in used]
    ordered = ordered[:target_k]

    if not ordered:
        ordered = sorted(chosen, key=lambda i: (int(best_segment[i]), -float(rel_scores[i])))

    # Final editorial pass: LLM reorders shortlisted clips for narrative flow.
    llm_ordered = _llm_editorial_rerank(
        story=story_text,
        transcriptions=transcriptions,
        candidate_indices=ordered,
        api_key=api_key,
        tone=tone,
    )
    if llm_ordered:
        ordered = llm_ordered

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
            "path": path,
            "narrative_role": role,
            "shot_type":       vm.get("shot_type", "unknown"),
            "emotion":         vm.get("emotion", "neutral"),
            "visual_score":    vm.get("visual_score", 0.5),
            "description":     vm.get("description", ""),
            # Vision-guided trim: GPT-4o's direct recommendation for where to cut
            "best_moment_sec": vm.get("best_moment_sec"),
            "trim_start_sec":  vm.get("trim_start_sec"),
            "trim_end_sec":    vm.get("trim_end_sec"),
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
        result = _fix_shot_continuity(result, full_pool, max_fixes=4)
        return result
    # Final fallback: original order with default metadata
    return [
        {"path": t["path"], "narrative_role": "development",
         "shot_type": "unknown", "emotion": "neutral", "visual_score": 0.5, "description": ""}
        for t in transcriptions if t.get("path")
    ]
