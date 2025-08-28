# scoring.py
from typing import List, Dict, Optional, Sequence
import os
import numpy as np

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
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=list(texts),
    )
    # shape: (n, 1536)
    vecs = [np.array(d.embedding, dtype=np.float32) for d in resp.data]
    return np.vstack(vecs)

def score_clips_with_story(
    transcriptions: List[Dict[str, str]],
    story: str,
    priority_keywords: Optional[List[str]] = None,
    exclude_keywords: Optional[List[str]] = None,
    tone: Optional[str] = None,
    openai_api_key: Optional[str] = None,
) -> List[str]:
    """
    Returns top clip paths ranked by semantic similarity to `story`,
    with small bonuses/penalties for keyword matches.
    """
    api_key = _get_api_key(openai_api_key)
    if not api_key:
        # If no key, just return paths in original order
        return [t["path"] for t in transcriptions]

    texts = [t.get("text", "") or "" for t in transcriptions]
    if not any(texts):
        return [t["path"] for t in transcriptions]

    story_vec = _embed_texts([story], api_key)[0]
    clip_vecs = _embed_texts(texts, api_key)

    pri = set((priority_keywords or []))
    exc = set((exclude_keywords or []))

    scored = []
    for i, t in enumerate(transcriptions):
        text_lower = (t.get("text", "") or "").lower()
        base = _cosine_sim(clip_vecs[i], story_vec)
        bonus = sum(1.0 for k in pri if k.lower() in text_lower) * 0.05
        penalty = sum(1.0 for k in exc if k.lower() in text_lower) * 0.10
        scored.append((t["path"], base + bonus - penalty))

    scored.sort(key=lambda x: x[1], reverse=True)
    # Cap to 6 like before (optional)
    return [p for p, _ in scored[:6]] or [t["path"] for t in transcriptions]
