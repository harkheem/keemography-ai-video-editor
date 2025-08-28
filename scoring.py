from typing import List, Dict, Optional, Iterable
import os
import numpy as np
from openai import OpenAI

# Simple cosine similarity
def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(a @ b / denom)

def _get_api_key(explicit_key: Optional[str] = None) -> Optional[str]:
    if explicit_key:
        return explicit_key
    return os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")

def _embed_texts(client: OpenAI, texts: Iterable[str]) -> List[List[float]]:
    # text-embedding-3-small is inexpensive & solid
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=list(texts),
    )
    return [d.embedding for d in resp.data]

def score_clips_with_story(
    transcriptions: List[Dict[str, str]],
    story: str,
    priority_keywords: Optional[List[str]] = None,
    exclude_keywords: Optional[List[str]] = None,
    tone: Optional[str] = None,
    api_key: Optional[str] = None,
    top_k: int = 6,
) -> List[str]:
    """
    Ranks clips by semantic similarity to the story using OpenAI embeddings.
    Adds small bonuses/penalties for included/excluded keywords.
    Returns: top_k clip PATHS (strings), sorted best-first.
    """
    key = _get_api_key(api_key)
    if not key:
        raise RuntimeError(
            "Missing OpenAI API key. Set API_KEY or OPENAI_API_KEY in env/secrets."
        )

    client = OpenAI(api_key=key)

    # Prepare text list (avoid None)
    clip_texts = [(item.get("path", ""), (item.get("text") or "")) for item in transcriptions]
    paths, texts = zip(*clip_texts) if clip_texts else ([], [])

    if not texts:
        # If we have no text at all, just return the first few paths
        return list(paths)[:top_k]

    # Embed story and all clip texts (batch in one request)
    story_vec = np.array(_embed_texts(client, [story])[0], dtype=np.float32)
    clip_vecs = [np.array(v, dtype=np.float32) for v in _embed_texts(client, texts)]

    pset = {k.strip().lower() for k in (priority_keywords or []) if k.strip()}
    eset = {k.strip().lower() for k in (exclude_keywords or []) if k.strip()}

    scored = []
    for (path, text), vec in zip(clip_texts, clip_vecs):
        base = _cosine(vec, story_vec)
        t = (text or "").lower()

        bonus = sum(1.0 for k in pset if k in t) * 0.05
        penalty = sum(1.0 for k in eset if k in t) * 0.10

        scored.append((path, base + bonus - penalty))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [p for p, _ in scored[:top_k]]
