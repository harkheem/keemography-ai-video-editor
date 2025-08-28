from sentence_transformers import SentenceTransformer, util
import torch

_model = None
def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def score_clips_with_story(transcriptions, story, priority_keywords=None, exclude_keywords=None, tone=None):
    model = _get_model()
    story_embedding = model.encode(story, convert_to_tensor=True, normalize_embeddings=True)
    results = []

    priority_keywords = set((priority_keywords or []))
    exclude_keywords = set((exclude_keywords or []))

    for item in transcriptions:
        text = (item.get("text") or "").lower()
        if not text.strip():
            continue
        clip_embedding = model.encode(text, convert_to_tensor=True, normalize_embeddings=True)
        base_score = util.cos_sim(clip_embedding, story_embedding).item()

        bonus = sum(1.0 for k in priority_keywords if k.lower() in text) * 0.05
        penalty = sum(1.0 for k in exclude_keywords if k.lower() in text) * 0.10

        final_score = base_score + bonus - penalty
        results.append((item["path"], final_score))

    results.sort(key=lambda x: x[1], reverse=True)
    # keep top 6 or all if fewer
    return [p for p, _ in results[:6]] if results else [t["path"] for t in transcriptions][:6]
