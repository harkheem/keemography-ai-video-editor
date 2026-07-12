import numpy as np

from scoring import _mmr_select


def _orthogonal_embeds(n: int, dim: int = 4) -> np.ndarray:
    """n mutually-orthogonal unit vectors -> zero cosine similarity between any pair."""
    eye = np.eye(dim, dtype=np.float32)
    return eye[:n]


def test_always_picks_highest_relevance_first():
    embeds = _orthogonal_embeds(3)
    rel = np.array([0.9, 0.5, 0.2], dtype=np.float32)
    chosen = _mmr_select([0, 1, 2], rel, embeds, k=3)
    assert chosen[0] == 0


def test_respects_k_upper_bound():
    embeds = _orthogonal_embeds(4)
    rel = np.array([0.9, 0.8, 0.7, 0.6], dtype=np.float32)
    chosen = _mmr_select([0, 1, 2, 3], rel, embeds, k=2)
    assert len(chosen) == 2


def test_stops_at_budget_once_min_count_reached():
    embeds = _orthogonal_embeds(4)
    rel = np.array([0.9, 0.8, 0.7, 0.6], dtype=np.float32)
    costs = [3.0, 3.0, 3.0, 3.0]
    chosen = _mmr_select(
        [0, 1, 2, 3], rel, embeds, k=4, costs=costs, budget_sec=5.0, min_count=2,
    )
    # First pick costs 3.0 (< budget), still under min_count=2 so a second pick
    # happens (now spent=6.0 >= budget=5.0 AND len(chosen)=2 >= min_count) -> stop.
    assert len(chosen) == 2


def test_never_selects_fewer_than_min_count_even_over_budget():
    embeds = _orthogonal_embeds(4)
    rel = np.array([0.9, 0.8, 0.7, 0.6], dtype=np.float32)
    costs = [10.0, 10.0, 10.0, 10.0]
    chosen = _mmr_select(
        [0, 1, 2, 3], rel, embeds, k=4, costs=costs, budget_sec=1.0, min_count=3,
    )
    assert len(chosen) == 3


def test_shot_type_diversity_penalty_prefers_variety():
    # Two clips are near-duplicates of the top pick (embeds identical -> max
    # redundancy) but have different shot types; a third, more different clip
    # shares the top pick's shot type. With a high enough relevance edge the
    # diversity penalty should still let variety win once a duplicate shot type
    # has already been used.
    dim = 4
    embeds = np.zeros((4, dim), dtype=np.float32)
    embeds[0] = [1, 0, 0, 0]
    embeds[1] = [1, 0, 0, 0]   # identical to 0 -> redundancy 1.0 with the top pick
    embeds[2] = [1, 0, 0, 0]   # identical to 0 -> redundancy 1.0 with the top pick
    embeds[3] = [0, 1, 0, 0]   # orthogonal -> redundancy 0.0 with the top pick

    rel = np.array([0.90, 0.70, 0.70, 0.65], dtype=np.float32)
    shot_types = ["wide", "wide", "close_up", "wide"]

    chosen = _mmr_select(
        [0, 1, 2, 3], rel, embeds, k=2, lam=0.78, shot_types=shot_types,
    )
    assert chosen[0] == 0
    # candidate 1 ("wide", dup of chosen[0]'s shot type, redundancy 1.0):
    #   score = 0.78*0.70 - 0.22*1.0 = 0.324
    # candidate 2 ("close_up", diff shot type, redundancy 1.0 too since same embed):
    #   score = 0.78*0.70 - 0.22*1.0 = 0.324 (no penalty yet, first occurrence)
    # candidate 3 ("wide", orthogonal, redundancy 0.0, but shot type dup +1 occurrence
    #   only counted for shot_types seen in `selected_shots`, none yet other than
    #   pick 0's "wide" -> already 1 occurrence of "wide" -> penalty applies):
    #   score = 0.78*0.65 - 0.22*0.0 - 0.18*max(0, 1-1) = 0.507
    assert chosen[1] == 3


def test_redundancy_falls_back_to_visual_signature_for_silent_clips():
    # Both clips have empty transcripts (text_empty=True); their text embeddings
    # are identical placeholders (would normally look maximally redundant), but
    # their vis_sigs are orthogonal -> should NOT be penalized as redundant.
    dim = 4
    embeds = np.zeros((2, dim), dtype=np.float32)
    embeds[0] = [1, 0, 0, 0]
    embeds[1] = [1, 0, 0, 0]  # identical text embedding (both silent clips)
    rel = np.array([0.9, 0.8], dtype=np.float32)
    vis_sigs = [[1, 0, 0], [0, 1, 0]]  # orthogonal visual signatures
    text_empty = [True, True]

    chosen = _mmr_select(
        [0, 1], rel, embeds, k=2, vis_sigs=vis_sigs, text_empty=text_empty,
    )
    assert set(chosen) == {0, 1}
