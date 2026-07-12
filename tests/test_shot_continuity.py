from scoring import _is_jarring_cut, _fix_shot_continuity


def test_same_close_up_is_jarring():
    assert _is_jarring_cut("close_up", "close_up") is True


def test_extreme_wide_to_close_up_is_jarring():
    assert _is_jarring_cut("extreme_wide", "close_up") is True
    assert _is_jarring_cut("close_up", "extreme_wide") is True  # symmetric


def test_extreme_wide_to_talking_head_is_jarring_via_rank_gap():
    # Not an explicit pair, but rank(extreme_wide)=0 vs rank(talking_head)=4 -> gap 4 >= 3
    assert _is_jarring_cut("extreme_wide", "talking_head") is True


def test_adjacent_ranks_not_jarring():
    assert _is_jarring_cut("wide", "medium") is False
    assert _is_jarring_cut("medium", "talking_head") is False


def test_unknown_shots_default_to_non_jarring():
    assert _is_jarring_cut("unknown", "unknown") is False


def _clip(path, shot, role="development"):
    return {"path": path, "shot_type": shot, "narrative_role": role, "visual_score": 0.6}


def test_inserts_broll_buffer_between_jarring_pair():
    # landscape(rank1) -> close_up(rank5) is jarring (gap 4); "medium"(rank3) sits
    # close enough to both ends to bridge it without introducing a new jarring cut.
    ordered = [_clip("a", "landscape"), _clip("b", "close_up")]
    pool = [_clip("a", "landscape"), _clip("b", "close_up"), _clip("buffer", "medium")]

    result = _fix_shot_continuity(ordered, pool, cost_of=lambda d: 3.0, budget_sec=100.0)

    assert [c["path"] for c in result] == ["a", "buffer", "b"]


def test_no_insert_when_budget_already_spent():
    ordered = [_clip("a", "landscape"), _clip("b", "close_up")]
    pool = [_clip("a", "landscape"), _clip("b", "close_up"), _clip("buffer", "medium"),
            _clip("swap_candidate", "medium")]

    # planned_total (from cost_of over `ordered`) already >= budget_sec, so
    # strategy A (insert) is skipped entirely and strategy B (swap) should fire.
    result = _fix_shot_continuity(
        ordered, pool, cost_of=lambda d: 5.0, budget_sec=5.0,
    )

    assert len(result) == 2  # no insert happened
    assert result[1]["path"] in ("buffer", "swap_candidate")  # b got swapped for one of them


def test_protect_ends_prevents_swapping_the_closer():
    # (a, b) is clean; (b, c) is jarring and sits at the very end.
    ordered = [_clip("a", "landscape"), _clip("b", "wide"), _clip("c", "close_up")]
    pool = [*ordered, _clip("swap_candidate", "wide")]

    # Budget fully spent so strategy A (insert) never fires, forcing the jarring
    # end pair through strategy B, where protect_ends should refuse to touch c.
    result = _fix_shot_continuity(
        ordered, pool, cost_of=lambda d: 1.0, budget_sec=0.0, protect_ends=True,
    )

    assert result[-1]["path"] == "c"


def test_leaves_clean_sequence_untouched():
    ordered = [_clip("a", "wide"), _clip("b", "medium"), _clip("c", "close_up")]
    result = _fix_shot_continuity(ordered, ordered, cost_of=lambda d: 1.0, budget_sec=10.0)
    assert [c["path"] for c in result] == ["a", "b", "c"]
