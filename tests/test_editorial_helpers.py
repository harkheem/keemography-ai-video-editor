from scoring import (
    _pin_hook_and_payoff,
    _reduce_adjacent_similarity,
    _text_quality,
    _lexical_overlap,
    _auto_keywords,
    _tokenize,
)


def _clip(path, role="development", emotion="neutral", shot="medium", visual_score=0.5, vis_sig=None):
    return {
        "path": path, "narrative_role": role, "emotion": emotion,
        "shot_type": shot, "visual_score": visual_score, "vis_sig": vis_sig,
    }


def test_pin_hook_and_payoff_moves_best_hook_to_opening():
    clips = [
        _clip("weak_open", role="development", emotion="neutral", shot="medium"),
        _clip("real_hook", role="hook", emotion="exciting", shot="action"),
        _clip("middle", role="development"),
    ]
    result = _pin_hook_and_payoff(clips)
    assert result[0]["path"] == "real_hook"


def test_pin_hook_and_payoff_moves_best_payoff_to_closing():
    clips = [
        _clip("hook", role="hook", emotion="exciting", shot="action"),
        _clip("real_payoff", role="payoff", emotion="inspiring", shot="close_up"),
        _clip("weak_close", role="development"),
    ]
    result = _pin_hook_and_payoff(clips)
    assert result[-1]["path"] == "real_payoff"


def test_pin_hook_and_payoff_noop_under_three_clips():
    clips = [_clip("a"), _clip("b")]
    result = _pin_hook_and_payoff(clips)
    assert [c["path"] for c in result] == ["a", "b"]


def test_reduce_adjacent_similarity_breaks_up_same_scene_run():
    clips = [
        _clip("opener", vis_sig=[1, 0, 0]),
        _clip("dup_a", vis_sig=[1, 0, 0]),      # same scene as dup_b
        _clip("dup_b", vis_sig=[1, 0, 0]),      # would sit right next to dup_a
        _clip("different", vis_sig=[0, 1, 0]),
        _clip("closer", vis_sig=[0, 0, 1]),
    ]
    result = _reduce_adjacent_similarity(clips, sim_threshold=0.92)
    paths = [c["path"] for c in result]
    # opener/closer stay pinned in place
    assert paths[0] == "opener"
    assert paths[-1] == "closer"
    # dup_a and dup_b should no longer be adjacent
    idx_a, idx_b = paths.index("dup_a"), paths.index("dup_b")
    assert abs(idx_a - idx_b) != 1


def test_reduce_adjacent_similarity_leaves_diverse_sequence_untouched():
    clips = [
        _clip("a", vis_sig=[1, 0, 0]),
        _clip("b", vis_sig=[0, 1, 0]),
        _clip("c", vis_sig=[0, 0, 1]),
        _clip("d", vis_sig=[1, 1, 0]),
    ]
    result = _reduce_adjacent_similarity(clips)
    assert [c["path"] for c in result] == ["a", "b", "c", "d"]


def test_tokenize_lowercases_and_splits_on_non_alnum():
    assert _tokenize("Hello, World! It's 2024.") == ["hello", "world", "it's", "2024"]


def test_text_quality_empty_string_is_zero():
    assert _text_quality("") == 0.0
    assert _text_quality("   ") == 0.0


def test_text_quality_longer_richer_text_scores_higher():
    short = _text_quality("hi there")
    long_rich = _text_quality(
        "We walked along the beach at sunset talking about everything and nothing at all"
    )
    assert long_rich > short


def test_lexical_overlap_counts_shared_keywords():
    keywords = {"wedding", "beach", "sunset"}
    assert _lexical_overlap("a beautiful wedding at sunset", keywords) == 2 / 3
    assert _lexical_overlap("completely unrelated text here", keywords) == 0.0
    assert _lexical_overlap("anything", set()) == 0.0


def test_auto_keywords_picks_frequent_meaningful_words():
    story = "wedding wedding wedding beach beach the a is are"
    keywords = _auto_keywords(story)
    assert "wedding" in keywords
    assert "beach" in keywords
    assert "the" not in keywords  # stopword
