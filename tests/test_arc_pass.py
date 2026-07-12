import json
from unittest.mock import MagicMock, patch

import numpy as np

from scoring import _llm_arc_pass

TRANSCRIPTIONS = [
    {"path": "/tmp/a.mp4", "text": "walking in nervous"},
    {"path": "/tmp/b.mp4", "text": "crowd cheering"},
    {"path": "/tmp/c.mp4", "text": "quiet hallway"},
    {"path": "/tmp/d.mp4", "text": "romantic kiss"},
    {"path": "/tmp/e.mp4", "text": "another quiet moment"},
    {"path": "/tmp/f.mp4", "text": "closing shot"},
]
VISUAL_META = {
    "/tmp/a.mp4": {"narrative_role": "hook", "shot_type": "wide", "emotion": "tense"},
    "/tmp/b.mp4": {"narrative_role": "development", "shot_type": "action", "emotion": "exciting"},
    "/tmp/c.mp4": {"narrative_role": "broll", "shot_type": "wide", "emotion": "neutral"},
    "/tmp/d.mp4": {"narrative_role": "payoff", "shot_type": "close_up", "emotion": "happy"},
    "/tmp/e.mp4": {"narrative_role": "development", "shot_type": "medium", "emotion": "calm"},
    "/tmp/f.mp4": {"narrative_role": "payoff", "shot_type": "wide", "emotion": "inspiring"},
}
# 6 candidates by default so tests can exercise dropping/trimming behavior
# without immediately colliding with the budget-aware min_keep floor (which
# is at least 4 — see _llm_arc_pass's avg_shot_len-driven calculation).
COSTS = [3.0, 3.0, 3.0, 3.0, 3.0, 3.0]
REL_SCORES = np.array([0.7, 0.8, 0.4, 0.9, 0.5, 0.3])


def _mock_openai_response(content: dict):
    fake_client = MagicMock()
    fake_response = MagicMock()
    fake_response.choices = [MagicMock(message=MagicMock(content=json.dumps(content)))]
    fake_client.chat.completions.create.return_value = fake_response
    return fake_client


def _call_arc_pass(mock_content: dict, **overrides):
    kwargs = dict(
        story="a wedding night",
        transcriptions=TRANSCRIPTIONS,
        candidate_indices=[0, 1, 2, 3, 4, 5],
        api_key="fake-key",
        tone="cinematic",
        visual_meta=VISUAL_META,
        costs=COSTS,
        budget_sec=12.0,
        rel_scores=REL_SCORES,
    )
    kwargs.update(overrides)
    with patch("openai.OpenAI", return_value=_mock_openai_response(mock_content)):
        return _llm_arc_pass(**kwargs)


def test_no_api_key_returns_none_without_calling_openai():
    with patch("openai.OpenAI") as mock_cls:
        result = _llm_arc_pass(
            story="x", transcriptions=TRANSCRIPTIONS, candidate_indices=[0, 1],
            api_key=None,
        )
    assert result is None
    mock_cls.assert_not_called()


def test_parses_order_weights_and_transitions():
    # 5 kept out of 6 candidates stays comfortably above the min_keep floor
    # (4 by default with no avg_shot_len given), so the explicit drop sticks.
    # budget_sec is raised so the separate budget-backstop trimming (tested
    # on its own below) doesn't also kick in and remove a 6th clip here.
    result = _call_arc_pass({
        "order": [0, 1, 3, 4, 5],
        "dropped": [2],
        "weights": {"0": 1.0, "1": 1.6, "3": 1.4},
        "transitions": {"1": "slide_up", "3": "zoom_out"},
    }, budget_sec=20.0)
    assert result is not None
    order, weights, transitions = result
    assert order == [0, 1, 3, 4, 5]
    assert weights == {0: 1.0, 1: 1.6, 3: 1.4}
    assert transitions == {1: "slide_up", 3: "zoom_out"}


def test_invalid_transition_name_is_dropped():
    result = _call_arc_pass({
        "order": [0, 1],
        "dropped": [],
        "weights": {},
        "transitions": {"1": "wipe_diagonal"},  # not in transition.py's vocabulary
    })
    _, _, transitions = result
    assert 1 not in transitions


def test_first_clip_in_order_has_no_transition_expected():
    # Well-behaved model output never assigns a transition to the opener; even
    # if it tried to, our id-keyed dict just wouldn't be consulted for index 0
    # by editor.py (it has nothing to transition FROM). Confirms parsing simply
    # passes through whatever valid entries exist.
    result = _call_arc_pass({
        "order": [0, 1],
        "dropped": [],
        "weights": {"0": 1.0, "1": 1.0},
        "transitions": {"0": "crossfade", "1": "fadein"},
    })
    _, _, transitions = result
    assert transitions == {0: "crossfade", 1: "fadein"}


def test_weights_are_clamped_to_valid_range():
    result = _call_arc_pass({
        "order": [0, 1],
        "dropped": [],
        "weights": {"0": 99.0, "1": -5.0},
    })
    _, weights, _ = result
    assert weights[0] == 2.5  # clamped to max
    assert 1 not in weights  # non-positive weight rejected outright


def test_unknown_clip_ids_in_response_are_ignored():
    result = _call_arc_pass({
        "order": [0, 1, 999],
        "dropped": [888],
        "weights": {"777": 1.0},
        "transitions": {"777": "crossfade"},
    })
    order, weights, transitions = result
    assert 999 not in order
    assert 777 not in weights
    assert 777 not in transitions


def test_never_drops_below_min_keep_floor():
    # Model tries to keep only clip 0 and drop 1-3; with 6 candidates and no
    # avg_shot_len given, the floor defaults to 4, so clips must be
    # resurrected (from the "neither kept nor dropped" pool first, i.e. 4
    # and 5, then explicitly-dropped ones if still short) until it's met.
    result = _call_arc_pass({
        "order": [0],
        "dropped": [1, 2, 3],
        "weights": {},
    })
    order, _, _ = result
    assert len(order) >= 4


def test_budget_backstop_trims_when_model_keeps_too_much():
    # 6 clips at 3.0s each = 18.0s against an 8.0s budget (*1.05 = 8.4s) -> the
    # backstop repeatedly drops the lowest-rel_score middle clip (never
    # first/last) but stops once it hits the min_keep floor of 4 (no
    # avg_shot_len given here, so the floor defaults to 4) — even though 4
    # clips (12.0s) still exceeds the budget. The floor takes priority over
    # an exact budget fit; this is the fix for the "arc pass over-drops"
    # issue that collapsed a real 81-clip edit down to just 4 clips.
    result = _call_arc_pass(
        {"order": [0, 1, 2, 3, 4, 5], "dropped": [], "weights": {}},
        budget_sec=8.0,
    )
    order, _, _ = result
    assert order[0] == 0
    assert order[-1] == 5
    assert len(order) == 4
    assert 2 not in order  # lowest rel_score middle clip, dropped first
    assert 4 not in order  # second-lowest rel_score middle clip, dropped second


def test_budget_aware_floor_resists_over_dropping_at_scale():
    # Regression test for the actual bug: a large duration budget with a
    # short avg_shot_len (Energetic tone) needs many more than 4 clips to
    # fill properly. If the model tries to prune down to a tiny handful
    # anyway, the budget-aware floor should refuse and resurrect clips
    # until there are enough to fill the budget without wildly overrunning
    # each clip's planned screen time.
    result = _call_arc_pass(
        {"order": [0, 1], "dropped": [2, 3, 4, 5], "weights": {}},
        budget_sec=60.0,
        avg_shot_len=2.6,  # Energetic tone's average shot length
    )
    order, _, _ = result
    # ceil(60.0 / 2.6) = 24, but only 6 candidates exist -> floor caps at 6
    assert len(order) == 6


def test_malformed_json_falls_back_to_none():
    fake_client = MagicMock()
    fake_response = MagicMock()
    fake_response.choices = [MagicMock(message=MagicMock(content="not json"))]
    fake_client.chat.completions.create.return_value = fake_response
    with patch("openai.OpenAI", return_value=fake_client):
        result = _llm_arc_pass(
            story="x", transcriptions=TRANSCRIPTIONS, candidate_indices=[0, 1],
            api_key="fake-key", visual_meta=VISUAL_META, costs=COSTS[:2],
            budget_sec=6.0, rel_scores=REL_SCORES[:2],
        )
    assert result is None


def test_openai_exception_falls_back_to_none():
    with patch("openai.OpenAI", side_effect=RuntimeError("network down")):
        result = _llm_arc_pass(
            story="x", transcriptions=TRANSCRIPTIONS, candidate_indices=[0, 1],
            api_key="fake-key",
        )
    assert result is None
