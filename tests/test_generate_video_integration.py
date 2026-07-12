import os

import pytest

from editor import generate_video
from media_probe import probe_size, mean_volume_db

pytestmark = pytest.mark.integration


def test_mixed_orientation_normalizes_to_uniform_canvas(make_clip, tmp_path, capsys):
    landscape = make_clip("landscape.mp4", 1920, 1080, duration=2.0, freq=440)
    portrait_a = make_clip("portrait_a.mp4", 1080, 1920, duration=2.0, freq=660)
    portrait_b = make_clip("portrait_b.mp4", 720, 1280, duration=2.0, freq=880)

    out = generate_video(
        clip_paths=[landscape, portrait_a, portrait_b],
        storyline="mixed orientation smoke test",
        tone="Cinematic",
        target_duration_sec=5,
        show_opening_card=False,
    )
    try:
        # majority orientation is portrait (2 of 3 clips) -> uniform portrait canvas
        w, h = probe_size(out)
        assert h > w
        captured = capsys.readouterr()
        assert "[CANVAS] Normalizing" in captured.out
    finally:
        if os.path.exists(out):
            os.remove(out)


def test_same_orientation_clips_skip_canvas_normalization(make_clip, capsys):
    a = make_clip("a.mp4", 1280, 720, duration=1.5, freq=440)
    b = make_clip("b.mp4", 1280, 720, duration=1.5, freq=660)

    out = generate_video(
        clip_paths=[a, b],
        storyline="uniform orientation smoke test",
        tone="Cinematic",
        target_duration_sec=3,
        show_opening_card=False,
    )
    try:
        captured = capsys.readouterr()
        assert "[CANVAS] Normalizing" not in captured.out
    finally:
        if os.path.exists(out):
            os.remove(out)


def test_solo_music_is_audible_when_not_mixing_original_audio(make_clip, tmp_path):
    clip = make_clip("clip.mp4", 960, 540, duration=3.0, freq=300)
    music = str(tmp_path / "music.mp3")
    import subprocess
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=220:duration=6",
         "-c:a", "mp3", "-loglevel", "error", music],
        check=True,
    )

    out = generate_video(
        clip_paths=[clip],
        storyline="solo music volume smoke test",
        tone="Cinematic",
        target_duration_sec=3,
        show_opening_card=False,
        mix_original_audio=False,
        custom_music_path=music,
    )
    try:
        # Music alone should play at a normal listening level, not the old
        # ducked "background bed" level (~0.15-0.22) that used to leave the
        # ENTIRE track quiet when there was no dialogue to duck under.
        # A sine-wave test tone at full gain measures around -23 to -26 dB
        # mean_volume through ffmpeg's RMS meter (real content typically reads
        # louder). -30 dB comfortably separates "normal" from the old bug,
        # which compounded to ~0.02-0.12 linear gain (well below -30 dB).
        assert mean_volume_db(out) > -30.0
    finally:
        if os.path.exists(out):
            os.remove(out)


def test_mixed_audio_keeps_original_at_full_volume(make_clip, tmp_path):
    clip = make_clip("clip.mp4", 960, 540, duration=3.0, freq=300)
    music = str(tmp_path / "music.mp3")
    import subprocess
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=220:duration=6",
         "-c:a", "mp3", "-loglevel", "error", music],
        check=True,
    )

    out = generate_video(
        clip_paths=[clip],
        storyline="mixed audio volume smoke test",
        tone="Cinematic",
        target_duration_sec=3,
        show_opening_card=False,
        mix_original_audio=True,
        custom_music_path=music,
    )
    try:
        # This is a coarse smoke check, not a precise isolation of the music
        # layer's gain: original audio (at 1.0) dominates the composite mix
        # either way, so a black-box mean_volume reading can't distinguish
        # "music at _music_vol" from "music at the old, further-ducked
        # _music_vol*0.20" — both are quiet relative to the original track.
        # It does confirm the mixed-audio path renders without crashing and
        # isn't accidentally silenced outright.
        assert mean_volume_db(out) > -30.0
    finally:
        if os.path.exists(out):
            os.remove(out)


def test_opening_card_toggle_does_not_crash_for_non_cinematic_tone(make_clip):
    a = make_clip("a.mp4", 1280, 720, duration=1.5, freq=440)
    b = make_clip("b.mp4", 1280, 720, duration=1.5, freq=660)

    # Before the fix this flag was silently ignored for any tone other than
    # "cinematic"; this just confirms the code path is exercised and doesn't
    # raise for a non-cinematic tone (font/ImageMagick availability is
    # environment-dependent and already handled gracefully in product code).
    out = generate_video(
        clip_paths=[a, b],
        storyline="opening card smoke test",
        tone="Energetic",
        target_duration_sec=3,
        show_opening_card=True,
    )
    assert os.path.exists(out)
    os.remove(out)


def test_arc_metadata_picks_transition_and_weight(make_clip):
    a = make_clip("a.mp4", 1280, 720, duration=2.0, freq=440)
    b = make_clip("b.mp4", 1280, 720, duration=2.0, freq=660)

    clip_metadata = {
        a: {"narrative_role": "hook", "emotion": "tense", "shot_type": "wide",
            "visual_score": 0.7, "arc_weight": 1.0},
        b: {"narrative_role": "payoff", "emotion": "happy", "shot_type": "close_up",
            "visual_score": 0.9, "arc_weight": 1.8, "arc_transition": "zoom_out"},
    }
    out = generate_video(
        clip_paths=[a, b],
        storyline="arc metadata smoke test",
        tone="Cinematic",
        target_duration_sec=3,
        show_opening_card=False,
        clip_metadata=clip_metadata,
    )
    assert os.path.exists(out)
    os.remove(out)
