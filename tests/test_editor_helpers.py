from editor import normalize_clip_paths, _probe_resolution, _is_4k


def test_normalize_clip_paths_filters_non_strings():
    result = normalize_clip_paths(["/a.mp4", None, 123, "/b.mp4", [], "/c.mp4"])
    assert result == ["/a.mp4", "/b.mp4", "/c.mp4"]


def test_normalize_clip_paths_empty_input():
    assert normalize_clip_paths([]) == []


def test_probe_resolution_missing_file_returns_zero():
    assert _probe_resolution("/no/such/file.mp4") == (0, 0)


def test_probe_resolution_none_path_returns_zero():
    assert _probe_resolution(None) == (0, 0)


def test_is_4k_missing_file_is_false():
    assert _is_4k("/no/such/file.mp4") is False


def test_probe_resolution_and_is_4k_on_real_clip(make_clip):
    hd_path = make_clip("hd.mp4", 1280, 720, duration=1.0)
    assert _probe_resolution(hd_path) == (1280, 720)
    assert _is_4k(hd_path) is False

    uhd_path = make_clip("uhd.mp4", 3840, 2160, duration=1.0)
    assert _probe_resolution(uhd_path) == (3840, 2160)
    assert _is_4k(uhd_path) is True
