import os
import shutil
import subprocess
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from media_probe import probe_size, mean_volume_db  # noqa: E402,F401  (re-exported for fixtures below)

HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


@pytest.fixture(scope="session")
def require_ffmpeg():
    if not HAS_FFMPEG:
        pytest.skip("ffmpeg/ffprobe not available on PATH")


@pytest.fixture
def make_clip(tmp_path, require_ffmpeg):
    """Factory fixture: make_clip(name, width, height, duration=2.0, hue=0, freq=440)
    renders a tiny synthetic test-pattern clip with a tone and returns its path."""

    def _make(name: str, width: int, height: int, duration: float = 2.0,
              hue: int = 0, freq: int = 440) -> str:
        out = str(tmp_path / name)
        vf = f"testsrc=size={width}x{height}:duration={duration}:rate=24"
        if hue:
            vf += f",hue=h={hue}"
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", vf,
            "-f", "lavfi", "-i", f"sine=frequency={freq}:duration={duration}",
            "-c:v", "libx264", "-c:a", "aac", "-pix_fmt", "yuv420p",
            "-loglevel", "error",
            out,
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return out

    return _make
