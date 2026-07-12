"""ffprobe/ffmpeg-based assertions shared by integration tests."""
import subprocess


def probe_size(path: str):
    """(width, height) via ffprobe, for use in integration test assertions."""
    res = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0", path],
        capture_output=True, text=True, check=True,
    )
    w, h = res.stdout.strip().split("x")
    return int(w), int(h)


def mean_volume_db(path: str) -> float:
    """Mean volume (dB) of a media file's audio track, via ffmpeg volumedetect."""
    res = subprocess.run(
        ["ffmpeg", "-i", path, "-af", "volumedetect", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    for line in res.stderr.splitlines():
        if "mean_volume" in line:
            return float(line.split(":")[1].strip().split(" ")[0])
    raise AssertionError(f"Could not find mean_volume in ffmpeg output for {path}")
