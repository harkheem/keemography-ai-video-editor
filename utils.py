# utils.py
import tempfile
from typing import List, Tuple, Optional


# -------------------------------
# Scene detection (lazy import)
# -------------------------------
def detect_scenes(video_path: str, threshold: float = 30.0) -> List[Tuple[float, float]]:
    """
    Try to detect scenes using PySceneDetect. If the package isn't available, return [].
    Returns a list of (start_seconds, end_seconds) tuples.
    """
    try:
        from scenedetect import VideoManager, SceneManager  # type: ignore
        from scenedetect.detectors import ContentDetector  # type: ignore
    except Exception as e:
        # Package not installed or failed to load; degrade gracefully.
        print(f"ℹ️ SceneDetect unavailable ({e}); continuing without scene cuts.")
        return []

    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))

    try:
        video_manager.set_downscale_factor()
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)
        scene_list = scene_manager.get_scene_list()
    finally:
        try:
            video_manager.release()
        except Exception:
            pass

    return [(start.get_seconds(), end.get_seconds()) for start, end in scene_list]


# -------------------------------
# Face crop (lazy import)
# -------------------------------
def crop_to_face(frame) -> "np.ndarray":
    """
    Crop around the largest detected face. If OpenCV isn't available or no faces are found,
    return the original frame. Expects an RGB frame (H x W x 3).
    """
    try:
        import cv2  # type: ignore
        import numpy as np  # noqa: F401  (type for return)
    except Exception as e:
        print(f"ℹ️ OpenCV unavailable ({e}); skipping face crop.")
        return frame

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return frame

    # Largest face by area
    (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
    padding = 0.3
    x1 = max(int(x - w * padding), 0)
    y1 = max(int(y - h * padding), 0)
    x2 = min(int(x + w * (1 + padding)), frame.shape[1])
    y2 = min(int(y + h * (1 + padding)), frame.shape[0])
    return frame[y1:y2, x1:x2]


# -------------------------------
# PIL text overlay (no ImageMagick)
# -------------------------------
def add_text_overlay(clip, text: str, fontsize: int = 48, color: str = "white"):
    """
    Draw a centered text overlay using PIL onto a transparent RGBA image,
    then composite it over the given MoviePy clip. No ImageMagick required.
    """
    from moviepy.editor import ImageClip, CompositeVideoClip  # lightweight
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    W, H = clip.size
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Try to use a common font installed via packages.txt (fonts-dejavu-core)
    def _load_font(size: int):
        try:
            return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
        except Exception:
            return ImageFont.load_default()

    font = _load_font(fontsize)

    # Fit text within 90% of width
    max_width = int(W * 0.9)
    bbox = draw.textbbox((0, 0), text, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    while w > max_width and fontsize > 16:
        fontsize -= 2
        font = _load_font(fontsize)
        bbox = draw.textbbox((0, 0), text, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]

    # Centered position
    x = (W - w) // 2
    y = (H - h) // 2

    # Soft shadow for readability
    shadow = (0, 0, 0, 180)
    draw.text((x + 2, y + 2), text, fill=shadow, font=font)
    draw.text((x, y), text, fill=color, font=font)

    overlay = ImageClip(np.array(img)).set_duration(clip.duration).set_position(("center", "center"))
    return CompositeVideoClip([clip, overlay])


# -------------------------------
# Basic audio gain
# -------------------------------
def normalize_audio(clip, gain: float = 1.5):
    """
    Simple gain boost via volumex. For true normalization/limiting,
    analyze peaks and apply compression.
    """
    try:
        return clip.volumex(gain)
    except Exception:
        return clip


# -------------------------------
# Temp file helper
# -------------------------------
def create_temp_file(suffix: str = ".mp4") -> str:
    """Create a named temp file and return its path."""
    return tempfile.NamedTemporaryFile(delete=False, suffix=suffix).name
