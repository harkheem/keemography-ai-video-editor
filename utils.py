import tempfile
from io import BytesIO

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from moviepy.editor import ImageClip, CompositeVideoClip


# --- SCENE DETECTION ---
def detect_scenes(video_path: str, threshold: float = 30.0):
    """
    Returns list of (start_seconds, end_seconds) tuples for detected scenes.
    """
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))

    video_manager.set_downscale_factor()
    video_manager.start()

    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list()
    video_manager.release()

    return [(start.get_seconds(), end.get_seconds()) for start, end in scene_list]


# --- FACE DETECTION & CROPPING ---
def crop_to_face(frame: np.ndarray) -> np.ndarray:
    """
    Given an RGB frame (H x W x 3), returns a cropped region around the largest face.
    If no face is detected, returns the original frame.
    """
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return frame

    # Pick the largest face by area
    (x, y, w, h) = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
    padding = 0.3
    x1 = max(int(x - w * padding), 0)
    y1 = max(int(y - h * padding), 0)
    x2 = min(int(x + w * (1 + padding)), frame.shape[1])
    y2 = min(int(y + h * (1 + padding)), frame.shape[0])
    return frame[y1:y2, x1:x2]


# --- PIL-BASED TEXT OVERLAY (no ImageMagick) ---
def add_text_overlay(clip, text: str, fontsize: int = 48, color: str = "white"):
    """
    Draw a centered text overlay using PIL on a transparent RGBA image,
    then composite it over the given MoviePy clip. No ImageMagick required.
    """
    W, H = clip.size
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Try a common font available via packages.txt (fonts-dejavu-core)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", fontsize)
    except Exception:
        font = ImageFont.load_default()

    # Shrink to fit width if necessary
    max_width = int(W * 0.9)
    txt = text
    bbox = draw.textbbox((0, 0), txt, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    while w > max_width and fontsize > 16:
        fontsize -= 2
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", fontsize)
        except Exception:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), txt, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]

    # Center position
    x = (W - w) // 2
    y = (H - h) // 2

    # Optional shadow to improve readability
    shadow_color = (0, 0, 0, 180)
    draw.text((x + 2, y + 2), txt, fill=shadow_color, font=font)
    draw.text((x, y), txt, fill=color, font=font)

    overlay = ImageClip(np.array(img)).set_duration(clip.duration).set_position(("center", "center"))
    return CompositeVideoClip([clip, overlay])


# --- BASIC AUDIO NORMALIZATION ---
def normalize_audio(clip, gain: float = 1.5):
    """
    Simple gain boost with volumex. For true normalization/limiting, you'd analyze peaks and apply compression.
    """
    try:
        return clip.volumex(gain)
    except Exception:
        return clip


# --- SAFE TEMP FILE GENERATOR ---
def create_temp_file(suffix: str = ".mp4") -> str:
    return tempfile.NamedTemporaryFile(delete=False, suffix=suffix).name
