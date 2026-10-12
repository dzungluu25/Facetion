"""Helpers shared by the capture, train and recognize steps."""

import json
from pathlib import Path

import cv2

from facetion import config


def load_names(data_dir: Path) -> dict[str, str]:
    """Return the ID -> name mapping, or an empty dict if none exists yet."""
    path = config.names_file(data_dir)
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def save_names(names: dict[str, str], data_dir: Path) -> None:
    """Write the ID -> name mapping to disk."""
    path = config.names_file(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(names, f, ensure_ascii=False, indent=4)


def parse_user_id(image_path: Path) -> int:
    """Extract the user ID from a file named ``user-<id>-<n>.jpg``."""
    return int(image_path.stem.split("-")[1])


def load_face_detector() -> cv2.CascadeClassifier:
    detector = cv2.CascadeClassifier(str(config.CASCADE_FILE))
    if detector.empty():
        raise RuntimeError(f"Could not load Haar cascade from {config.CASCADE_FILE}")
    return detector


def open_camera(index: int = config.CAMERA_INDEX) -> cv2.VideoCapture:
    camera = cv2.VideoCapture(index)
    if not camera.isOpened():
        raise SystemExit(
            f"Could not open camera {index}. Check that it is connected and "
            "that your terminal has camera permission."
        )
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    return camera
