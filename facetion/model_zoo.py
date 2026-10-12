"""Download and cache the deep-learning models (face and hand)."""

import hashlib
import urllib.request
from pathlib import Path

import cv2

from facetion import config

# Hide OpenCV's harmless DNN target warnings
if hasattr(cv2, "utils") and hasattr(cv2.utils, "logging"):
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)

_ZOO_URL = "https://github.com/opencv/opencv_zoo/raw/main/models"

MODELS = {
    # YuNet face detector (MIT license)
    "yunet": (
        f"{_ZOO_URL}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    ),
    # SFace face recognizer (Apache 2.0 license)
    "sface": (
        f"{_ZOO_URL}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
    ),
    # MediaPipe palm detector and hand landmarks, converted by OpenCV Zoo (Apache 2.0)
    "palm": (
        f"{_ZOO_URL}/palm_detection_mediapipe/palm_detection_mediapipe_2023feb.onnx",
        "78ff51c38496b7fc8b8ebdb6cc8c1abb02fa6c38427c6848254cdaba57fcce7c",
    ),
    "handpose": (
        f"{_ZOO_URL}/handpose_estimation_mediapipe/handpose_estimation_mediapipe_2023feb.onnx",
        "db0898ae717b76b075d9bf563af315b29562e11f8df5027a1ef07b02bef6d81c",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_model(name: str, cache_dir: Path = config.MODELS_CACHE_DIR) -> Path:
    """Return the local path to a model, downloading and verifying it if needed."""
    url, expected_sha256 = MODELS[name]
    path = cache_dir / url.rsplit("/", 1)[-1]
    if path.exists() and _sha256(path) == expected_sha256:
        return path

    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Downloading {name} model (one time only)...")
    tmp_path = path.with_suffix(".part")
    try:
        urllib.request.urlretrieve(url, tmp_path)
    except OSError as exc:
        raise SystemExit(f"Could not download the {name} model: {exc}") from exc

    if _sha256(tmp_path) != expected_sha256:
        tmp_path.unlink()
        raise SystemExit(f"Downloaded {name} model failed its checksum; please retry.")
    tmp_path.replace(path)
    return path
