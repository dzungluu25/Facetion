"""Default paths and tunable settings."""

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
CASCADE_FILE = PACKAGE_DIR / "models" / "haarcascade_frontalface_default.xml"

# Where face images, names and trained models are stored.
# Override with the FACETION_DATA_DIR environment variable or --data-dir.
DEFAULT_DATA_DIR = Path(os.environ.get("FACETION_DATA_DIR", "data"))

# Where downloaded deep-learning models are cached.
MODELS_CACHE_DIR = Path(
    os.environ.get("FACETION_MODELS_DIR", Path.home() / ".cache" / "facetion")
)

CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

SAMPLES_PER_USER = 50

# Extra margin saved around each captured face, as a fraction of its size
CROP_PADDING = 0.25

DEFAULT_BACKEND = "lbph"

ESC_KEY = 27

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def images_dir(data_dir: Path) -> Path:
    return data_dir / "images"


def names_file(data_dir: Path) -> Path:
    return data_dir / "names.json"
