"""Default paths and tunable settings."""

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
CASCADE_FILE = PACKAGE_DIR / "models" / "haarcascade_frontalface_default.xml"

# Where face images, names and the trained model are stored.
# Override with the FACETION_DATA_DIR environment variable or --data-dir.
DEFAULT_DATA_DIR = Path(os.environ.get("FACETION_DATA_DIR", "data"))

CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

SAMPLES_PER_USER = 50

# Max LBPH distance accepted as a match (lower = stricter)
CONFIDENCE_THRESHOLD = 51

ESC_KEY = 27


def images_dir(data_dir: Path) -> Path:
    return data_dir / "images"


def names_file(data_dir: Path) -> Path:
    return data_dir / "names.json"


def model_file(data_dir: Path) -> Path:
    return data_dir / "trainer.yml"
