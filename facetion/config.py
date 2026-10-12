"""Shared paths and tunable settings."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
IMAGES_DIR = DATA_DIR / "images"
NAMES_FILE = DATA_DIR / "names.json"
MODEL_FILE = DATA_DIR / "trainer.yml"

MODELS_DIR = ROOT_DIR / "models"
CASCADE_FILE = MODELS_DIR / "haarcascade_frontalface_default.xml"

CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

SAMPLES_PER_USER = 50

# Max LBPH distance accepted as a match (lower = stricter)
CONFIDENCE_THRESHOLD = 51

ESC_KEY = 27
