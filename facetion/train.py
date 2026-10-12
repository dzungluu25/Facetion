"""Step 2: train an LBPH face recognizer on the captured samples."""

from pathlib import Path

import cv2
import numpy as np

from facetion import config
from facetion.utils import parse_user_id


def load_samples(data_dir: Path) -> tuple[list[np.ndarray], list[int]]:
    """Load the grayscale face crops and their user IDs."""
    faces, ids = [], []
    for image_path in sorted(config.images_dir(data_dir).glob("user-*-*.jpg")):
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        faces.append(image)
        ids.append(parse_user_id(image_path))
    return faces, ids


def train_model(data_dir: Path = config.DEFAULT_DATA_DIR) -> int:
    """Train and save the model; return the number of distinct users."""
    faces, ids = load_samples(data_dir)
    if not faces:
        raise SystemExit(
            f"No samples found in {config.images_dir(data_dir)}. "
            "Run 'facetion capture' first."
        )

    print("\n[INFO] Training...")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(ids))
    model_path = config.model_file(data_dir)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    recognizer.write(str(model_path))

    user_count = len(set(ids))
    print(f"\n[INFO] {user_count} users trained. Model saved to {model_path}")
    return user_count
