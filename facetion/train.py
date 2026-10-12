"""Step 2: train an LBPH face recognizer on the captured samples."""

import cv2
import numpy as np

from facetion import config
from facetion.utils import parse_user_id


def load_samples() -> tuple[list[np.ndarray], list[int]]:
    """Load the grayscale face crops and their user IDs."""
    faces, ids = [], []
    for image_path in sorted(config.IMAGES_DIR.glob("user-*-*.jpg")):
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        faces.append(image)
        ids.append(parse_user_id(image_path))
    return faces, ids


def train_model() -> int:
    """Train and save the model; return the number of distinct users."""
    faces, ids = load_samples()
    if not faces:
        raise SystemExit(
            f"No samples found in {config.IMAGES_DIR}. Run the capture step first."
        )

    print("\n[INFO] Training...")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(ids))
    config.MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    recognizer.write(str(config.MODEL_FILE))

    user_count = len(set(ids))
    print(f"\n[INFO] {user_count} users trained. Model saved to {config.MODEL_FILE}")
    return user_count


def main() -> None:
    train_model()


if __name__ == "__main__":
    main()
