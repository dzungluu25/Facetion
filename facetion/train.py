"""Step 2: train a recognition backend on the captured samples."""

from pathlib import Path

from facetion import config
from facetion.backends import get_backend


def train_model(
    data_dir: Path = config.DEFAULT_DATA_DIR, backend: str = config.DEFAULT_BACKEND
) -> int:
    """Train and save the model; return the number of distinct users."""
    model = get_backend(backend, data_dir)
    print(f"\n[INFO] Training {backend} model...")
    user_count = model.train()
    print(f"\n[INFO] {user_count} users trained. Model saved to {model.model_path()}")
    return user_count
