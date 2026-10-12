import cv2
import numpy as np
import pytest


@pytest.fixture
def data_dir(tmp_path):
    return tmp_path / "data"


@pytest.fixture
def make_samples(data_dir):
    """Write fake grayscale face crops for the given user IDs."""

    def _make(user_ids, per_user=3):
        images_dir = data_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(0)
        for user_id in user_ids:
            for n in range(1, per_user + 1):
                image = rng.integers(0, 255, (100, 100), dtype=np.uint8)
                cv2.imwrite(str(images_dir / f"user-{user_id}-{n}.jpg"), image)

    return _make
