"""Common interface for recognition backends."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from facetion import config
from facetion.utils import load_names, parse_user_id

UNKNOWN_LABEL = "Who are you?"


@dataclass
class Match:
    box: tuple[int, int, int, int]  # x, y, w, h
    name: str
    score: float | None  # 0-100, higher is more confident; None if unknown


class Backend:
    """A face detector + recognizer pair that can be trained and run on frames."""

    name: str
    default_threshold: float

    def __init__(self, data_dir: Path, threshold: float | None = None):
        self.data_dir = data_dir
        self.threshold = self.default_threshold if threshold is None else threshold
        self.names: dict[str, str] = {}

    def model_path(self) -> Path:
        raise NotImplementedError

    def train(self) -> int:
        """Train on the captured images and save the model; return the user count."""
        raise NotImplementedError

    def load(self) -> None:
        """Load the trained model; exits with a helpful message if it is missing."""
        if not self.model_path().exists():
            raise SystemExit(
                f"No trained {self.name} model in {self.data_dir}. "
                f"Run 'facetion train --backend {self.name}' first."
            )
        self.names = load_names(self.data_dir)

    def identify(self, frame: np.ndarray) -> list[Match]:
        """Find and identify every face in a BGR frame."""
        raise NotImplementedError

    def _label(self, user_id: int, score: float, accepted: bool) -> tuple[str, float | None]:
        if accepted and str(user_id) in self.names:
            return self.names[str(user_id)], score
        return UNKNOWN_LABEL, None


def iter_samples(data_dir: Path):
    """Yield (user_id, BGR image) for every captured face image."""
    import cv2

    for image_path in sorted(config.images_dir(data_dir).glob("user-*-*.jpg")):
        image = cv2.imread(str(image_path))
        if image is not None:
            yield parse_user_id(image_path), image


def no_samples_exit(data_dir: Path) -> SystemExit:
    return SystemExit(
        f"No samples found in {config.images_dir(data_dir)}. Run 'facetion capture' first."
    )
