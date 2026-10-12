"""Classic backend: Haar cascade detection + LBPH recognition. Fast, no downloads."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from facetion.backends.base import Backend, Match, iter_samples, no_samples_exit
from facetion.utils import load_face_detector


def _largest(faces) -> tuple[int, int, int, int] | None:
    if len(faces) == 0:
        return None
    return tuple(max(faces, key=lambda f: f[2] * f[3]))


class LBPHBackend(Backend):
    name = "lbph"
    # LBPH returns a distance: lower means a better match
    default_threshold = 51

    def __init__(self, data_dir: Path, threshold: float | None = None):
        super().__init__(data_dir, threshold)
        self.detector = load_face_detector()
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()

    def model_path(self) -> Path:
        return self.data_dir / "trainer.yml"

    def _face_crop(self, image: np.ndarray) -> np.ndarray:
        """Crop the face out of a captured sample, falling back to the whole image."""
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        box = _largest(self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5))
        if box is None:
            return gray
        x, y, w, h = box
        return gray[y:y + h, x:x + w]

    def train(self) -> int:
        faces, ids = [], []
        for user_id, image in iter_samples(self.data_dir):
            faces.append(self._face_crop(image))
            ids.append(user_id)
        if not faces:
            raise no_samples_exit(self.data_dir)

        self.recognizer.train(faces, np.array(ids))
        self.model_path().parent.mkdir(parents=True, exist_ok=True)
        self.recognizer.write(str(self.model_path()))
        return len(set(ids))

    def load(self) -> None:
        super().load()
        self.recognizer.read(str(self.model_path()))

    def identify(self, frame: np.ndarray) -> list[Match]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        min_size = (int(0.05 * width), int(0.05 * height))
        faces = self.detector.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5, minSize=min_size
        )

        matches = []
        for x, y, w, h in faces:
            user_id, distance = self.recognizer.predict(gray[y:y + h, x:x + w])
            name, score = self._label(
                user_id, max(0.0, 100 - distance), distance < self.threshold
            )
            matches.append(Match((int(x), int(y), int(w), int(h)), name, score))
        return matches
