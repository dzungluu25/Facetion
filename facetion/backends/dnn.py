"""Deep-learning backend: YuNet detection + SFace embeddings. More accurate."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from facetion.backends.base import Backend, Match, iter_samples, no_samples_exit
from facetion.model_zoo import ensure_model

# YuNet works poorly on tiny inputs, so small samples are upscaled first
_MIN_DETECT_SIZE = 320


class DNNBackend(Backend):
    name = "dnn"
    # Cosine similarity between face embeddings: higher means a better match.
    # 0.363 is the threshold recommended by the SFace authors.
    default_threshold = 0.363

    def __init__(self, data_dir: Path, threshold: float | None = None):
        super().__init__(data_dir, threshold)
        self.detector = cv2.FaceDetectorYN.create(str(ensure_model("yunet")), "", (320, 320))
        self.recognizer = cv2.FaceRecognizerSF.create(str(ensure_model("sface")), "")
        self.features = np.empty((0, 128), dtype=np.float32)
        self.ids = np.empty(0, dtype=np.int64)

    def model_path(self) -> Path:
        return self.data_dir / "embeddings.npz"

    def _detect(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        self.detector.setInputSize((width, height))
        _, faces = self.detector.detect(image)
        return np.empty((0, 15), dtype=np.float32) if faces is None else faces

    def _embed(self, image: np.ndarray, face: np.ndarray) -> np.ndarray:
        aligned = self.recognizer.alignCrop(image, face)
        feature = self.recognizer.feature(aligned).flatten()
        return feature / np.linalg.norm(feature)

    def _sample_feature(self, image: np.ndarray) -> np.ndarray | None:
        scale = max(1.0, _MIN_DETECT_SIZE / min(image.shape[:2]))
        if scale > 1:
            image = cv2.resize(image, None, fx=scale, fy=scale)
        faces = self._detect(image)
        if len(faces) == 0:
            return None
        best = max(faces, key=lambda f: f[-1])  # highest detection score
        return self._embed(image, best)

    def train(self) -> int:
        features, ids = [], []
        for user_id, image in iter_samples(self.data_dir):
            feature = self._sample_feature(image)
            if feature is not None:
                features.append(feature)
                ids.append(user_id)
        if not features:
            raise no_samples_exit(self.data_dir)

        self.model_path().parent.mkdir(parents=True, exist_ok=True)
        np.savez(self.model_path(), features=np.array(features), ids=np.array(ids))
        return len(set(ids))

    def load(self) -> None:
        super().load()
        with np.load(self.model_path()) as model:
            self.features = model["features"]
            self.ids = model["ids"]

    def identify(self, frame: np.ndarray) -> list[Match]:
        matches = []
        for face in self._detect(frame):
            feature = self._embed(frame, face)
            similarities = self.features @ feature
            best = int(np.argmax(similarities))
            similarity = float(similarities[best])
            name, score = self._label(
                int(self.ids[best]), max(0.0, similarity) * 100, similarity >= self.threshold
            )
            x, y, w, h = (int(v) for v in face[:4])
            matches.append(Match((x, y, w, h), name, score))
        return matches
