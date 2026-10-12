"""Lightweight hand tracking with OpenCV DNN: MediaPipe palm detector + hand landmarks.

Ported from OpenCV Zoo's ``palm_detection_mediapipe`` and
``handpose_estimation_mediapipe`` demos (Apache License 2.0):
https://github.com/opencv/opencv_zoo

Only OpenCV is needed (no MediaPipe package), so it runs anywhere OpenCV does.
After a hand is found, the next frame is cropped around its previous position
and the palm detector is skipped, which roughly doubles the frame rate.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from facetion.model_zoo import ensure_model

# Hand landmarks that correspond to the palm detector's 7 keypoints
_PALM_LANDMARK_IDS = [0, 5, 9, 13, 17, 1, 2]


@dataclass
class Hand:
    landmarks: np.ndarray  # (21, 2) pixel coordinates
    handedness: float      # 0 = left, 1 = right (as seen in the image)
    confidence: float

    def normalized(self, width: int, height: int) -> list[tuple[float, float]]:
        return [(float(x) / width, float(y) / height) for x, y in self.landmarks]


def _palm_anchors() -> np.ndarray:
    """SSD anchors for the 192x192 palm detector (2 per cell at stride 8, 6 at stride 16)."""
    anchors = []
    for grid, per_cell in ((24, 2), (12, 6)):
        for y in range(grid):
            for x in range(grid):
                anchors.extend([((x + 0.5) / grid, (y + 0.5) / grid)] * per_cell)
    return np.array(anchors, dtype=np.float32)


class PalmDetector:
    input_size = np.array([192, 192])

    def __init__(self, model_path: str, score_threshold: float = 0.5, nms_threshold: float = 0.3):
        self.net = cv2.dnn.readNet(model_path)
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.anchors = _palm_anchors()

    def detect(self, image: np.ndarray) -> np.ndarray:
        """Return palms as rows of [x1, y1, x2, y2, 7 keypoints (x, y), score]."""
        height, width = image.shape[:2]
        ratio = min(self.input_size / np.array([height, width]))
        new_h, new_w = int(height * ratio), int(width * ratio)
        resized = cv2.resize(image, (new_w, new_h))
        pad_w, pad_h = self.input_size[1] - new_w, self.input_size[0] - new_h
        left, top = pad_w // 2, pad_h // 2
        padded = cv2.copyMakeBorder(
            resized, top, pad_h - top, left, pad_w - left, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        blob = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32)[np.newaxis] / 255.0
        pad_bias = (np.array([left, top]) / ratio).astype(np.int32)

        self.net.setInput(blob)
        regressors, scores = self.net.forward(self.net.getUnconnectedOutLayersNames())

        score = 1 / (1 + np.exp(-scores[0, :, 0].astype(np.float64)))
        deltas = regressors[0]
        scale = max(width, height)
        center = deltas[:, :2] / self.input_size
        size = deltas[:, 2:4] / self.input_size
        top_left = (center - size / 2 + self.anchors) * scale
        bottom_right = (center + size / 2 + self.anchors) * scale
        boxes = np.concatenate([top_left, bottom_right], axis=1)
        boxes -= np.tile(pad_bias, 2)

        keep = cv2.dnn.NMSBoxes(boxes, score, self.score_threshold, self.nms_threshold)
        if len(keep) == 0:
            return np.empty((0, 19))
        keep = np.array(keep).flatten()
        keypoints = deltas[keep, 4:].reshape(-1, 7, 2) / self.input_size
        keypoints = (keypoints + self.anchors[keep][:, None, :]) * scale - pad_bias
        return np.c_[boxes[keep], keypoints.reshape(-1, 14), score[keep]]


class HandLandmarker:
    input_size = np.array([224, 224])

    def __init__(self, model_path: str, confidence_threshold: float = 0.8):
        self.net = cv2.dnn.readNet(model_path)
        self.confidence_threshold = confidence_threshold

    @staticmethod
    def _crop_square(image, box, shift, enlarge, for_rotation):
        """Shift and enlarge a [[x1, y1], [x2, y2]] box, crop it and pad it square."""
        box = box + np.array(shift) * (box[1] - box[0])
        center = box.sum(axis=0) / 2
        half = (box[1] - box[0]) * enlarge / 2
        box = np.array([center - half, center + half]).astype(np.int32)
        box[:, 0] = np.clip(box[:, 0], 0, image.shape[1])
        box[:, 1] = np.clip(box[:, 1], 0, image.shape[0])
        crop = image[box[0][1]:box[1][1], box[0][0]:box[1][0]]
        side = int(np.linalg.norm(crop.shape[:2]) if for_rotation else max(crop.shape[:2]))
        pad_h, pad_w = side - crop.shape[0], side - crop.shape[1]
        left, top = pad_w // 2, pad_h // 2
        crop = cv2.copyMakeBorder(
            crop, top, pad_h - top, left, pad_w - left, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        return crop, box, box[0] - [left, top]

    def detect(self, image: np.ndarray, palm: np.ndarray) -> Hand | None:
        # Crop generously around the palm, then rotate so the hand points up
        crop, _, pad_bias = self._crop_square(
            image, palm[0:4].reshape(2, 2), (0, 0), 4, for_rotation=True
        )
        if crop.size == 0:
            return None
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        keypoints = palm[4:18].reshape(7, 2) - pad_bias
        wrist, middle = keypoints[0], keypoints[2]
        radians = np.pi / 2 - np.arctan2(-(middle[1] - wrist[1]), middle[0] - wrist[0])
        radians -= 2 * np.pi * np.floor((radians + np.pi) / (2 * np.pi))
        angle = np.rad2deg(radians)

        palm_box = palm[0:4].reshape(2, 2) - pad_bias
        rotation = cv2.getRotationMatrix2D(palm_box.sum(axis=0) / 2, angle, 1.0)
        rotated = cv2.warpAffine(crop, rotation, (crop.shape[1], crop.shape[0]))
        rotated_points = np.c_[keypoints, np.ones(len(keypoints))] @ rotation.T
        rotated_box = np.array([rotated_points.min(axis=0), rotated_points.max(axis=0)])
        hand_crop, rotated_box, _ = self._crop_square(
            rotated, rotated_box, (0, -0.4), 3, for_rotation=False
        )
        if hand_crop.size == 0:
            return None

        blob = cv2.resize(hand_crop, tuple(self.input_size), interpolation=cv2.INTER_AREA)
        self.net.setInput(blob.astype(np.float32)[np.newaxis] / 255.0)
        landmarks, confidence, handedness, _ = self.net.forward(
            self.net.getUnconnectedOutLayersNames()
        )
        confidence = float(confidence[0][0])
        if confidence < self.confidence_threshold:
            return None

        # Map the landmarks back into the original image
        points = landmarks[0].reshape(-1, 3)[:, :2]
        scale = max((rotated_box[1] - rotated_box[0]) / self.input_size)
        points = (points - self.input_size / 2) * scale
        points = points @ cv2.getRotationMatrix2D((0, 0), angle, 1.0)[:, :2]
        inverse = cv2.invertAffineTransform(rotation)
        center = np.append(rotated_box.sum(axis=0) / 2, 1) @ inverse.T
        points = points + center + pad_bias
        return Hand(points, float(handedness[0][0]), confidence)


class HandTracker:
    """Finds one hand per frame, tracking it between frames for speed."""

    def __init__(self, palm_score_threshold: float = 0.5, confidence_threshold: float = 0.8):
        self.palm_detector = PalmDetector(str(ensure_model("palm")), palm_score_threshold)
        self.landmarker = HandLandmarker(str(ensure_model("handpose")), confidence_threshold)
        self._previous: Hand | None = None

    @staticmethod
    def _palm_from_hand(hand: Hand) -> np.ndarray:
        keypoints = hand.landmarks[_PALM_LANDMARK_IDS]
        box = np.r_[keypoints.min(axis=0), keypoints.max(axis=0)]
        return np.r_[box, keypoints.reshape(-1), hand.confidence]

    def process(self, frame: np.ndarray) -> Hand | None:
        hand = None
        if self._previous is not None:
            hand = self.landmarker.detect(frame, self._palm_from_hand(self._previous))
        if hand is None:
            palms = self.palm_detector.detect(frame)
            if len(palms):
                best = palms[np.argmax(palms[:, -1])]
                hand = self.landmarker.detect(frame, best)
        self._previous = hand
        return hand
