"""Turn 21 hand landmarks into a gesture name.

Landmark indices follow MediaPipe's hand model:
0 wrist; thumb 1-4; index 5-8; middle 9-12; ring 13-16; pinky 17-20.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from enum import Enum

Point = Sequence[float]  # (x, y) normalized to the image, y pointing down

WRIST = 0
THUMB_MCP, THUMB_IP, THUMB_TIP = 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_PIP, RING_TIP = 14, 16
PINKY_MCP, PINKY_PIP, PINKY_TIP = 17, 18, 20

# Distances below are divided by the hand size (wrist to middle knuckle),
# so gestures work at any distance from the camera.
PINCH_RATIO = 0.3
FINGER_EXTENDED_RATIO = 1.15


class Gesture(str, Enum):
    NONE = "none"
    POINT = "point"            # index finger up: move the cursor
    PINCH = "pinch"            # thumb + index touching: click / drag
    PINCH_MIDDLE = "pinch_middle"  # thumb + middle touching: right click
    SCROLL = "scroll"          # index + middle up: scroll
    THUMBS_UP = "thumbs_up"    # thumb up, fist closed: play / pause
    OPEN_PALM = "open_palm"    # all fingers up: pause control
    FIST = "fist"              # all fingers closed: pause control


def _distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def hand_size(landmarks: Sequence[Point]) -> float:
    return _distance(landmarks[WRIST], landmarks[MIDDLE_MCP]) or 1e-6


def fingers_extended(landmarks: Sequence[Point]) -> dict[str, bool]:
    """Return which fingers are straightened out."""
    wrist = landmarks[WRIST]

    def extended(tip: int, pip: int) -> bool:
        reach = _distance(landmarks[tip], wrist)
        return reach > _distance(landmarks[pip], wrist) * FINGER_EXTENDED_RATIO

    # The thumb is extended when its tip is far from the palm (pinky knuckle)
    thumb = (
        _distance(landmarks[THUMB_TIP], landmarks[PINKY_MCP])
        > _distance(landmarks[THUMB_MCP], landmarks[PINKY_MCP]) * 1.2
    )
    return {
        "thumb": thumb,
        "index": extended(INDEX_TIP, INDEX_PIP),
        "middle": extended(MIDDLE_TIP, MIDDLE_PIP),
        "ring": extended(RING_TIP, RING_PIP),
        "pinky": extended(PINKY_TIP, PINKY_PIP),
    }


def pinch_distance(landmarks: Sequence[Point], finger_tip: int) -> float:
    """Thumb-to-fingertip distance, relative to hand size."""
    return _distance(landmarks[THUMB_TIP], landmarks[finger_tip]) / hand_size(landmarks)


def _reaching(landmarks: Sequence[Point], tip: int, mcp: int) -> bool:
    """True if a fingertip is held out past its knuckle (not curled into a fist)."""
    wrist = landmarks[WRIST]
    return _distance(landmarks[tip], wrist) > _distance(landmarks[mcp], wrist) * 1.2


def classify(landmarks: Sequence[Point]) -> Gesture:
    """Classify a single hand pose."""
    fingers = fingers_extended(landmarks)
    others = (fingers["index"], fingers["middle"], fingers["ring"], fingers["pinky"])

    # A pinch uses whichever fingertip is closer to the thumb. The finger must be
    # held out, so resting the thumb on a closed fist never clicks.
    to_index = pinch_distance(landmarks, INDEX_TIP)
    to_middle = pinch_distance(landmarks, MIDDLE_TIP)
    if to_index < PINCH_RATIO and to_index <= to_middle and _reaching(
        landmarks, INDEX_TIP, INDEX_MCP
    ):
        return Gesture.PINCH
    if to_middle < PINCH_RATIO and to_middle < to_index and _reaching(
        landmarks, MIDDLE_TIP, MIDDLE_MCP
    ):
        return Gesture.PINCH_MIDDLE
    if all(others):
        return Gesture.OPEN_PALM
    if others == (True, False, False, False):
        return Gesture.POINT
    if others == (True, True, False, False):
        return Gesture.SCROLL
    if not any(others):
        thumb_up = landmarks[THUMB_TIP][1] < landmarks[INDEX_MCP][1]
        if fingers["thumb"] and thumb_up:
            return Gesture.THUMBS_UP
        return Gesture.FIST
    return Gesture.NONE
