"""``facetion control``: move the mouse, click and scroll with your hand."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from facetion import config
from facetion.control.controller import ControlSettings, GestureController
from facetion.control.gestures import Gesture, classify
from facetion.control.hands import Hand, HandTracker
from facetion.utils import open_camera

FONT = cv2.FONT_HERSHEY_SIMPLEX
OWNER_CHECK_EVERY = 10    # frames between face checks
OWNER_GRACE_SECONDS = 2.0  # stay unlocked this long after the owner was last seen

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]

HELP = [
    "Point (index up)   move",
    "Pinch thumb+index  click / drag",
    "Pinch thumb+middle right click",
    "Peace sign         scroll",
    "Thumbs up (hold)   play / pause",
    "Fist / open palm   pause",
    "P = pause   Esc = quit",
]


def _draw_hand(frame: np.ndarray, hand: Hand) -> None:
    points = hand.landmarks.astype(int)
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, tuple(points[a]), tuple(points[b]), (255, 255, 255), 2)
    for point in points:
        cv2.circle(frame, tuple(point), 4, (0, 128, 255), -1)


def _draw_status(frame: np.ndarray, status: str, color, gesture: Gesture, action: str) -> None:
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 36), (30, 30, 30), -1)
    cv2.putText(frame, status, (10, 25), FONT, 0.7, color, 2)
    cv2.putText(frame, f"{gesture.value}  {action}", (200, 25), FONT, 0.6, (255, 255, 255), 1)
    for i, line in enumerate(HELP):
        cv2.putText(frame, line, (10, frame.shape[0] - 12 - 20 * (len(HELP) - 1 - i)),
                    FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


class OwnerLock:
    """Only allow control while a specific registered person is in view."""

    def __init__(self, owner: str, data_dir: Path, backend: str):
        from facetion.backends import get_backend

        self.owner = owner
        self.model = get_backend(backend, data_dir)
        self.model.load()
        if owner not in self.model.names.values():
            known = ", ".join(sorted(set(self.model.names.values()))) or "nobody yet"
            raise SystemExit(f"'{owner}' is not registered (registered: {known}).")
        self._last_seen = float("-inf")
        self._frame = 0

    def allows(self, frame: np.ndarray, now: float) -> bool:
        if self._frame % OWNER_CHECK_EVERY == 0:
            if any(m.name == self.owner for m in self.model.identify(frame)):
                self._last_seen = now
        self._frame += 1
        return now - self._last_seen < OWNER_GRACE_SECONDS


def run_control(
    camera_index: int = config.CAMERA_INDEX,
    owner: str | None = None,
    data_dir: Path = config.DEFAULT_DATA_DIR,
    backend: str = config.DEFAULT_BACKEND,
    settings: ControlSettings | None = None,
    dry_run: bool = False,
    show: bool = True,
) -> None:
    if dry_run:
        from facetion.control.actuators import DryRunActuator

        actuator = DryRunActuator(verbose=True)
    else:
        from facetion.control.actuators import PynputActuator

        actuator = PynputActuator()

    controller = GestureController(actuator, settings or ControlSettings())
    tracker = HandTracker()
    lock = OwnerLock(owner, data_dir, backend) if owner else None
    camera = open_camera(camera_index)
    paused = False

    print("[INFO] Hand control started. Show your hand to the camera. Press Esc to quit.")
    if dry_run:
        print("[INFO] Dry run: gestures are shown but the mouse is not moved.")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Failed to read a frame from the camera")
            frame = cv2.flip(frame, 1)  # mirror, so moving right moves the cursor right
            now = time.monotonic()
            height, width = frame.shape[:2]

            owner_ok = lock.allows(frame, now) if lock else True
            hand = tracker.process(frame)
            landmarks = hand.normalized(width, height) if hand else None
            gesture = classify(landmarks) if landmarks else Gesture.NONE

            if paused or not owner_ok:
                controller.release_all()
                controller.update(Gesture.NONE, None, now)
            else:
                controller.update(gesture, landmarks, now)

            if show:
                if hand:
                    _draw_hand(frame, hand)
                if paused:
                    status, color = "PAUSED", (0, 200, 255)
                elif not owner_ok:
                    status, color = f"LOCKED ({lock.owner}?)", (0, 0, 255)
                else:
                    status, color = "ACTIVE", (0, 220, 0)
                _draw_status(frame, status, color, controller.gesture, controller.last_action)
                cv2.imshow("Facetion - hand control", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == config.ESC_KEY:
                break
            if key in (ord("p"), ord("P")):
                paused = not paused
    finally:
        controller.release_all()
        camera.release()
        cv2.destroyAllWindows()

    print("[INFO] Hand control stopped.")
