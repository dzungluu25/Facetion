"""End-to-end test of the OpenCV hand tracker on a real photo (skipped when offline)."""

import urllib.request

import cv2
import pytest

from facetion.control.gestures import Gesture, classify

PHOTO_URL = "https://storage.googleapis.com/mediapipe-tasks/gesture_recognizer/pointing_up.jpg"


@pytest.fixture(scope="module")
def tracker_and_photo(tmp_path_factory):
    path = tmp_path_factory.mktemp("hands") / "pointing_up.jpg"
    try:
        urllib.request.urlretrieve(PHOTO_URL, path)
        from facetion.control.hands import HandTracker

        tracker = HandTracker()
    except (OSError, SystemExit) as exc:
        pytest.skip(f"network unavailable: {exc}")
    return tracker, cv2.imread(str(path))


def test_tracker_finds_pointing_hand(tracker_and_photo):
    tracker, photo = tracker_and_photo
    height, width = photo.shape[:2]
    hand = tracker.process(photo)
    assert hand is not None
    assert classify(hand.normalized(width, height)) == Gesture.POINT

    # Second frame uses tracking instead of palm detection and must agree
    tracked = tracker.process(photo)
    assert tracked is not None
    assert classify(tracked.normalized(width, height)) == Gesture.POINT
