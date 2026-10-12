import numpy as np

from facetion.backends import Match
from facetion.capture import padded_crop
from facetion.recognize import annotate


def test_padded_crop_adds_margin_and_clips_to_frame():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    assert padded_crop(frame, (40, 40, 20, 20), padding=0.25).shape[:2] == (30, 30)
    assert padded_crop(frame, (0, 0, 20, 20), padding=0.5).shape[:2] == (30, 30)


def test_annotate_draws_on_frame():
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    annotate(frame, [Match((50, 50, 60, 60), "Joe", 87.0), Match((10, 120, 40, 40), "?", None)])
    assert frame.any()
