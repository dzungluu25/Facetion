import json
from pathlib import Path

import numpy as np
import pytest

from facetion.control.actuators import DryRunActuator
from facetion.control.controller import STABLE_FRAMES, ControlSettings, GestureController
from facetion.control.gestures import (
    INDEX_TIP,
    MIDDLE_TIP,
    THUMB_TIP,
    Gesture,
    classify,
)
from facetion.control.hands import _palm_anchors

# Landmarks detected on real photos (from MediaPipe's sample images)
HANDS = json.loads((Path(__file__).parent / "fixtures" / "hands.json").read_text())


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("woman_hands_0", Gesture.OPEN_PALM),
        ("woman_hands_1", Gesture.OPEN_PALM),
        ("thumbs_up_0", Gesture.THUMBS_UP),
        ("pointing_up_0", Gesture.POINT),
        ("victory_0", Gesture.SCROLL),
    ],
)
def test_classify_real_hands(name, expected):
    assert classify(HANDS[name]) == expected


def test_classify_pinch():
    hand = [list(p) for p in HANDS["pointing_up_0"]]
    hand[THUMB_TIP] = list(hand[INDEX_TIP])
    assert classify(hand) == Gesture.PINCH


def test_classify_pinch_middle():
    hand = [list(p) for p in HANDS["victory_0"]]
    hand[THUMB_TIP] = list(hand[MIDDLE_TIP])
    assert classify(hand) == Gesture.PINCH_MIDDLE


def test_thumb_resting_on_fist_is_not_a_click():
    hand = [list(p) for p in HANDS["thumbs_up_0"]]
    for tip in (INDEX_TIP, MIDDLE_TIP):
        hand[THUMB_TIP] = list(hand[tip])  # thumb touches a curled finger
        assert classify(hand) == Gesture.FIST


def test_palm_anchor_count():
    assert _palm_anchors().shape == (2016, 2)


def _feed(controller, gesture, landmarks, frames, start=0.0):
    for i in range(frames):
        controller.update(gesture, landmarks, now=start + i / 30)


def _hand_at(x, y):
    hand = np.tile([x, y], (21, 1)).tolist()
    return hand


def test_point_moves_cursor_across_whole_screen():
    actuator = DryRunActuator(size=(1000, 500))
    controller = GestureController(actuator, ControlSettings(smoothing=1.0, margin=0.1))
    _feed(controller, Gesture.POINT, _hand_at(0.5, 0.5), STABLE_FRAMES + 1)
    assert actuator.position == (500, 250)
    _feed(controller, Gesture.POINT, _hand_at(0.95, 0.02), 2)
    assert actuator.position == (999, 0)


def test_single_misread_frame_does_not_click():
    actuator = DryRunActuator()
    controller = GestureController(actuator)
    _feed(controller, Gesture.POINT, _hand_at(0.5, 0.5), 5)
    controller.update(Gesture.PINCH, _hand_at(0.5, 0.5))
    _feed(controller, Gesture.POINT, _hand_at(0.5, 0.5), 5)
    assert actuator.events == []


def test_pinch_presses_and_releases():
    actuator = DryRunActuator()
    controller = GestureController(actuator)
    _feed(controller, Gesture.PINCH, _hand_at(0.5, 0.5), STABLE_FRAMES)
    assert actuator.events == [("press", "left")]
    _feed(controller, Gesture.POINT, _hand_at(0.5, 0.5), STABLE_FRAMES)
    assert actuator.events == [("press", "left"), ("release", "left")]


def test_release_all_lets_go_of_drag():
    actuator = DryRunActuator()
    controller = GestureController(actuator)
    _feed(controller, Gesture.PINCH, _hand_at(0.5, 0.5), STABLE_FRAMES)
    controller.release_all()
    assert actuator.events[-1] == ("release", "left")


def test_pinch_middle_right_clicks_once():
    actuator = DryRunActuator()
    controller = GestureController(actuator)
    _feed(controller, Gesture.PINCH_MIDDLE, _hand_at(0.5, 0.5), 10)
    assert actuator.events == [("click", "right")]


def test_scroll_follows_hand_movement():
    actuator = DryRunActuator()
    controller = GestureController(actuator)
    _feed(controller, Gesture.SCROLL, _hand_at(0.5, 0.5), STABLE_FRAMES)
    controller.update(Gesture.SCROLL, _hand_at(0.5, 0.4))  # hand moves up
    assert actuator.events and actuator.events[-1][0] == "scroll"
    assert actuator.events[-1][1] > 0


def test_thumbs_up_must_be_held_and_has_cooldown():
    actuator = DryRunActuator()
    controller = GestureController(actuator)
    hand = _hand_at(0.5, 0.5)
    _feed(controller, Gesture.THUMBS_UP, hand, 5)  # ~0.17 s: too short
    assert actuator.events == []
    _feed(controller, Gesture.THUMBS_UP, hand, 60, start=5 / 30)  # 2 s more
    assert actuator.events.count(("play_pause",)) == 1


def test_lost_hand_releases_drag():
    actuator = DryRunActuator()
    controller = GestureController(actuator)
    _feed(controller, Gesture.PINCH, _hand_at(0.5, 0.5), STABLE_FRAMES)
    _feed(controller, Gesture.NONE, None, STABLE_FRAMES)
    assert actuator.events[-1] == ("release", "left")
