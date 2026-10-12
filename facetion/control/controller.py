"""Turn a stream of gestures into mouse and keyboard actions."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol

from facetion.control.gestures import INDEX_MCP, INDEX_TIP, Gesture, Point

# Gestures must be seen this many frames in a row before they take effect,
# so a single misread frame never clicks.
STABLE_FRAMES = 3
THUMBS_UP_HOLD = 0.6  # seconds
THUMBS_UP_COOLDOWN = 1.5
SCROLL_SPEED = 40  # scroll steps per full frame height of movement


class Actuator(Protocol):
    """Something that can move the mouse and press keys."""

    def screen_size(self) -> tuple[int, int]: ...
    def move(self, x: int, y: int) -> None: ...
    def press(self, button: str) -> None: ...
    def release(self, button: str) -> None: ...
    def click(self, button: str) -> None: ...
    def scroll(self, steps: int) -> None: ...
    def play_pause(self) -> None: ...


@dataclass
class ControlSettings:
    smoothing: float = 0.35   # 0 = frozen, 1 = no smoothing
    margin: float = 0.15      # ignore this fraction of the frame edge, so corners are reachable


@dataclass
class GestureController:
    actuator: Actuator
    settings: ControlSettings = field(default_factory=ControlSettings)

    gesture: Gesture = Gesture.NONE        # current stable gesture
    _candidate: Gesture = Gesture.NONE
    _candidate_frames: int = 0
    _cursor: tuple[float, float] | None = None
    _dragging: bool = False
    _scroll_y: float | None = None
    _scroll_rest: float = 0.0
    _thumbs_up_since: float | None = None
    _last_play_pause: float = 0.0
    last_action: str = ""

    def update(self, gesture: Gesture, landmarks: list[Point] | None,
               now: float | None = None) -> None:
        """Feed one frame: the classified gesture and the hand's landmarks (or None)."""
        now = time.monotonic() if now is None else now
        if landmarks is None:
            gesture = Gesture.NONE

        if gesture == self._candidate:
            self._candidate_frames += 1
        else:
            self._candidate, self._candidate_frames = gesture, 1
        if self._candidate_frames >= STABLE_FRAMES and self._candidate != self.gesture:
            self._change_gesture(self.gesture, self._candidate)

        if landmarks is not None:
            self._act(landmarks, now)

    def release_all(self) -> None:
        if self._dragging:
            self.actuator.release("left")
            self._dragging = False

    # -- internals --------------------------------------------------------

    def _change_gesture(self, old: Gesture, new: Gesture) -> None:
        self.gesture = new
        if old == Gesture.PINCH and self._dragging:
            self.actuator.release("left")
            self._dragging = False
            self.last_action = "release"
        if new == Gesture.PINCH:
            self.actuator.press("left")
            self._dragging = True
            self.last_action = "click"
        elif new == Gesture.PINCH_MIDDLE:
            self.actuator.click("right")
            self.last_action = "right click"
        if new != Gesture.SCROLL:
            self._scroll_y = None
            self._scroll_rest = 0.0
        if new != Gesture.THUMBS_UP:
            self._thumbs_up_since = None

    def _act(self, landmarks: list[Point], now: float) -> None:
        if self.gesture == Gesture.POINT:
            self._move_cursor(landmarks[INDEX_TIP])
        elif self.gesture == Gesture.PINCH:
            # Follow the knuckle while pinching: it stays still as the fingers close
            self._move_cursor(landmarks[INDEX_MCP])
        elif self.gesture == Gesture.SCROLL:
            self._scroll(landmarks[INDEX_TIP][1])
        elif self.gesture == Gesture.THUMBS_UP:
            if self._thumbs_up_since is None:
                self._thumbs_up_since = now
            held = now - self._thumbs_up_since >= THUMBS_UP_HOLD
            if held and now - self._last_play_pause >= THUMBS_UP_COOLDOWN:
                self.actuator.play_pause()
                self._last_play_pause = now
                self.last_action = "play/pause"

    def _move_cursor(self, point: Point) -> None:
        width, height = self.actuator.screen_size()
        margin = self.settings.margin
        span = 1 - 2 * margin

        def to_screen(value: float, size: int) -> float:
            fraction = min(max((value - margin) / span, 0.0), 1.0)
            return fraction * (size - 1)

        target = (to_screen(point[0], width), to_screen(point[1], height))
        if self._cursor is None:
            self._cursor = target
        else:
            alpha = self.settings.smoothing
            self._cursor = (
                self._cursor[0] + alpha * (target[0] - self._cursor[0]),
                self._cursor[1] + alpha * (target[1] - self._cursor[1]),
            )
        self.actuator.move(round(self._cursor[0]), round(self._cursor[1]))

    def _scroll(self, y: float) -> None:
        if self._scroll_y is not None:
            # Moving the hand up scrolls up, like dragging a page on a phone
            self._scroll_rest += (self._scroll_y - y) * SCROLL_SPEED
            steps = int(self._scroll_rest)
            if steps:
                self.actuator.scroll(steps)
                self._scroll_rest -= steps
                self.last_action = "scroll"
        self._scroll_y = y
