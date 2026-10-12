"""Mouse/keyboard backends: real (pynput) and dry-run (prints only)."""

from __future__ import annotations

import sys


def detect_screen_size() -> tuple[int, int]:
    """Return the main screen size in the coordinates the mouse uses."""
    if sys.platform == "darwin":
        import Quartz  # installed with pynput on macOS

        bounds = Quartz.CGDisplayBounds(Quartz.CGMainDisplayID())
        return int(bounds.size.width), int(bounds.size.height)
    if sys.platform == "win32":
        import ctypes

        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()
        return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    try:
        from Xlib import display  # installed with pynput on Linux

        screen = display.Display().screen()
        return screen.width_in_pixels, screen.height_in_pixels
    except Exception:
        import tkinter

        root = tkinter.Tk()
        root.withdraw()
        size = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        return size


class PynputActuator:
    """Controls the real mouse and keyboard."""

    def __init__(self) -> None:
        try:
            from pynput import keyboard, mouse
        except ImportError as exc:
            raise SystemExit(
                "Hand control needs extra packages. Install them with:\n"
                "  pip install \"facetion[control]\""
            ) from exc
        self._mouse = mouse.Controller()
        self._keyboard = keyboard.Controller()
        self._buttons = {"left": mouse.Button.left, "right": mouse.Button.right}
        self._play_pause_key = keyboard.Key.media_play_pause
        self._size = detect_screen_size()

    def screen_size(self) -> tuple[int, int]:
        return self._size

    def move(self, x: int, y: int) -> None:
        self._mouse.position = (x, y)

    def press(self, button: str) -> None:
        self._mouse.press(self._buttons[button])

    def release(self, button: str) -> None:
        self._mouse.release(self._buttons[button])

    def click(self, button: str) -> None:
        self._mouse.click(self._buttons[button])

    def scroll(self, steps: int) -> None:
        self._mouse.scroll(0, steps)

    def play_pause(self) -> None:
        self._keyboard.press(self._play_pause_key)
        self._keyboard.release(self._play_pause_key)


class DryRunActuator:
    """Records actions instead of performing them (for --dry-run and tests)."""

    def __init__(self, size: tuple[int, int] = (1920, 1080), verbose: bool = False):
        self._size = size
        self.verbose = verbose
        self.position = (0, 0)
        self.events: list[tuple] = []

    def screen_size(self) -> tuple[int, int]:
        return self._size

    def _record(self, *event) -> None:
        self.events.append(event)
        if self.verbose:
            print("[dry-run]", *event)

    def move(self, x: int, y: int) -> None:
        self.position = (x, y)

    def press(self, button: str) -> None:
        self._record("press", button)

    def release(self, button: str) -> None:
        self._record("release", button)

    def click(self, button: str) -> None:
        self._record("click", button)

    def scroll(self, steps: int) -> None:
        self._record("scroll", steps)

    def play_pause(self) -> None:
        self._record("play_pause")
