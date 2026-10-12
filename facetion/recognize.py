"""Step 3: recognize faces in a webcam stream, video file or image."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from facetion import config
from facetion.backends import Backend, Match, get_backend
from facetion.utils import open_camera

FONT = cv2.FONT_HERSHEY_SIMPLEX
KNOWN_COLOR = (0, 200, 0)
UNKNOWN_COLOR = (0, 0, 230)


def annotate(frame: np.ndarray, matches: list[Match]) -> np.ndarray:
    """Draw a box, name and score for each match onto the frame."""
    for match in matches:
        x, y, w, h = match.box
        color = KNOWN_COLOR if match.score is not None else UNKNOWN_COLOR
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        label = match.name if match.score is None else f"{match.name} {round(match.score)}%"
        (text_w, text_h), _ = cv2.getTextSize(label, FONT, 0.7, 2)
        top = max(y - text_h - 10, 0)
        cv2.rectangle(frame, (x, top), (x + text_w + 10, top + text_h + 10), color, -1)
        cv2.putText(frame, label, (x + 5, top + text_h + 4), FONT, 0.7, (255, 255, 255), 2)
    return frame


def recognize_image(
    model: Backend, image_path: Path, output: Path | None = None, show: bool = True
) -> list[Match]:
    """Identify faces in a single image file."""
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise SystemExit(f"Could not read image {image_path}")

    matches = model.identify(frame)
    for match in matches:
        print(f"{match.name}" + ("" if match.score is None else f" ({round(match.score)}%)"))
    if not matches:
        print("No faces found.")

    annotate(frame, matches)
    if output:
        cv2.imwrite(str(output), frame)
        print(f"[INFO] Saved annotated image to {output}")
    if show:
        cv2.imshow("Facetion - recognize", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return matches


def recognize_stream(
    model: Backend,
    capture: cv2.VideoCapture,
    output: Path | None = None,
    show: bool = True,
) -> None:
    """Identify faces frame by frame in a webcam or video stream."""
    writer = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            annotate(frame, model.identify(frame))

            if output:
                if writer is None:
                    fps = capture.get(cv2.CAP_PROP_FPS) or 25
                    height, width = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(output), fourcc, fps, (width, height))
                writer.write(frame)

            if show:
                cv2.imshow("Facetion - recognize", frame)
                if cv2.waitKey(1) & 0xFF == config.ESC_KEY:
                    break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
            print(f"[INFO] Saved annotated video to {output}")
        cv2.destroyAllWindows()


def run_recognition(
    data_dir: Path = config.DEFAULT_DATA_DIR,
    backend: str = config.DEFAULT_BACKEND,
    source: str | None = None,
    camera_index: int = config.CAMERA_INDEX,
    threshold: float | None = None,
    output: Path | None = None,
    show: bool = True,
) -> None:
    """Recognize faces from the webcam, or from an image or video file if ``source`` is set."""
    model = get_backend(backend, data_dir, threshold)
    model.load()

    if source is None:
        recognize_stream(model, open_camera(camera_index), output, show)
        return

    path = Path(source)
    if not path.exists():
        raise SystemExit(f"File not found: {path}")
    if path.suffix.lower() in config.IMAGE_EXTENSIONS:
        recognize_image(model, path, output, show)
    else:
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise SystemExit(f"Could not open video {path}")
        recognize_stream(model, capture, output, show)
