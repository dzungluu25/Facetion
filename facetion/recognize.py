"""Step 3: recognize faces from the webcam in real time."""

from pathlib import Path

import cv2

from facetion import config
from facetion.utils import load_face_detector, load_names, open_camera

UNKNOWN_LABEL = "Who are you?"
FONT = cv2.FONT_HERSHEY_SIMPLEX


def run_recognition(
    data_dir: Path = config.DEFAULT_DATA_DIR,
    camera_index: int = config.CAMERA_INDEX,
    threshold: float = config.CONFIDENCE_THRESHOLD,
) -> None:
    model_path = config.model_file(data_dir)
    if not model_path.exists():
        raise SystemExit(f"Model not found at {model_path}. Run 'facetion train' first.")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(model_path))
    detector = load_face_detector()
    names = load_names(data_dir)
    camera = open_camera(camera_index)

    # Ignore detections smaller than 10% of the frame
    min_size = (int(0.1 * config.FRAME_WIDTH), int(0.1 * config.FRAME_HEIGHT))

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Failed to read a frame from the camera")

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(
                gray, scaleFactor=1.2, minNeighbors=5, minSize=min_size
            )

            for x, y, w, h in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                user_id, distance = recognizer.predict(gray[y:y + h, x:x + w])

                # LBPH returns a distance: lower means a better match
                if distance < threshold and str(user_id) in names:
                    label = names[str(user_id)]
                    score = f"{round(100 - distance)}%"
                else:
                    label = UNKNOWN_LABEL
                    score = "N/A"

                cv2.putText(frame, label, (x + 5, y - 5), FONT, 1, (255, 255, 255), 2)
                cv2.putText(frame, score, (x + 5, y + h - 5), FONT, 1, (255, 255, 0), 1)

            cv2.imshow("Facetion - recognize", frame)
            if cv2.waitKey(10) & 0xFF == config.ESC_KEY:
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()

    print("\n[INFO] Exiting program.")
