"""Step 1: capture face samples for a new user from the webcam."""

import cv2

from facetion import config
from facetion.utils import (
    load_face_detector,
    load_names,
    open_camera,
    save_names,
)


def next_user_id(names: dict[str, str]) -> int:
    """Return the smallest unused user ID."""
    used = {int(key) for key in names}
    user_id = 0
    while user_id in used:
        user_id += 1
    return user_id


def capture_faces(user_name: str) -> int:
    """Save face samples for ``user_name`` and return the assigned user ID."""
    config.IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    names = load_names()
    user_id = next_user_id(names)
    names[str(user_id)] = user_name
    save_names(names)

    detector = load_face_detector()
    camera = open_camera()
    print("\n[INFO] Initializing face capture. Look at the camera and wait...")

    count = 0
    try:
        while count < config.SAMPLES_PER_USER:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Failed to read a frame from the camera")

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for x, y, w, h in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                count += 1
                image_path = config.IMAGES_DIR / f"user-{user_id}-{count}.jpg"
                cv2.imwrite(str(image_path), gray[y:y + h, x:x + w])

            cv2.imshow("Facetion - capture", frame)
            if cv2.waitKey(200) & 0xFF == config.ESC_KEY:
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()

    print(f"\n[INFO] Saved {count} samples for '{user_name}' (ID {user_id}).")
    return user_id


def main() -> None:
    user_name = input("\nEnter user name and press <return> --> ").strip()
    if not user_name:
        raise SystemExit("User name cannot be empty.")
    capture_faces(user_name)


if __name__ == "__main__":
    main()
