import cv2
import pytest

from facetion import config
from facetion.train import train_model


def test_train_without_samples_exits(data_dir):
    with pytest.raises(SystemExit):
        train_model(data_dir)


def test_train_creates_model_that_predicts_ids(data_dir, make_samples):
    make_samples([0, 1])
    assert train_model(data_dir) == 2

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(config.model_file(data_dir)))
    image = cv2.imread(str(data_dir / "images" / "user-1-1.jpg"), cv2.IMREAD_GRAYSCALE)
    user_id, distance = recognizer.predict(image)
    assert user_id == 1
    assert distance < config.CONFIDENCE_THRESHOLD
