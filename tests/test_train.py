import pytest

from facetion.backends import get_backend
from facetion.train import train_model


def test_train_without_samples_exits(data_dir):
    with pytest.raises(SystemExit):
        train_model(data_dir, "lbph")


def test_lbph_train_creates_model_that_predicts_ids(data_dir, make_samples):
    make_samples([0, 1])
    assert train_model(data_dir, "lbph") == 2

    model = get_backend("lbph", data_dir)
    model.load()
    import cv2

    image = cv2.imread(str(data_dir / "images" / "user-1-1.jpg"), cv2.IMREAD_GRAYSCALE)
    user_id, distance = model.recognizer.predict(image)
    assert user_id == 1
    assert distance < model.default_threshold


def test_load_without_model_exits(data_dir):
    with pytest.raises(SystemExit, match="facetion train"):
        get_backend("lbph", data_dir).load()


def test_unknown_backend_raises(data_dir):
    with pytest.raises(ValueError):
        get_backend("nope", data_dir)
