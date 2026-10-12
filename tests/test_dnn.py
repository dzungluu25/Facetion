"""End-to-end test of the deep-learning backend on real photos.

Downloads the YuNet/SFace models and two sample photos from the OpenCV
repositories; skipped when offline.
"""

import urllib.request

import cv2
import numpy as np
import pytest

from facetion.recognize import run_recognition
from facetion.train import train_model
from facetion.utils import save_names

SAMPLES_URL = "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/"


@pytest.fixture(scope="module")
def photos(tmp_path_factory):
    folder = tmp_path_factory.mktemp("photos")
    paths = {}
    try:
        for name in ("lena.jpg", "messi5.jpg"):
            paths[name] = folder / name
            urllib.request.urlretrieve(SAMPLES_URL + name, paths[name])
        from facetion.model_zoo import ensure_model

        ensure_model("yunet")
        ensure_model("sface")
    except (OSError, SystemExit) as exc:
        pytest.skip(f"network unavailable: {exc}")
    return paths


def _augment(image, n):
    """Make a few slightly different copies of an image, like webcam frames."""
    rng = np.random.default_rng(n)
    shifted = np.roll(image, rng.integers(-5, 5), axis=1)
    return cv2.convertScaleAbs(shifted, alpha=rng.uniform(0.8, 1.2), beta=rng.uniform(-20, 20))


def test_dnn_recognizes_the_registered_person(data_dir, photos, capsys):
    from facetion.backends import get_backend

    lena = cv2.imread(str(photos["lena.jpg"]))
    images_dir = data_dir / "images"
    images_dir.mkdir(parents=True)
    for n in range(1, 4):
        cv2.imwrite(str(images_dir / f"user-0-{n}.jpg"), _augment(lena, n))
    save_names({"0": "Lena"}, data_dir)

    assert train_model(data_dir, "dnn") == 1

    model = get_backend("dnn", data_dir)
    model.load()
    matches = model.identify(lena)
    assert [m.name for m in matches] == ["Lena"]

    # A different person must not be labeled as Lena
    messi = cv2.imread(str(photos["messi5.jpg"]))
    assert all(m.name != "Lena" for m in model.identify(messi))

    # The CLI path: recognize an image file and save the annotated result
    output = data_dir / "out.jpg"
    run_recognition(data_dir, "dnn", str(photos["lena.jpg"]), output=output, show=False)
    assert "Lena" in capsys.readouterr().out
    assert output.exists()
