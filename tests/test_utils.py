from pathlib import Path

from facetion.capture import next_user_id
from facetion.utils import load_face_detector, load_names, parse_user_id, save_names


def test_load_names_missing_file_returns_empty(data_dir):
    assert load_names(data_dir) == {}


def test_save_and_load_names_round_trip(data_dir):
    save_names({"0": "Joe", "1": "Zoë"}, data_dir)
    assert load_names(data_dir) == {"0": "Joe", "1": "Zoë"}


def test_parse_user_id():
    assert parse_user_id(Path("user-12-3.jpg")) == 12


def test_next_user_id_fills_gaps():
    assert next_user_id({}) == 0
    assert next_user_id({"0": "a", "1": "b"}) == 2
    assert next_user_id({"0": "a", "2": "b"}) == 1


def test_bundled_face_detector_loads():
    assert not load_face_detector().empty()
