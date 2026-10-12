import pytest

from facetion.__main__ import main
from facetion.utils import save_names


def test_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "facetion" in capsys.readouterr().out


def test_users_list_empty(data_dir, capsys):
    main(["--data-dir", str(data_dir), "users", "list"])
    assert "No users registered" in capsys.readouterr().out


def test_users_list_and_remove(data_dir, make_samples, capsys):
    save_names({"0": "Joe", "1": "Jane"}, data_dir)
    make_samples([0, 1])

    main(["--data-dir", str(data_dir), "users", "list"])
    out = capsys.readouterr().out
    assert "Joe" in out and "Jane" in out

    main(["--data-dir", str(data_dir), "users", "remove", "0"])
    assert "3 images" in capsys.readouterr().out
    assert not list((data_dir / "images").glob("user-0-*.jpg"))

    main(["--data-dir", str(data_dir), "users", "list"])
    assert "Joe" not in capsys.readouterr().out


def test_remove_unknown_user_exits(data_dir):
    with pytest.raises(SystemExit):
        main(["--data-dir", str(data_dir), "users", "remove", "42"])
