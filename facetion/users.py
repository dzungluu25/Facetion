"""Manage registered users."""

from pathlib import Path

from facetion import config
from facetion.utils import load_names, save_names


def list_users(data_dir: Path = config.DEFAULT_DATA_DIR) -> dict[str, str]:
    """Return registered users, sorted by ID."""
    names = load_names(data_dir)
    return dict(sorted(names.items(), key=lambda item: int(item[0])))


def remove_user(user_id: int, data_dir: Path = config.DEFAULT_DATA_DIR) -> int:
    """Remove a user's name and face images; return the number of images deleted."""
    names = load_names(data_dir)
    if str(user_id) not in names:
        raise SystemExit(f"No user with ID {user_id}.")
    del names[str(user_id)]
    save_names(names, data_dir)

    removed = 0
    for image_path in config.images_dir(data_dir).glob(f"user-{user_id}-*.jpg"):
        image_path.unlink()
        removed += 1
    return removed
