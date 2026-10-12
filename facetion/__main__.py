"""Command-line interface: ``facetion <command>`` or ``python -m facetion <command>``."""

from __future__ import annotations

import argparse
from pathlib import Path

from facetion import __version__, config
from facetion.backends import BACKENDS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="facetion",
        description="Face recognition and hand-gesture computer control with OpenCV.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--data-dir", type=Path, default=config.DEFAULT_DATA_DIR,
        help="where face images, names and the model are stored (default: %(default)s)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture = subparsers.add_parser("capture", help="register a new person")
    capture.add_argument("--name", help="person's name (prompted if omitted)")
    capture.add_argument("--samples", type=int, default=config.SAMPLES_PER_USER,
                         help="number of face images to take (default: %(default)s)")
    capture.add_argument("--camera", type=int, default=config.CAMERA_INDEX,
                         help="camera index (default: %(default)s)")

    backend_help = "lbph = fast classic model, dnn = more accurate deep-learning model"

    train = subparsers.add_parser("train", help="train the model on captured faces")
    train.add_argument("--backend", choices=BACKENDS, default=config.DEFAULT_BACKEND,
                       help=f"{backend_help} (default: %(default)s)")

    recognize = subparsers.add_parser(
        "recognize", help="recognize faces from the webcam, an image or a video"
    )
    recognize.add_argument("source", nargs="?",
                           help="image or video file (default: use the webcam)")
    recognize.add_argument("--backend", choices=BACKENDS, default=config.DEFAULT_BACKEND,
                           help=f"{backend_help} (default: %(default)s)")
    recognize.add_argument("--camera", type=int, default=config.CAMERA_INDEX,
                           help="camera index (default: %(default)s)")
    recognize.add_argument("--threshold", type=float,
                           help="match threshold (default: 51 for lbph, lower is stricter; "
                                "0.363 for dnn, higher is stricter)")
    recognize.add_argument("--output", "-o", type=Path,
                           help="save the annotated image or video to this file")
    recognize.add_argument("--no-show", action="store_true",
                           help="don't open a window (useful with --output)")

    control = subparsers.add_parser(
        "control", help="control the mouse with hand gestures (needs facetion[control])"
    )
    control.add_argument("--camera", type=int, default=config.CAMERA_INDEX,
                         help="camera index (default: %(default)s)")
    control.add_argument("--owner", metavar="NAME",
                         help="only allow control while this registered person is in view")
    control.add_argument("--backend", choices=BACKENDS, default=config.DEFAULT_BACKEND,
                         help="face backend used by --owner (default: %(default)s)")
    control.add_argument("--smoothing", type=float, default=0.35,
                         help="cursor smoothing, 0.1 = very smooth, 1 = raw (default: %(default)s)")
    control.add_argument("--margin", type=float, default=0.15,
                         help="ignored border of the camera image, so screen corners are "
                              "easy to reach (default: %(default)s)")
    control.add_argument("--dry-run", action="store_true",
                         help="show detected gestures without moving the mouse")

    users = subparsers.add_parser("users", help="list or remove registered people")
    users_sub = users.add_subparsers(dest="users_command", required=True)
    users_sub.add_parser("list", help="list registered people")
    remove = users_sub.add_parser("remove", help="remove a person and their images")
    remove.add_argument("user_id", type=int)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.command == "capture":
        from facetion.capture import capture_faces

        name = args.name or input("\nEnter user name and press <return> --> ").strip()
        if not name:
            raise SystemExit("User name cannot be empty.")
        capture_faces(name, args.data_dir, args.samples, args.camera)

    elif args.command == "train":
        from facetion.train import train_model

        train_model(args.data_dir, args.backend)

    elif args.command == "recognize":
        from facetion.recognize import run_recognition

        run_recognition(
            args.data_dir, args.backend, args.source, args.camera, args.threshold,
            args.output, show=not args.no_show,
        )

    elif args.command == "control":
        from facetion.control.app import run_control
        from facetion.control.controller import ControlSettings

        run_control(
            args.camera, args.owner, args.data_dir, args.backend,
            ControlSettings(smoothing=args.smoothing, margin=args.margin),
            dry_run=args.dry_run,
        )

    elif args.command == "users":
        from facetion.users import list_users, remove_user

        if args.users_command == "list":
            users = list_users(args.data_dir)
            if not users:
                print("No users registered yet. Run 'facetion capture'.")
            for user_id, name in users.items():
                print(f"{user_id:>4}  {name}")
        else:
            removed = remove_user(args.user_id, args.data_dir)
            print(f"Removed user {args.user_id} ({removed} images). "
                  "Run 'facetion train' to update the model.")


if __name__ == "__main__":
    main()
