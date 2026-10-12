"""Command-line entry point: ``python -m facetion <command>``."""

import argparse

from facetion import capture, recognize, train

COMMANDS = {
    "capture": capture.main,
    "train": train.main,
    "recognize": recognize.main,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="facetion", description="Real-time face recognition with OpenCV."
    )
    parser.add_argument("command", choices=COMMANDS, help="step to run")
    args = parser.parse_args()
    COMMANDS[args.command]()


if __name__ == "__main__":
    main()
