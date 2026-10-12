"""Recognition backends: ``lbph`` (classic, fast) and ``dnn`` (deep learning, accurate)."""

from pathlib import Path

from facetion.backends.base import UNKNOWN_LABEL, Backend, Match

BACKENDS = ("lbph", "dnn")

__all__ = ["BACKENDS", "UNKNOWN_LABEL", "Backend", "Match", "get_backend"]


def get_backend(name: str, data_dir: Path, threshold: float | None = None) -> Backend:
    if name == "lbph":
        from facetion.backends.lbph import LBPHBackend

        return LBPHBackend(data_dir, threshold)
    if name == "dnn":
        from facetion.backends.dnn import DNNBackend

        return DNNBackend(data_dir, threshold)
    raise ValueError(f"Unknown backend {name!r}; choose from {', '.join(BACKENDS)}")
