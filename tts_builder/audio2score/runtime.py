from __future__ import annotations

import importlib.util


REQUIRED_MODULES = ("librosa", "basic_pitch", "onnxruntime", "pretty_midi", "resampy")


def audio2score_runtime_status() -> dict[str, str | bool]:
    status: dict[str, str | bool] = {
        name: importlib.util.find_spec(name) is not None for name in REQUIRED_MODULES
    }
    status["ready"] = all(bool(status[name]) for name in REQUIRED_MODULES)
    return status


def require_audio2score_runtime() -> None:
    status = audio2score_runtime_status()
    missing = [name for name in REQUIRED_MODULES if not status[name]]
    if missing:
        raise RuntimeError(
            "Audio2Score dependencies are missing: "
            f"{', '.join(missing)}. Install requirements-audio2score.txt, then run "
            "uv pip install --no-deps basic-pitch==0.4.0"
        )
