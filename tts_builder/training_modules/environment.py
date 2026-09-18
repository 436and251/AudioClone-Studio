from __future__ import annotations

import os


_PARENT_ONLY = {"PYTHONPATH", "PYTHONHOME", "HF_HOME", "TORCH_HOME"}


def child_environment() -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if key.upper() not in _PARENT_ONLY
    }
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    return environment
