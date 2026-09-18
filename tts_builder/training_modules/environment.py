from __future__ import annotations

import os
from pathlib import Path


_PARENT_ONLY = {"PYTHONPATH", "PYTHONHOME", "HF_HOME", "TORCH_HOME"}


def child_environment(model_root: Path | None = None) -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if key.upper() not in _PARENT_ONLY
    }
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    if model_root is not None:
        root = Path(model_root).resolve()
        environment["HF_HOME"] = str(root / "huggingface")
        environment["TORCH_HOME"] = str(root / "torch")
    return environment
