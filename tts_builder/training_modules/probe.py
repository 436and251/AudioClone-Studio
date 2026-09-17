from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
from typing import Callable

from tts_builder.gui.settings import TrainingModuleSetting

from .models import ProbeResult
from .protocol import parse_descriptor


_MODULE_NAME = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*")


def probe_module(
    setting: TrainingModuleSetting,
    *,
    timeout_seconds: float = 10,
    run: Callable[..., object] = subprocess.run,
) -> ProbeResult:
    error = _setting_error(setting)
    if error:
        return ProbeResult(False, error=error)
    command = [
        str(setting.python_executable),
        "-m",
        setting.module_name,
        "module",
        "describe",
        "--json",
    ]
    try:
        completed = run(
            command,
            cwd=setting.project_root,
            shell=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout_seconds,
        )
        if completed.returncode != 0:
            return ProbeResult(False, error=f"module exited with code {completed.returncode}")
        descriptor = parse_descriptor(json.loads(completed.stdout))
        return ProbeResult(True, descriptor=descriptor)
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError, ValueError) as error:
        return ProbeResult(False, error=str(error))


def _setting_error(setting: TrainingModuleSetting) -> str | None:
    project_root = Path(setting.project_root)
    python = Path(setting.python_executable)
    checks = (
        (not project_root.is_absolute() or not project_root.is_dir(), "module project root is unavailable"),
        (not python.is_absolute() or not python.is_file(), "module Python executable is unavailable"),
        (_MODULE_NAME.fullmatch(setting.module_name) is None, "module name is invalid"),
    )
    return next((message for failed, message in checks if failed), None)
