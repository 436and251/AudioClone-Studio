from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence
import uuid

from .models import FrameworkDescriptor, ModuleDescriptor


_STAGES = ("preprocess", "s2", "s1", "evaluate")
_VALUE_KEYS = {
    "project_name", "output_root", "device", "precision", "parameters", "reference",
}


def build_job(
    project_dir: Path,
    module: ModuleDescriptor,
    framework: FrameworkDescriptor,
    values: Mapping[str, object],
    stages: Sequence[str],
    training_data: Path,
) -> Path:
    project_root = _existing_directory(project_dir, "project")
    if module.protocol_version != 2:
        raise ValueError("unsupported module protocol")
    if framework not in module.frameworks:
        raise ValueError("framework does not belong to module")
    training_path = _training_data(training_data, framework, project_root)
    if set(values) != _VALUE_KEYS:
        raise ValueError("job values must contain exactly the supported fields")

    output_root = _contained_path(values["output_root"], project_root, "output")
    selected = tuple(stage for stage in _STAGES if stage in stages)
    if not selected or any(stage not in _STAGES for stage in stages):
        raise ValueError("stages must select supported training stages")

    job_id = str(uuid.uuid4())
    job_dir = project_root / "jobs" / job_id
    job_dir.mkdir(parents=True, exist_ok=False)
    job_path = job_dir / "job.json"
    temporary = job_dir / ".job.json.tmp"
    payload = {
        "protocol_version": 2,
        "job_id": job_id,
        "module_id": module.module_id,
        "project_name": values["project_name"],
        "project_root": str(project_root),
        "output_root": str(output_root),
        "training_data": {
            "path": str(training_path),
            "kind": framework.training_data.kind,
        },
        "framework": framework.id,
        "stages": list(selected),
        "device": values["device"],
        "precision": values["precision"],
        "parameters": values["parameters"],
        "reference": _reference(values["reference"], project_root),
        "job_dir": str(job_dir),
    }
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        temporary.replace(job_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        try:
            job_dir.rmdir()
        except OSError:
            pass
        raise
    return job_path


def _existing_directory(path: Path, name: str) -> Path:
    value = Path(path)
    if not value.is_absolute():
        raise ValueError(f"{name} must be an absolute path")
    resolved = value.resolve()
    if not resolved.is_dir():
        raise ValueError(f"{name} must be an existing directory")
    return resolved


def _contained_path(value: object, root: Path, name: str) -> Path:
    path = Path(value) if isinstance(value, (str, Path)) else Path()
    if not path.is_absolute():
        raise ValueError(f"{name} must be an absolute path")
    resolved = path.resolve()
    if resolved != root and not resolved.is_relative_to(root):
        raise ValueError(f"{name} must remain inside project")
    return resolved


def _contained_file(path: Path, root: Path, name: str) -> Path:
    resolved = _contained_path(path, root, name)
    if not resolved.is_file():
        raise ValueError(f"{name} must be an existing file")
    return resolved


def _training_data(
    path: Path,
    framework: FrameworkDescriptor,
    root: Path,
) -> Path:
    resolved = _contained_path(path, root, "training data")
    descriptor = framework.training_data
    if descriptor.kind == "directory":
        if not resolved.is_dir():
            raise ValueError("training data must be an existing directory")
        return resolved
    if not resolved.is_file():
        raise ValueError("training data must be an existing file")
    extensions = {extension.casefold() for extension in descriptor.extensions}
    if extensions and resolved.suffix.casefold() not in extensions:
        raise ValueError("training data has an unsupported extension")
    return resolved


def _reference(value: object, root: Path) -> object:
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != {"audio", "text", "language"}:
        raise ValueError("reference must contain audio, text, and language")
    return {
        "audio": str(_contained_file(value["audio"], root, "reference audio")),
        "text": value["text"],
        "language": value["language"],
    }
