import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

from tts_builder.gui.settings import TrainingModuleSetting
from tts_builder.training_modules.probe import probe_module


def descriptor_payload() -> dict:
    return {
        "protocol_version": 2,
        "module_id": "gpt-sovits-v2proplus",
        "module_version": "0.1.0",
        "frameworks": [
            {
                "id": "v2ProPlus",
                "display_name": "GPT-SoVITS v2ProPlus",
                "capabilities": ["train"],
                "training_data": {"kind": "file", "extensions": [".list"]},
                "fields": [],
            }
        ],
    }


def _setting(tmp_path: Path) -> TrainingModuleSetting:
    project = tmp_path / "voice-pipeline"
    project.mkdir()
    python = tmp_path / "venv" / "python.exe"
    python.parent.mkdir()
    python.write_bytes(b"")
    return TrainingModuleSetting("GPT-SoVITS", project, python, "voice_pipeline")


def test_probe_uses_only_explicit_module_configuration(tmp_path: Path, monkeypatch):
    setting = _setting(tmp_path)
    calls = []

    def run(command, **options):
        calls.append((command, options))
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(descriptor_payload()),
            stderr="",
        )

    for key in ("PYTHONPATH", "PYTHONHOME", "HF_HOME", "TORCH_HOME"):
        monkeypatch.setenv(key, "parent-only")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    result = probe_module(setting, run=run)

    assert result.available is True
    assert result.descriptor.module_id == "gpt-sovits-v2proplus"
    assert calls == [
        (
            [
                str(setting.python_executable),
                "-m",
                "voice_pipeline",
                "module",
                "describe",
                "--json",
            ],
            {
                "cwd": setting.project_root,
                "shell": False,
                "capture_output": True,
                "text": True,
                "encoding": "utf-8",
                "timeout": 10,
                "env": {
                    **{
                        key: value
                        for key, value in os.environ.items()
                        if key not in {"PYTHONPATH", "PYTHONHOME", "HF_HOME", "TORCH_HOME"}
                    },
                    "PYTHONNOUSERSITE": "1",
                },
            },
        )
    ]


def test_probe_rejects_missing_explicit_paths_without_starting_process(tmp_path: Path):
    setting = TrainingModuleSetting(
        "GPT-SoVITS",
        tmp_path / "missing-project",
        tmp_path / "missing-python.exe",
        "voice_pipeline",
    )

    result = probe_module(
        setting,
        run=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("process must not start")
        ),
    )

    assert result.available is False
    assert result.descriptor is None


def test_probe_converts_process_and_protocol_failures_to_unavailable(tmp_path: Path):
    setting = _setting(tmp_path)
    failures = [
        subprocess.TimeoutExpired("voice_pipeline", 10),
        OSError("cannot execute"),
        SimpleNamespace(returncode=3, stdout="", stderr="failed"),
        SimpleNamespace(returncode=0, stdout="not json", stderr=""),
    ]

    for failure in failures:
        def run(*_args, failure=failure, **_kwargs):
            if isinstance(failure, BaseException):
                raise failure
            return failure

        result = probe_module(setting, run=run)
        assert result.available is False
        assert result.descriptor is None
        assert result.error
