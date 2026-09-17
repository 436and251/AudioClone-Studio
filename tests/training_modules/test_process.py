import json
import os
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from tts_builder.gui.app import create_application
from tts_builder.gui.settings import TrainingModuleSetting


class FakeProcess:
    def __init__(self):
        self.killed = False
        self.returncode = None

    def poll(self):
        return self.returncode

    def kill(self):
        self.killed = True


def _setting(tmp_path: Path):
    module_root = tmp_path / "voice-pipeline"
    module_root.mkdir()
    python = tmp_path / "venv" / "python.exe"
    python.parent.mkdir()
    python.write_bytes(b"")
    return TrainingModuleSetting("GPT-SoVITS", module_root, python, "voice_pipeline")


def _job(tmp_path: Path):
    project = tmp_path / "Acane"
    job_dir = project / "jobs" / "job-1"
    job_dir.mkdir(parents=True)
    path = job_dir / "job.json"
    path.write_text(
        json.dumps(
            {
                "protocol_version": 1,
                "job_id": "job-1",
                "project_root": str(project.resolve()),
                "job_dir": str(job_dir.resolve()),
            }
        ),
        encoding="utf-8",
    )
    return path


def test_start_uses_explicit_interpreter_safe_paths_and_isolated_environment(
    tmp_path: Path, monkeypatch
):
    from tts_builder.training_modules.process import ModuleProcessController

    create_application([])
    setting = _setting(tmp_path)
    job = _job(tmp_path)
    calls = []
    process = FakeProcess()

    def popen(command, **options):
        calls.append((command, options))
        return process

    for key in ("PYTHONPATH", "PYTHONHOME", "HF_HOME", "TORCH_HOME"):
        monkeypatch.setenv(key, "parent-only")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    controller = ModuleProcessController(setting, popen=popen)
    controller.start(job)

    command, options = calls[0]
    assert command == [
        str(setting.python_executable.resolve()), "-m", "voice_pipeline",
        "module", "run", "--job", str(job.resolve()), "--events-jsonl",
    ]
    assert options["cwd"] == setting.project_root.resolve()
    assert options["shell"] is False
    assert Path(options["stdout"].name).parent == job.parent.resolve()
    assert Path(options["stderr"].name).parent == job.parent.resolve()
    assert options["env"]["CUDA_VISIBLE_DEVICES"] == "0"
    assert options["env"]["PYTHONNOUSERSITE"] == "1"
    for key in ("PYTHONPATH", "PYTHONHOME", "HF_HOME", "TORCH_HOME"):
        assert key not in options["env"]
    if os.name == "nt":
        assert options["creationflags"] != 0
        assert "start_new_session" not in options
    else:
        assert options["start_new_session"] is True
    assert controller._timer.interval() == 250
    controller.detach()


def test_promote_uses_explicit_selection(tmp_path: Path):
    from tts_builder.training_modules.process import ModuleProcessController

    create_application([])
    setting = _setting(tmp_path)
    job = _job(tmp_path)
    calls = []
    controller = ModuleProcessController(
        setting,
        popen=lambda command, **options: calls.append((command, options)) or FakeProcess(),
    )

    controller.promote(job, "candidate_A")

    assert calls[0][0][-7:] == [
        "module", "promote", "--job", str(job.resolve()),
        "--selection", "candidate_A", "--events-jsonl",
    ]
    controller.detach()


def test_cooperative_stop_force_kill_grace_and_detach(tmp_path: Path):
    from tts_builder.training_modules.process import ModuleProcessController

    create_application([])
    setting = _setting(tmp_path)
    job = _job(tmp_path)
    now = [100.0]
    process = FakeProcess()
    controller = ModuleProcessController(
        setting,
        popen=lambda *_args, **_kwargs: process,
        monotonic=lambda: now[0],
        force_kill_grace_seconds=5,
    )
    controller.start(job)

    with pytest.raises(RuntimeError, match="stop request"):
        controller.force_kill()
    controller.request_stop()
    assert (job.parent / "cancel.requested").read_text(encoding="utf-8") == "requested\n"
    assert process.killed is False
    with pytest.raises(RuntimeError, match="grace"):
        controller.force_kill()
    now[0] += 5
    controller.force_kill()
    assert process.killed is True

    process.killed = False
    controller.detach()
    assert process.killed is False


def test_poll_emits_journal_stderr_and_completion(tmp_path: Path):
    from tts_builder.training_modules.process import ModuleProcessController

    create_application([])
    setting = _setting(tmp_path)
    job = _job(tmp_path)
    process = FakeProcess()
    controller = ModuleProcessController(
        setting, popen=lambda *_args, **_kwargs: process
    )
    events, diagnostics, completed = [], [], []
    controller.event_received.connect(events.append)
    controller.stderr_received.connect(diagnostics.append)
    controller.completed.connect(completed.append)
    controller.start(job)
    (job.parent / "events.jsonl").write_text(
        json.dumps(
            {
                "protocol_version": 1,
                "job_id": "job-1",
                "type": "job_completed",
                "timestamp": "2026-09-17T00:00:00+00:00",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with (job.parent / "module.stderr.log").open("ab") as stream:
        stream.write("训练完成\n".encode("utf-8"))
    process.returncode = 0

    controller._poll()

    assert [event.type for event in events] == ["job_completed"]
    assert "".join(diagnostics) == "训练完成\n"
    assert completed == [0]


def test_poll_bounds_stderr_work_and_reports_protocol_errors(tmp_path: Path):
    from tts_builder.training_modules.process import ModuleProcessController

    create_application([])
    setting = _setting(tmp_path)
    job = _job(tmp_path)
    process = FakeProcess()
    controller = ModuleProcessController(
        setting, popen=lambda *_args, **_kwargs: process
    )
    diagnostics, failures = [], []
    controller.stderr_received.connect(diagnostics.append)
    controller.protocol_failed.connect(failures.append)
    controller.start(job)
    with (job.parent / "module.stderr.log").open("ab") as stream:
        stream.write(b"x" * (70 * 1024))
    (job.parent / "events.jsonl").write_text("not-json\n", encoding="utf-8")

    controller._poll()

    assert len("".join(diagnostics)) == 64 * 1024
    assert len(failures) == 1
    controller.detach()
