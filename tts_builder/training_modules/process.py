from __future__ import annotations

import codecs
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Callable

from PySide6.QtCore import QObject, QTimer, Signal

from tts_builder.gui.settings import TrainingModuleSetting

from .environment import child_environment
from .event_store import EventStore


class ModuleProcessController(QObject):
    event_received = Signal(object)
    stderr_received = Signal(str)
    completed = Signal(int)
    protocol_failed = Signal(str)

    def __init__(
        self,
        setting: TrainingModuleSetting,
        parent=None,
        *,
        popen: Callable[..., object] = subprocess.Popen,
        monotonic: Callable[[], float] = time.monotonic,
        force_kill_grace_seconds: float = 10,
    ) -> None:
        super().__init__(parent)
        self.setting = setting
        self._popen = popen
        self._monotonic = monotonic
        self._grace = force_kill_grace_seconds
        self._process = None
        self._job_dir: Path | None = None
        self._event_store: EventStore | None = None
        self._stdout = None
        self._stderr = None
        self._stderr_path: Path | None = None
        self._stderr_offset = 0
        self._stderr_decoder = codecs.getincrementaldecoder("utf-8")(
            errors="replace"
        )
        self._stop_requested_at: float | None = None
        self._timer = QTimer(self)
        self._timer.setInterval(250)
        self._timer.timeout.connect(self._poll)

    def start(self, job_path: Path) -> None:
        job = Path(job_path).resolve()
        self._launch(job, ("run", "--job", str(job)))

    def promote(self, job_path: Path, selection: str) -> None:
        if not selection:
            raise ValueError("selection must not be empty")
        job = Path(job_path).resolve()
        self._launch(
            job, ("promote", "--job", str(job), "--selection", selection)
        )

    def infer(self, request_path: Path) -> None:
        request = Path(request_path).resolve()
        if not request.is_file():
            raise ValueError("inference request is unavailable")
        job = request.parent.parent.parent / "job.json"
        if request.name != "request.json" or request.parent.parent.name != "inference":
            raise ValueError("inference request path is invalid")
        self._launch(job, ("infer", "--request", str(request)))

    def request_stop(self) -> None:
        if self._job_dir is None:
            raise RuntimeError("no active job")
        marker = self._job_dir / "cancel.requested"
        if not marker.exists():
            temporary = self._job_dir / ".cancel.requested.tmp"
            with temporary.open("x", encoding="utf-8") as stream:
                stream.write("requested\n")
            temporary.replace(marker)
        if self._stop_requested_at is None:
            self._stop_requested_at = self._monotonic()

    def force_kill(self) -> None:
        if self._process is None:
            raise RuntimeError("no active process")
        if self._stop_requested_at is None:
            raise RuntimeError("a stop request is required before force kill")
        if self._monotonic() - self._stop_requested_at < self._grace:
            raise RuntimeError("cooperative cancellation grace period has not elapsed")
        self._process.kill()

    def detach(self) -> None:
        self._timer.stop()
        self._close_logs()
        self._process = None
        self._event_store = None
        self._job_dir = None

    def _launch(self, job_path: Path, action: tuple[str, ...]) -> None:
        if self._process is not None and self._process.poll() is None:
            raise RuntimeError("a module process is already running")
        module_root, python = _validated_setting(self.setting)
        job, project_root, job_id = _job_context(job_path)
        self._job_dir = job.parent
        self._event_store = EventStore(
            self._job_dir / "events.jsonl", project_root, expected_job_id=job_id
        )
        self._stderr_path = _contained_output(self._job_dir, "module.stderr.log")
        stdout_path = _contained_output(self._job_dir, "module.stdout.log")
        self._stderr_offset = (
            self._stderr_path.stat().st_size if self._stderr_path.exists() else 0
        )
        self._stderr_decoder.reset()
        self._stdout = stdout_path.open("ab")
        self._stderr = self._stderr_path.open("ab")
        command = [
            str(python), "-m", self.setting.module_name, "module", *action,
            "--events-jsonl",
        ]
        options = {
            "cwd": module_root,
            "shell": False,
            "stdout": self._stdout,
            "stderr": self._stderr,
            "env": child_environment(),
        }
        if os.name == "nt":
            options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            options["start_new_session"] = True
        try:
            self._process = self._popen(command, **options)
        except Exception:
            self._close_logs()
            self._event_store = None
            self._job_dir = None
            raise
        self._stop_requested_at = None
        self._timer.start()

    def _poll(self) -> None:
        if self._process is None:
            return
        self._drain()
        returncode = self._process.poll()
        if returncode is None:
            return
        self._timer.stop()
        self._drain()
        tail = self._stderr_decoder.decode(b"", final=True)
        if tail:
            self.stderr_received.emit(tail)
        self._close_logs()
        self._process = None
        self.completed.emit(returncode)

    def _drain(self) -> None:
        if self._event_store is not None:
            batch = self._event_store.read_new()
            for event in batch.events:
                self.event_received.emit(event)
            for error in batch.errors:
                self.protocol_failed.emit(error)
        if self._stderr_path is not None and self._stderr_path.exists():
            with self._stderr_path.open("rb") as stream:
                stream.seek(self._stderr_offset)
                chunk = stream.read(64 * 1024)
                if chunk:
                    self._stderr_offset += len(chunk)
                    text = self._stderr_decoder.decode(chunk, final=False)
                    if text:
                        self.stderr_received.emit(text)

    def _close_logs(self) -> None:
        for stream in (self._stdout, self._stderr):
            if stream is not None:
                stream.close()
        self._stdout = None
        self._stderr = None


def _validated_setting(setting: TrainingModuleSetting) -> tuple[Path, Path]:
    project = Path(setting.project_root)
    python = Path(setting.python_executable)
    if not project.is_absolute() or not project.is_dir():
        raise ValueError("module project root is unavailable")
    if not python.is_absolute() or not python.is_file():
        raise ValueError("module Python executable is unavailable")
    return project.resolve(), python.resolve()


def _job_context(job_path: Path) -> tuple[Path, Path, str]:
    job = Path(job_path).resolve()
    if not job.is_file():
        raise ValueError("job path is unavailable")
    try:
        payload = json.loads(job.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid job JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("job JSON must be an object")
    root_value = payload.get("project_root")
    directory_value = payload.get("job_dir")
    job_id = payload.get("job_id")
    if not isinstance(root_value, str) or not Path(root_value).is_absolute():
        raise ValueError("project_root must be absolute")
    if not isinstance(directory_value, str) or not Path(directory_value).is_absolute():
        raise ValueError("job_dir must be absolute")
    if not isinstance(job_id, str) or not job_id:
        raise ValueError("job_id must not be empty")
    project_root = Path(root_value).resolve()
    job_dir = Path(directory_value).resolve()
    if not project_root.is_dir():
        raise ValueError("project_root is unavailable")
    if job_dir != job.parent or (
        job_dir != project_root and not job_dir.is_relative_to(project_root)
    ):
        raise ValueError("job directory must remain inside project_root")
    return job, project_root, job_id


def _contained_output(directory: Path, name: str) -> Path:
    path = (directory / name).resolve()
    if path.parent != directory.resolve():
        raise ValueError("module log path escapes the job directory")
    return path
