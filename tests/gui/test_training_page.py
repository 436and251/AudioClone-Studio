import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QObject, Signal

from tts_builder.gui.app import create_application
from tts_builder.gui.i18n import LocaleController
from tts_builder.gui.settings import AppSettings, TrainingModuleSetting
from tts_builder.training_modules.models import FrameworkDescriptor, ModuleDescriptor, ModuleEvent


class FakeModuleProcess(QObject):
    event_received = Signal(object)
    stderr_received = Signal(str)
    completed = Signal(int)
    protocol_failed = Signal(str)

    def __init__(self):
        super().__init__()
        self.started = []
        self.stopped = False

    def start(self, path):
        self.started.append(path)

    def request_stop(self):
        self.stopped = True


class FakeDatasetController(QObject):
    event_received = Signal(object)
    task_completed = Signal(object)
    task_failed = Signal(str, str, str)
    task_cancelled = Signal()
    running_changed = Signal(bool)

    def __init__(self):
        super().__init__()
        self.running = False

    def start(self, *_):
        pass

    def stop(self):
        pass


def _binding(tmp_path: Path):
    module_root = tmp_path / "module"
    python = tmp_path / "venv" / "python.exe"
    module_root.mkdir()
    python.parent.mkdir()
    python.write_bytes(b"")
    setting = TrainingModuleSetting("GPT-SoVITS", module_root, python, "voice_pipeline")
    framework = FrameworkDescriptor(
        "v2ProPlus", "GPT-SoVITS v2ProPlus",
        ("preprocess", "train", "evaluate", "listen", "promote"), (),
    )
    return setting, ModuleDescriptor(1, "gpt-sovits", "1.0", (framework,))


def _project(tmp_path: Path):
    project = tmp_path / "Acane"
    dataset = project / "dataset.list"
    project.mkdir()
    dataset.write_text("clip.wav|Acane|ja|test\n", encoding="utf-8")
    return project, dataset


def test_training_page_builds_job_and_starts_selected_module(tmp_path):
    from tts_builder.gui.training_page import TrainingPage

    app = create_application([])
    process = FakeModuleProcess()
    calls = []
    project, dataset = _project(tmp_path)
    job = project / "jobs" / "job-1" / "job.json"
    job.parent.mkdir(parents=True)
    job.write_text("{}", encoding="utf-8")

    def build(*args):
        calls.append(args)
        return job

    page = TrainingPage(
        (_binding(tmp_path),), LocaleController("en"),
        build_job=build, process_factory=lambda _setting: process,
    )
    page.prefill_dataset(dataset)
    reference = project / "reference.wav"
    reference.write_bytes(b"wav")
    page.form.reference_audio.setText(str(reference))
    app.processEvents()
    assert page.start_button.isEnabled()

    page.start_button.click()

    assert process.started == [job]
    assert calls[0][0] == project.resolve()
    assert calls[0][4] == ("preprocess", "s2", "s1", "evaluate")
    assert calls[0][5] == dataset.resolve()
    page.close()


def test_training_progress_and_errors_are_immediately_visible_and_bounded(tmp_path):
    from tts_builder.gui.training_page import TrainingPage

    app = create_application([])
    process = FakeModuleProcess()
    page = TrainingPage(
        (_binding(tmp_path),), LocaleController("zh_CN"),
        process_factory=lambda _setting: process,
    )
    page.attach_process(process)
    process.event_received.emit(ModuleEvent(
        1, "job", "stage_started", "now", stage="s2"
    ))
    process.event_received.emit(ModuleEvent(
        1, "job", "stage_progress", "now", stage="s2", current=25, total=100
    ))
    app.processEvents()
    assert page.progress.rows["s2"].bar.value() == 25
    assert page.progress.rows["s2"].status.text() == "进行中"

    process.event_received.emit(ModuleEvent(
        1, "job", "stage_failed", "now", stage="s2",
        message_key="pipeline.stage_failed", message_args={"error": "CUDA OOM"},
    ))
    process.stderr_received.emit("CUDA out of memory")
    app.processEvents()
    assert page.error_summary.isVisibleTo(page)
    assert "CUDA OOM" in page.error_summary.text()
    assert "CUDA out of memory" in page.activity.toPlainText()
    assert page.activity.maximumBlockCount() == 500

    process.event_received.emit(ModuleEvent(
        1, "job", "future_event", "now", message_key="module.future"
    ))
    app.processEvents()
    assert "future_event" in page.activity.toPlainText()
    page.close()


def test_dataset_completion_offers_explicit_continue_and_prefills_training(tmp_path):
    from tts_builder.gui.studio_window import StudioWindow

    app = create_application([])
    binding = _binding(tmp_path)
    output = tmp_path / "Acane"
    output.mkdir()
    dataset = output / "dataset.list"
    dataset.write_text("clip.wav|Acane|ja|test\n", encoding="utf-8")
    settings = AppSettings(
        first_run_completed=True,
        model_root=tmp_path / "models",
        output_root=output,
        training_modules=(binding[0],),
    )
    controller = FakeDatasetController()
    window = StudioWindow(settings, (binding,), controller)
    controller.task_completed.emit(SimpleNamespace(accepted=1, rejected=0))
    app.processEvents()

    assert window.dataset_page.continue_training.isVisibleTo(window.dataset_page)
    window.dataset_page.continue_training.click()
    app.processEvents()

    assert window.pages.currentIndex() == 1
    assert Path(window.training_page.form.dataset_edit.text()) == dataset.resolve()
    window.close()
