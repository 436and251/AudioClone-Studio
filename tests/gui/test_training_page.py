import os
import hashlib
import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QMessageBox

from tts_builder.gui.app import create_application
from tts_builder.gui.i18n import LocaleController
from tts_builder.gui.settings import AppSettings, TrainingModuleSetting
from tts_builder.training_modules.models import (
    FieldDescriptor,
    FrameworkDescriptor,
    ModuleDescriptor,
    ModuleEvent,
    TrainingDataDescriptor,
)


class FakeModuleProcess(QObject):
    event_received = Signal(object)
    stderr_received = Signal(str)
    completed = Signal(int)
    protocol_failed = Signal(str)

    def __init__(self):
        super().__init__()
        self.started = []
        self.stopped = False
        self.promoted = []
        self.inferred = []

    def start(self, path):
        self.started.append(path)

    def request_stop(self):
        self.stopped = True

    def promote(self, job, selection):
        self.promoted.append((job, selection))

    def infer(self, request):
        self.inferred.append(request)


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
    labels = {"en": "Batch size", "zh_CN": "批大小", "ja": "バッチサイズ"}
    framework = FrameworkDescriptor(
        "v2ProPlus", "GPT-SoVITS v2ProPlus",
        ("preprocess", "train", "evaluate", "listen", "promote", "infer"),
        TrainingDataDescriptor("file", (".list",)),
        (FieldDescriptor("s2.batch_size", "integer", 2, {"minimum": 1}, labels),),
    )
    return setting, ModuleDescriptor(2, "gpt-sovits", "1.0", (framework,))


def _project(tmp_path: Path):
    project = tmp_path / "Acane"
    dataset = project / "dataset.list"
    project.mkdir()
    dataset.write_text("clip.wav|Acane|ja|test\n", encoding="utf-8")
    return project, dataset


def _listening_manifest(root: Path):
    entries = []
    for letter in "ABC":
        samples = []
        for language in ("zh", "ja", "en"):
            wav = root / f"candidate_{letter}" / f"{language}.wav"
            wav.parent.mkdir(parents=True, exist_ok=True)
            wav.write_bytes(f"{letter}-{language}".encode())
            samples.append({
                "language": language,
                "text": f"{language} sample",
                "wav": f"candidate_{letter}/{language}.wav",
                "sha256": hashlib.sha256(wav.read_bytes()).hexdigest(),
            })
        entries.append({"candidate": f"candidate_{letter}", "samples": samples})
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps({"schema_version": 1, "candidates": entries}), encoding="utf-8"
    )
    return manifest


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


@pytest.mark.parametrize(
    ("locale", "labels"),
    [
        (
            "en",
            ["Training configuration", "Candidate listening", "Inference trial"],
        ),
        ("zh_CN", ["训练配置", "候选试听", "推理试验"]),
        ("ja", ["トレーニング設定", "候補試聴", "推論テスト"]),
    ],
)
def test_training_workspace_has_three_fixed_localized_pages(tmp_path, locale, labels):
    from tts_builder.gui.training_page import TrainingPage

    app = create_application([])
    page = TrainingPage((_binding(tmp_path),), LocaleController(locale))

    assert [page.tabs.tabText(index) for index in range(page.tabs.count())] == labels
    assert page.tabs.widget(0) is page.config_page
    assert page.tabs.widget(1) is page.candidate_page
    assert page.tabs.widget(2) is page.inference_page
    assert page.config_page.isAncestorOf(page.form)
    assert page.config_page.isAncestorOf(page.progress)
    assert not page.candidate_page.isAncestorOf(page.form)
    page.close()


def test_running_job_uses_snapshot_while_form_edits_apply_next_time(tmp_path):
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
    page.start_button.click()
    snapshot = page.active_selection

    page.form.project_name.setText("NextRun")
    page.form.advanced_fields["s2.batch_size"].setValue(8)
    app.processEvents()

    assert snapshot.values["project_name"] == "Acane"
    assert snapshot.values["parameters"]["s2.batch_size"] == 2
    assert calls[0][3]["project_name"] == "Acane"
    assert calls[0][3]["parameters"]["s2.batch_size"] == 2
    assert page.form.project_name.text() == "NextRun"
    assert page.form.advanced_fields["s2.batch_size"].value() == 8
    assert page.form.isEnabled()
    assert not page.start_button.isEnabled()
    assert page.stop_button.isEnabled()
    assert page.next_run_hint.isVisibleTo(page)
    assert page.next_run_hint.text() == "Changes apply to the next run."
    with pytest.raises(FrozenInstanceError):
        snapshot.project_dir = tmp_path
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
    assert page.config_page.isAncestorOf(page.error_summary)
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


def test_listening_artifact_enables_human_promotion_and_failure_keeps_candidates(
    tmp_path, monkeypatch
):
    from tts_builder.gui.training_page import TrainingPage

    app = create_application([])
    process = FakeModuleProcess()
    page = TrainingPage(
        (_binding(tmp_path),), LocaleController("en"),
        process_factory=lambda _setting: process,
    )
    page.attach_process(process)
    job = tmp_path / "job.json"
    job.write_text("{}", encoding="utf-8")
    page.job_path = job
    manifest = _listening_manifest(tmp_path / "listening")
    process.event_received.emit(ModuleEvent(
        1,
        "job",
        "artifact",
        "now",
        artifacts=({"type": "listening_manifest", "path": str(manifest)},),
    ))
    app.processEvents()
    assert len(page.candidate_page.cards) == 3

    monkeypatch.setattr(QMessageBox, "question", lambda *_args, **_kwargs: QMessageBox.Yes)
    page.candidate_page.cards[0].select.click()
    page.candidate_page.promote_button.click()
    assert process.promoted == [(job, "candidate_A")]
    process.completed.emit(2)
    app.processEvents()

    assert len(page.candidate_page.cards) == 3
    assert page.tabs.currentIndex() == 0
    assert page.error_summary.isVisibleTo(page)
    assert page.config_page.isAncestorOf(page.error_summary)
    page.close()


def test_promotion_unlocks_inference_and_audio_artifact_loads_result(tmp_path):
    from tts_builder.gui.training_page import TrainingPage

    app = create_application([])
    process = FakeModuleProcess()
    built = []
    request = tmp_path / "request.json"
    request.write_text("{}", encoding="utf-8")

    def build_request(*args):
        built.append(args)
        return request

    page = TrainingPage(
        (_binding(tmp_path),),
        LocaleController("en"),
        process_factory=lambda _setting: process,
        build_inference_request=build_request,
    )
    page.attach_process(process)
    project, _ = _project(tmp_path)
    job = project / "jobs" / "job-1" / "job.json"
    job.parent.mkdir(parents=True)
    job.write_text("{}", encoding="utf-8")
    page.job_path = job
    model = project / "models" / "Acane"
    model.mkdir(parents=True)

    process.event_received.emit(ModuleEvent(
        1, "job", "artifact", "now",
        artifacts=({"type": "promoted_model", "path": str(model.resolve())},),
    ))
    assert page.inference_page.model is None
    process.event_received.emit(ModuleEvent(1, "job", "promotion_completed", "now"))
    assert page.inference_page.model == model.resolve()

    page.inference_page.text.setPlainText("hello")
    page.inference_page.start_button.click()
    assert built and built[0][0] == job
    assert built[0][1] == model.resolve()
    assert process.inferred == [request]
    assert page.form.isEnabled()
    assert not page.candidate_page.promote_button.isEnabled()
    assert not page.inference_page.start_button.isEnabled()

    output = project / "outputs" / "Acane" / "gui" / "result.wav"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"wav")
    process.event_received.emit(ModuleEvent(
        1, "job", "artifact", "now",
        artifacts=({"type": "inference_audio", "path": str(output.resolve())},),
    ))
    assert page.inference_page.result == output.resolve()
    page.close()


def test_idle_form_identity_recovers_and_relocks_promoted_model(tmp_path):
    from tts_builder.gui.training_page import TrainingPage

    app = create_application([])
    setting, module = _binding(tmp_path)
    primary = module.frameworks[0]
    alternate = FrameworkDescriptor(
        "alternate", "Alternate", primary.capabilities,
        primary.training_data, primary.fields,
    )
    binding = setting, ModuleDescriptor(
        module.protocol_version, module.module_id, module.module_version,
        (primary, alternate),
    )
    project, dataset = _project(tmp_path)
    model = project / "models" / "Acane"
    model.mkdir(parents=True)
    calls = []

    def recover(*identity):
        calls.append(identity)
        expected = (project.resolve(), module.module_id, primary.id, "Acane")
        return model if identity == expected else None

    page = TrainingPage(
        (binding,), LocaleController("en"), recover_promoted_model=recover,
        process_factory=lambda _setting: pytest.fail("must not launch a process"),
    )
    page.prefill_dataset(dataset)
    app.processEvents()

    assert calls[-1] == (project.resolve(), module.module_id, primary.id, "Acane")
    assert page.inference_page.model == model.resolve()

    page.form.project_name.setText("Other")
    assert page.inference_page.model is None
    page.form.project_name.setText("Acane")
    assert page.inference_page.model == model.resolve()
    page.form.framework_combo.setCurrentIndex(1)
    assert page.inference_page.model is None
    page.close()


def test_form_changes_do_not_recover_while_operation_is_running(tmp_path):
    from tts_builder.gui.training_page import TrainingPage

    create_application([])
    calls = []
    page = TrainingPage(
        (_binding(tmp_path),), LocaleController("en"),
        recover_promoted_model=lambda *args: calls.append(args),
    )
    page._set_running(True)
    page.form.project_name.setText("Acane")

    assert calls == []
    page.close()
