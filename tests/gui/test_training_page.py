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
from tts_builder.training_modules.artifacts import load_listening_manifest
from tts_builder.training_modules.retention import CleanupSummary


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
    binding = _binding(tmp_path)
    job = project / "jobs" / "job-1" / "job.json"
    job.parent.mkdir(parents=True)
    job.write_text("{}", encoding="utf-8")

    def build(*args):
        calls.append(args)
        return job

    page = TrainingPage(
        (binding,), LocaleController("en"),
        build_job=build, process_factory=lambda _setting: process,
    )
    page.prefill_dataset(dataset)
    app.processEvents()
    assert page.start_button.isEnabled()

    page.start_button.click()

    assert process.started == [job]
    assert calls[0][0] == binding[0].project_root.resolve()
    assert calls[0][4] == ("preprocess", "s2", "s1", "evaluate")
    assert calls[0][5] == dataset.resolve()
    page.close()


def test_training_start_explains_invalid_configuration_instead_of_disabling(tmp_path):
    from tts_builder.gui.training_page import TrainingPage

    app = create_application([])
    page = TrainingPage((_binding(tmp_path),), LocaleController("en"))

    assert page.start_button.isEnabled()
    page.start_button.click()

    assert page.error_summary.text() == "Enter a valid project name using letters, numbers, '-' or '_'."
    assert page.error_summary.isVisibleTo(page)
    assert page.progress_card.isHidden()
    page.form.project_name.setText("Acane")
    page.start_button.click()
    assert page.error_summary.text() == "Choose valid training data for this framework."
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
    assert page.isAncestorOf(page.progress)
    assert not page.config_page.isAncestorOf(page.progress)
    assert page.isAncestorOf(page.activity_panel)
    assert not page.config_page.isAncestorOf(page.activity_panel)
    assert page.isAncestorOf(page.operation_status)
    assert not page.config_page.isAncestorOf(page.operation_status)
    assert not page.candidate_page.isAncestorOf(page.form)
    assert page.progress_card.isHidden()
    assert not page.activity_panel.toggle.isChecked()
    assert page.activity.isHidden()
    page.close()


def test_running_job_uses_snapshot_while_form_edits_apply_next_time(tmp_path):
    from tts_builder.gui.training_page import TrainingPage

    app = create_application([])
    process = FakeModuleProcess()
    calls = []
    project, dataset = _project(tmp_path)
    binding = _binding(tmp_path)
    job = project / "jobs" / "job-1" / "job.json"
    job.parent.mkdir(parents=True)
    job.write_text("{}", encoding="utf-8")

    def build(*args):
        calls.append(args)
        return job

    page = TrainingPage(
        (binding,), LocaleController("en"),
        build_job=build, process_factory=lambda _setting: process,
    )
    page.prefill_dataset(dataset)
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
    app.processEvents()
    assert page.progress.rows["s2"].bar.minimum() == 0
    assert page.progress.rows["s2"].bar.maximum() == 100
    assert page.progress.rows["s2"].bar.value() == 0
    process.event_received.emit(ModuleEvent(
        1, "job", "stage_progress", "now", stage="s2", current=25, total=100
    ))
    app.processEvents()
    assert page.progress.rows["s2"].bar.value() == 25
    assert page.progress.rows["s2"].status.text() == "进行中"
    assert "stage_progress" not in page.activity.toPlainText()

    process.event_received.emit(ModuleEvent(
        1, "job", "stage_failed", "now", stage="s2",
        message_key="pipeline.stage_failed", message_args={"error": "CUDA OOM"},
    ))
    process.stderr_received.emit("CUDA out of memory")
    app.processEvents()
    assert page.error_summary.isVisibleTo(page)
    assert page.isAncestorOf(page.error_summary)
    assert not page.config_page.isAncestorOf(page.error_summary)
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
    window.training_page.tabs.setCurrentIndex(2)
    window.navigation.set_current(1)
    window.pages.setCurrentIndex(0)
    window.dataset_page.continue_training.click()
    app.processEvents()

    assert window.pages.currentIndex() == 1
    assert window.training_page.tabs.currentIndex() == 0
    assert Path(window.training_page.form.dataset_edit.text()) == dataset.resolve()
    window.close()


def test_continue_to_training_explains_when_completed_dataset_was_removed(tmp_path):
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
    dataset.unlink()

    window.dataset_page.continue_training.click()

    assert window.dataset_page.status.text() == "Training data is no longer available."
    assert window.pages.currentIndex() == 0
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
    assert page.candidate_page.promote_button.text() == "Promoting A…"
    assert "Promoting A" in page.operation_status.text()
    process.completed.emit(2)
    app.processEvents()

    assert len(page.candidate_page.cards) == 3
    assert page.tabs.currentIndex() == 1
    assert page.candidate_page.operation_hint.isVisibleTo(page.candidate_page)
    assert "code 2" in page.candidate_page.operation_hint.text()
    page.close()


def test_successful_promotion_shows_feedback_and_opens_inference(tmp_path, monkeypatch):
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
    page.candidate_page.set_candidates(load_listening_manifest(
        _listening_manifest(tmp_path / "listening"),
        (tmp_path / "listening").resolve(),
    ))
    monkeypatch.setattr(QMessageBox, "question", lambda *_args, **_kwargs: QMessageBox.Yes)
    page.candidate_page.cards[0].select.click()
    page.candidate_page.promote_button.click()

    model = tmp_path / "models" / "Acane"
    model.mkdir(parents=True)
    process.event_received.emit(ModuleEvent(
        1, "job", "artifact", "now",
        artifacts=({"type": "promoted_model", "path": str(model.resolve())},),
    ))
    process.event_received.emit(ModuleEvent(1, "job", "promotion_completed", "now"))
    app.processEvents()

    assert page.tabs.currentIndex() == 2
    assert page.inference_page.model == model.resolve()
    assert "promoted" in page.operation_status.text().lower()
    assert page.candidate_page.promoted_id == "candidate_A"
    assert not page.candidate_page.promote_button.isEnabled()
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


def test_next_inference_recreates_job_removed_after_previous_completion(tmp_path):
    from tts_builder.gui.training_page import TrainingPage

    app = create_application([])
    process = FakeModuleProcess()
    setting, module = _binding(tmp_path)
    model = setting.project_root / "models" / "Lucy"
    model.mkdir(parents=True)
    job = setting.project_root / "jobs" / "inference-Lucy" / "job.json"
    ensured = []
    requests = []

    def ensure_job(*args):
        ensured.append(args)
        job.parent.mkdir(parents=True, exist_ok=True)
        job.write_text("{}", encoding="utf-8")
        return job

    def build_request(job_path, *args):
        if not job_path.is_file():
            raise ValueError("invalid job JSON")
        request = tmp_path / f"request-{len(requests) + 1}.json"
        request.write_text("{}", encoding="utf-8")
        requests.append(request)
        return request

    def cleanup(_root):
        if job.is_file():
            job.unlink()
            job.parent.rmdir()
            return CleanupSummary(removed=1)
        return CleanupSummary()

    page = TrainingPage(
        ((setting, module),),
        LocaleController("en"),
        process_factory=lambda _setting: process,
        ensure_inference_job=ensure_job,
        build_inference_request=build_request,
        cleanup=cleanup,
    )
    page.inference_page.set_model(model)
    page.inference_page.text.setPlainText("first")
    page.inference_page.start_button.click()
    process.completed.emit(0)
    app.processEvents()

    assert not job.exists()

    page.inference_page.text.setPlainText("second")
    page.inference_page.start_button.click()

    assert len(ensured) == 2
    assert process.inferred == requests
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
    model = setting.project_root / "models" / "Acane"
    model.mkdir(parents=True)
    calls = []

    def recover(*identity):
        calls.append(identity)
        expected = (setting.project_root.resolve(), module.module_id, primary.id, "Acane")
        return model if identity == expected else None

    page = TrainingPage(
        (binding,), LocaleController("en"), recover_promoted_model=recover,
        process_factory=lambda _setting: pytest.fail("must not launch a process"),
    )
    page.prefill_dataset(dataset)
    app.processEvents()

    assert calls[-1] == (
        setting.project_root.resolve(), module.module_id, primary.id, "Acane"
    )
    assert page.inference_page.model == model.resolve()
    assert page.candidate_page.operation_hint.text() == "A promoted model is ready."
    assert not page.candidate_page.promote_button.isEnabled()

    page.form.project_name.setText("Other")
    assert page.inference_page.model == model.resolve()
    assert page.candidate_page.operation_hint.isHidden()
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


def test_failed_job_can_be_retried_without_building_a_new_job(tmp_path):
    from tts_builder.gui.training_page import TrainingPage
    from tts_builder.training_modules.recovery import FailedJob

    app = create_application([])
    process = FakeModuleProcess()
    project, dataset = _project(tmp_path)
    setting, module = _binding(tmp_path)
    job = setting.project_root / "jobs" / "failed" / "job.json"
    job.parent.mkdir(parents=True)
    job.write_text("{}", encoding="utf-8")
    builds = []
    page = TrainingPage(
        ((setting, module),),
        LocaleController("zh_CN"),
        build_job=lambda *args: builds.append(args),
        recover_failed_job=lambda *args: FailedJob(job.resolve(), "evaluate"),
        process_factory=lambda _setting: process,
    )

    page.prefill_dataset(dataset)
    app.processEvents()

    assert page.retry_button.isVisibleTo(page)
    assert page.retry_button.text() == "↻  继续失败阶段：自动评测"
    page.retry_button.click()

    assert builds == []
    assert process.started == [job.resolve()]
    assert page.job_path == job.resolve()
    page.close()


def test_retry_explains_when_failed_job_was_removed(tmp_path):
    from tts_builder.gui.training_page import TrainingPage
    from tts_builder.training_modules.recovery import FailedJob

    app = create_application([])
    project, dataset = _project(tmp_path)
    setting, module = _binding(tmp_path)
    job = setting.project_root / "jobs" / "failed" / "job.json"
    job.parent.mkdir(parents=True)
    job.write_text("{}", encoding="utf-8")
    page = TrainingPage(
        ((setting, module),),
        LocaleController("en"),
        recover_failed_job=lambda *args: FailedJob(job.resolve(), "evaluate"),
    )
    page.prefill_dataset(dataset)
    app.processEvents()
    job.unlink()

    page.retry_button.click()

    assert page.error_summary.text() == "The failed job record is no longer available."
    assert page.error_summary.isVisibleTo(page)
    page.close()


def test_training_page_cleans_each_unique_module_root_on_startup(tmp_path):
    from tts_builder.gui.training_page import TrainingPage

    create_application([])
    binding = _binding(tmp_path)
    calls = []

    page = TrainingPage(
        (binding, binding),
        LocaleController("en"),
        cleanup=lambda root: calls.append(Path(root).resolve()) or CleanupSummary(),
    )

    assert calls == [binding[0].project_root.resolve()]
    page.close()


def test_process_completion_runs_cleanup_after_terminal_events(tmp_path):
    from tts_builder.gui.training_page import TrainingPage

    app = create_application([])
    process = FakeModuleProcess()
    calls = []
    page = TrainingPage(
        (_binding(tmp_path),),
        LocaleController("en"),
        cleanup=lambda root: calls.append(Path(root).resolve()) or CleanupSummary(),
    )
    page.attach_process(process)
    calls.clear()

    process.completed.emit(0)
    app.processEvents()

    assert calls == [page.bindings[0][0].project_root.resolve()]
    page.close()


def test_cleanup_failure_does_not_fail_the_completed_operation(tmp_path):
    from tts_builder.gui.training_page import TrainingPage

    app = create_application([])
    process = FakeModuleProcess()
    calls = 0

    def cleanup(_root):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise OSError("locked job directory")
        return CleanupSummary()

    page = TrainingPage(
        (_binding(tmp_path),), LocaleController("en"), cleanup=cleanup
    )
    page.attach_process(process)

    process.completed.emit(0)
    app.processEvents()

    assert page.status.text() == "Training task completed"
    assert "locked job directory" in page.activity.toPlainText()
    page.close()


def test_cleanup_summary_adds_only_one_activity_line(tmp_path):
    from tts_builder.gui.training_page import TrainingPage

    create_application([])
    page = TrainingPage(
        (_binding(tmp_path),),
        LocaleController("zh_CN"),
        cleanup=lambda _root: CleanupSummary(removed=2, retained_failures=1),
    )

    lines = page.activity.toPlainText().splitlines()
    assert len(lines) == 1
    assert "已清理 2 个过期任务" in lines[0]
    assert "保留 1 个可恢复失败任务" in lines[0]
    page.close()
