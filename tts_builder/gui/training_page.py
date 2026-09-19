from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence

from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea, QTabWidget,
    QVBoxLayout, QWidget,
)

from ..training_modules.job import build_job as default_build_job
from ..training_modules.inference import (
    build_inference_request as default_build_inference_request,
)
from ..training_modules.artifacts import load_listening_manifest
from ..training_modules.models import ModuleEvent
from ..training_modules.process import ModuleProcessController
from ..training_modules.recovery import (
    FailedJob,
    latest_failed_job,
    latest_promoted_model,
)
from .i18n import LocaleController, Translator
from .candidate_page import CandidatePage
from .inference_page import InferenceInput, InferencePage
from .log_panel import LogPanel
from .styles import FAILED, MUTED
from .training_form import ModuleBinding, TrainingForm, TrainingSelection
from .training_progress import TrainingProgress


class TrainingPage(QWidget):
    def __init__(
        self,
        bindings: Sequence[ModuleBinding],
        locale_controller: LocaleController,
        parent=None,
        *,
        build_job: Callable[..., Path] = default_build_job,
        build_inference_request: Callable[..., Path] = default_build_inference_request,
        recover_promoted_model: Callable[..., Path | None] = latest_promoted_model,
        recover_failed_job: Callable[..., FailedJob | None] = latest_failed_job,
        process_factory: Callable[..., object] = ModuleProcessController,
    ) -> None:
        super().__init__(parent)
        self.bindings = tuple(bindings)
        self.locale_controller = locale_controller
        self.translator = Translator(locale_controller.locale)
        self._build_job = build_job
        self._build_inference_request = build_inference_request
        self._recover_model = recover_promoted_model
        self._recover_failed = recover_failed_job
        self._process_factory = process_factory
        self.process = None
        self.job_path: Path | None = None
        self.active_selection: TrainingSelection | None = None
        self._running = False
        self._operation = "run"
        self._pending_promoted_model: Path | None = None
        self.failed_job: FailedJob | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 24, 30, 24)
        layout.setSpacing(14)
        self.title = QLabel()
        self.title.setObjectName("Title")
        layout.addWidget(self.title)

        self.tabs = QTabWidget()
        self._tab_ids = ("config", "candidates", "inference")
        self.config_page = QScrollArea()
        self.config_page.setWidgetResizable(True)
        config_content = QWidget()
        self.config_page.setWidget(config_content)
        config_layout = QVBoxLayout(config_content)
        config_layout.setContentsMargins(0, 14, 0, 0)
        config_layout.setSpacing(14)
        self.candidate_page = CandidatePage(locale_controller)
        self.inference_page = InferencePage(locale_controller)
        self.tabs.addTab(self.config_page, "")
        self.tabs.addTab(self.candidate_page, "")
        self.tabs.addTab(self.inference_page, "")
        layout.addWidget(self.tabs, 1)

        form_card = QFrame()
        form_card.setObjectName("Card")
        form_layout = QVBoxLayout(form_card)
        self.form = TrainingForm(self.bindings, self.translator)
        form_layout.addWidget(self.form)
        config_layout.addWidget(form_card)

        actions = QHBoxLayout()
        self.stop_button = QPushButton()
        self.stop_button.setObjectName("Danger")
        self.retry_button = QPushButton()
        self.start_button = QPushButton()
        self.start_button.setObjectName("Primary")
        actions.addStretch(1)
        actions.addWidget(self.retry_button)
        actions.addWidget(self.stop_button)
        actions.addWidget(self.start_button)
        config_layout.addLayout(actions)
        self.next_run_hint = QLabel()
        self.next_run_hint.setStyleSheet(f"color:{MUTED}")
        self.next_run_hint.hide()
        config_layout.addWidget(self.next_run_hint)

        self.progress_card = QFrame()
        self.progress_card.setObjectName("Card")
        progress_layout = QVBoxLayout(self.progress_card)
        self.progress = TrainingProgress(self.translator)
        self.error_summary = QLabel()
        self.error_summary.setWordWrap(True)
        self.error_summary.setStyleSheet(f"color:{FAILED}")
        self.error_summary.hide()
        progress_layout.addWidget(self.progress)
        progress_layout.addWidget(self.error_summary)
        layout.addWidget(self.progress_card)
        self.progress_card.hide()

        self.operation_status = QLabel()
        self.operation_status.setObjectName("OperationStatus")
        self.operation_status.setStyleSheet(f"color:{MUTED}")
        self.operation_status.setWordWrap(True)
        self.status = self.operation_status
        layout.addWidget(self.operation_status)

        self.activity_panel = LogPanel(self.translator)
        self.activity = self.activity_panel.view
        self.activity.setMaximumBlockCount(500)
        self.activity.setMaximumHeight(190)
        layout.addWidget(self.activity_panel)
        config_layout.addStretch(1)

        self.form.validity_changed.connect(self._update_actions)
        self.form.project_name.textChanged.connect(self._recover_promoted_model)
        self.form.framework_combo.currentIndexChanged.connect(
            self._recover_promoted_model
        )
        self.start_button.clicked.connect(self._start)
        self.stop_button.clicked.connect(self._stop)
        self.retry_button.clicked.connect(self._retry_failed)
        self.candidate_page.promotion_requested.connect(self._promote)
        self.inference_page.inference_requested.connect(self._infer)
        locale_controller.locale_changed.connect(self._locale_changed)
        self.retranslate_ui(self.translator)
        self._set_running(False)

    def prefill_dataset(self, path: Path) -> None:
        self.form.prefill_dataset(path)
        self._update_actions()

    def attach_process(self, process) -> None:
        self.process = process
        process.event_received.connect(self._event)
        process.stderr_received.connect(self._stderr)
        process.completed.connect(self._completed)
        process.protocol_failed.connect(self._protocol_failed)

    def retranslate_ui(self, translator: Translator) -> None:
        self.translator = translator
        self.title.setText(translator.text("training.title"))
        for index, tab_id in enumerate(self._tab_ids):
            self.tabs.setTabText(index, translator.text(f"training.tab.{tab_id}"))
        self.start_button.setText(translator.text("training.start"))
        self._update_retry_button()
        self.stop_button.setText(translator.text("actions.stop"))
        self.activity_panel.retranslate_ui(translator)
        self.next_run_hint.setText(translator.text("training.next_run_hint"))
        self.form.retranslate_ui(translator)
        self.progress.retranslate_ui(translator)

    def _start(self) -> None:
        if not self.form.is_valid():
            return
        selection = self.form.snapshot()
        if selection.setting is None:
            self._show_error(self.translator.text("training.module_unavailable"))
            return
        try:
            self.job_path = self._build_job(
                selection.project_dir,
                selection.module,
                selection.framework,
                selection.values,
                selection.stages,
                selection.training_data,
            )
            self.active_selection = selection
            self._pending_promoted_model = None
            self.inference_page.set_model(None)
            process = self._process_factory(selection.setting)
            self.attach_process(process)
            self._operation = "run"
            self.error_summary.hide()
            self.progress_card.show()
            self._set_running(True)
            process.start(self.job_path)
            self.status.setText(self.translator.text("training.running"))
        except Exception as error:
            self.active_selection = None
            self._set_running(False)
            self._show_error(str(error))

    def _stop(self) -> None:
        if self.process is not None:
            self.process.request_stop()
            self.status.setText(self.translator.text("status.stopping"))

    def _retry_failed(self) -> None:
        failed = self.failed_job
        if failed is None or not failed.job_path.is_file():
            self._recover_promoted_model()
            return
        try:
            setting, _, _ = self.form.selection()
            if setting is None:
                raise ValueError(self.translator.text("training.module_unavailable"))
            self.job_path = failed.job_path
            self.active_selection = None
            process = self._process_factory(setting)
            self.attach_process(process)
            self._operation = "run"
            self.error_summary.hide()
            self.progress_card.show()
            self._set_running(True)
            process.start(self.job_path)
            self.status.setText(self.translator.text("training.running"))
        except Exception as error:
            self._set_running(False)
            self._show_error(str(error))

    def _event(self, event: ModuleEvent) -> None:
        if event.stage is not None:
            self.progress_card.show()
        self.progress.consume(event)
        message = _event_text(event)
        if event.type not in {"stage_progress", "inference_progress"}:
            self._append_activity(f"{event.type} · {message}" if message else event.type)
        if event.type in {"stage_failed", "job_failed"}:
            self._show_error(message or event.type)
        elif event.type == "job_cancelled":
            self.status.setText(self.translator.text("status.stopped"))
        elif event.type == "inference_progress" and event.total:
            self.status.setText(self.translator.text(
                "inference.progress", current=int(event.current or 0), total=int(event.total)
            ))
        elif event.type == "promotion_failed":
            self.candidate_page.set_promotion_error(message or event.type)
            self.status.setText(message or event.type)
            self.tabs.setCurrentIndex(1)
        elif event.type == "inference_failed":
            self.inference_page.set_error(message or event.type)
            self.status.setText(message or event.type)
            self.tabs.setCurrentIndex(2)
        for artifact in event.artifacts:
            artifact_type = artifact.get("type")
            if artifact_type == "listening_manifest":
                self._load_candidates(Path(str(artifact.get("path"))))
            elif artifact_type == "promoted_model":
                model = Path(str(artifact.get("path"))).resolve()
                if not model.is_dir():
                    self._protocol_failed("promoted model is unavailable")
                    return
                self._pending_promoted_model = model
            elif artifact_type == "inference_audio":
                audio = Path(str(artifact.get("path"))).resolve()
                if not audio.is_file():
                    self._protocol_failed("inference audio is unavailable")
                    return
                self.inference_page.set_result(audio)
                self.tabs.setCurrentIndex(2)
        if event.type == "promotion_completed" and self._pending_promoted_model is not None:
            self.inference_page.set_model(self._pending_promoted_model)
            self._pending_promoted_model = None
            self.candidate_page.set_promoted()
            self.status.setText(self.translator.text("training.promoted"))
            self.tabs.setCurrentIndex(2)

    def _stderr(self, text: str) -> None:
        for line in text.splitlines():
            lowered = line.casefold()
            if line and (
                line.startswith("Error:")
                or " failed:" in line
                or "error" in lowered
                or "exception" in lowered
                or "traceback" in lowered
                or "out of memory" in lowered
            ):
                self._append_activity(line)

    def _protocol_failed(self, message: str) -> None:
        self._append_activity(message)
        if self._operation == "infer":
            self.inference_page.set_error(message)
            self.tabs.setCurrentIndex(2)
        elif self._operation == "promote":
            self.candidate_page.set_promotion_error(message)
            self.status.setText(message)
            self.tabs.setCurrentIndex(1)
        else:
            self._show_error(message)

    def _completed(self, returncode: int) -> None:
        self._set_running(False)
        operation = self._operation
        if returncode == 0:
            key = {
                "promote": "training.promoted",
                "infer": "inference.completed",
            }.get(self._operation, "training.completed")
            self.status.setText(self.translator.text(key))
        else:
            message = self.translator.text(
                "training.process_failed", returncode=returncode
            )
            if self._operation == "infer":
                self.inference_page.set_error(message)
                self.tabs.setCurrentIndex(2)
                self.status.setText(self.translator.text("training.failed"))
            elif self._operation == "promote":
                self.candidate_page.set_promotion_error(message)
                self.tabs.setCurrentIndex(1)
                self.status.setText(self.translator.text("training.failed"))
            else:
                self._show_error(message)
            if self._operation == "promote":
                self._pending_promoted_model = None
        self._operation = "run"
        if not (returncode != 0 and operation == "promote"):
            self._recover_promoted_model()

    def _show_error(self, message: str) -> None:
        self.progress_card.show()
        self.error_summary.setText(message)
        self.error_summary.show()
        self.tabs.setCurrentIndex(0)
        self.status.setText(self.translator.text("training.failed"))

    def _append_activity(self, text: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        self.activity.appendPlainText(f"{stamp}  {text}")

    def _set_running(self, running: bool) -> None:
        self._running = running
        self.form.setEnabled(True)
        self.stop_button.setEnabled(running)
        self.start_button.setEnabled(not running and self.form.is_valid())
        self.retry_button.setEnabled(not running and self.failed_job is not None)
        self.next_run_hint.setVisible(running)
        self.candidate_page.set_busy(running)
        self.inference_page.set_busy(running)

    def _update_actions(self, *_):
        self.start_button.setEnabled(not self._running and self.form.is_valid())

    def _recover_promoted_model(self, *_):
        if self._running:
            return
        self.inference_page.set_model(None)
        self.candidate_page.clear_promotion()
        self.failed_job = None
        self._update_retry_button()
        project_name = self.form.project_name.text().strip()
        if not project_name:
            return
        try:
            setting, module, framework = self.form.selection()
            if setting is None:
                return
            project = Path(setting.project_root).resolve()
            if not project.is_dir():
                return
            model = self._recover_model(
                project, module.module_id, framework.id, project_name
            )
            self.failed_job = self._recover_failed(
                project, module.module_id, framework.id, project_name
            )
        except (OSError, ValueError):
            return
        self.inference_page.set_model(model)
        if model is not None:
            self.candidate_page.set_promoted()
        self._update_retry_button()

    def _update_retry_button(self) -> None:
        failed = self.failed_job
        self.retry_button.setVisible(failed is not None)
        self.retry_button.setEnabled(not self._running and failed is not None)
        if failed is not None:
            stage = self.translator.text(f"training.stage.{failed.stage}")
            self.retry_button.setText(
                self.translator.text("training.retry_failed", stage=stage)
            )

    def _locale_changed(self, locale: str) -> None:
        self.retranslate_ui(Translator(locale))

    def _load_candidates(self, manifest: Path) -> None:
        try:
            candidates = load_listening_manifest(manifest, manifest.resolve().parent)
            self.candidate_page.set_candidates(candidates)
            self.tabs.setCurrentIndex(1)
        except (OSError, ValueError) as error:
            self._protocol_failed(str(error))

    def _promote(self, selection: str) -> None:
        if self.process is None or self.job_path is None:
            self._show_error(self.translator.text("training.promotion_unavailable"))
            return
        try:
            self._operation = "promote"
            self.candidate_page.set_promoting(selection)
            self._set_running(True)
            self.status.setText(self.translator.text(
                "candidate.promoting", name=selection[-1]
            ))
            self.process.promote(self.job_path, selection)
        except Exception as error:
            self._operation = "run"
            self._set_running(False)
            self._show_error(str(error))

    def _infer(self, values: InferenceInput) -> None:
        if self.process is None or self.job_path is None or self.inference_page.model is None:
            self.inference_page.set_error(self.translator.text("inference.no_model"))
            return
        try:
            request = self._build_inference_request(
                self.job_path,
                self.inference_page.model,
                values.text,
                values.text_file,
                values.language,
                values.device,
                datetime.now(),
            )
            self._operation = "infer"
            self._set_running(True)
            self.status.setText(self.translator.text("inference.running"))
            self.process.infer(request)
        except Exception as error:
            self._operation = "run"
            self._set_running(False)
            self.inference_page.set_error(str(error))
            self.tabs.setCurrentIndex(2)


def _event_text(event: ModuleEvent) -> str:
    arguments = event.message_args or {}
    if "error" in arguments:
        return str(arguments["error"])
    if event.message_key:
        details = ", ".join(f"{key}={value}" for key, value in arguments.items())
        return f"{event.message_key}: {details}" if details else event.message_key
    return ""
