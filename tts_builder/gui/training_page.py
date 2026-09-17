from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence

from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QTabWidget,
    QVBoxLayout, QWidget,
)

from ..training_modules.job import build_job as default_build_job
from ..training_modules.artifacts import load_listening_manifest
from ..training_modules.models import ModuleEvent
from ..training_modules.process import ModuleProcessController
from .i18n import LocaleController, Translator
from .candidate_page import CandidatePage
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
        process_factory: Callable[..., object] = ModuleProcessController,
    ) -> None:
        super().__init__(parent)
        self.bindings = tuple(bindings)
        self.locale_controller = locale_controller
        self.translator = Translator(locale_controller.locale)
        self._build_job = build_job
        self._process_factory = process_factory
        self.process = None
        self.job_path: Path | None = None
        self.active_selection: TrainingSelection | None = None
        self._running = False
        self._operation = "run"

        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 24, 30, 24)
        layout.setSpacing(14)
        self.title = QLabel()
        self.title.setObjectName("Title")
        layout.addWidget(self.title)

        self.tabs = QTabWidget()
        self._tab_ids = ("config", "candidates", "inference")
        self.config_page = QWidget()
        config_layout = QVBoxLayout(self.config_page)
        config_layout.setContentsMargins(0, 14, 0, 0)
        config_layout.setSpacing(14)
        self.candidate_page = CandidatePage(locale_controller)
        self.inference_page = QWidget()
        self.tabs.addTab(self.config_page, "")
        self.tabs.addTab(self.candidate_page, "")
        self.tabs.addTab(self.inference_page, "")
        layout.addWidget(self.tabs)

        form_card = QFrame()
        form_card.setObjectName("Card")
        form_layout = QVBoxLayout(form_card)
        self.form = TrainingForm(self.bindings, self.translator)
        form_layout.addWidget(self.form)
        config_layout.addWidget(form_card)

        actions = QHBoxLayout()
        self.status = QLabel()
        self.status.setStyleSheet(f"color:{MUTED}")
        self.stop_button = QPushButton()
        self.stop_button.setObjectName("Danger")
        self.start_button = QPushButton()
        self.start_button.setObjectName("Primary")
        actions.addWidget(self.status, 1)
        actions.addWidget(self.stop_button)
        actions.addWidget(self.start_button)
        config_layout.addLayout(actions)
        self.next_run_hint = QLabel()
        self.next_run_hint.setStyleSheet(f"color:{MUTED}")
        self.next_run_hint.hide()
        config_layout.addWidget(self.next_run_hint)

        progress_card = QFrame()
        progress_card.setObjectName("Card")
        progress_layout = QVBoxLayout(progress_card)
        self.progress = TrainingProgress(self.translator)
        self.error_summary = QLabel()
        self.error_summary.setWordWrap(True)
        self.error_summary.setStyleSheet(f"color:{FAILED}")
        self.error_summary.hide()
        progress_layout.addWidget(self.progress)
        progress_layout.addWidget(self.error_summary)
        config_layout.addWidget(progress_card)

        self.activity_label = QLabel()
        self.activity = QPlainTextEdit()
        self.activity.setReadOnly(True)
        self.activity.setMaximumBlockCount(500)
        self.activity.setMaximumHeight(190)
        config_layout.addWidget(self.activity_label)
        config_layout.addWidget(self.activity)
        config_layout.addStretch(1)

        self.form.validity_changed.connect(self._update_actions)
        self.start_button.clicked.connect(self._start)
        self.stop_button.clicked.connect(self._stop)
        self.candidate_page.promotion_requested.connect(self._promote)
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
        self.stop_button.setText(translator.text("actions.stop"))
        self.activity_label.setText(translator.text("logs.activity"))
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
            process = self._process_factory(selection.setting)
            self.attach_process(process)
            self._operation = "run"
            self.error_summary.hide()
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

    def _event(self, event: ModuleEvent) -> None:
        self.progress.consume(event)
        message = _event_text(event)
        self._append_activity(f"{event.type} · {message}" if message else event.type)
        if event.type in {"stage_failed", "job_failed"}:
            self._show_error(message or event.type)
        elif event.type == "job_cancelled":
            self.status.setText(self.translator.text("status.stopped"))
        for artifact in event.artifacts:
            if artifact.get("type") == "listening_manifest":
                self._load_candidates(Path(str(artifact.get("path"))))

    def _stderr(self, text: str) -> None:
        for line in text.splitlines():
            if line:
                self._append_activity(line)

    def _protocol_failed(self, message: str) -> None:
        self._append_activity(message)
        self._show_error(message)

    def _completed(self, returncode: int) -> None:
        self._set_running(False)
        if returncode == 0:
            key = "training.promoted" if self._operation == "promote" else "training.completed"
            self.status.setText(self.translator.text(key))
        else:
            self._show_error(
                self.translator.text("training.process_failed", returncode=returncode)
            )
        self._operation = "run"

    def _show_error(self, message: str) -> None:
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
        self.next_run_hint.setVisible(running)
        self.candidate_page.set_busy(running)

    def _update_actions(self, *_):
        self.start_button.setEnabled(not self._running and self.form.is_valid())

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
            self._set_running(True)
            self.status.setText(self.translator.text("training.promoting"))
            self.process.promote(self.job_path, selection)
        except Exception as error:
            self._operation = "run"
            self._set_running(False)
            self._show_error(str(error))


def _event_text(event: ModuleEvent) -> str:
    arguments = event.message_args or {}
    if "error" in arguments:
        return str(arguments["error"])
    if event.message_key:
        details = ", ".join(f"{key}={value}" for key, value in arguments.items())
        return f"{event.message_key}: {details}" if details else event.message_key
    return ""
