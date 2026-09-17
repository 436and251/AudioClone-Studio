from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QFrame, QHBoxLayout, QLabel, QLineEdit, QMainWindow,
    QMessageBox, QPushButton, QVBoxLayout, QWidget, QToolButton
)

from ..config import BuildConfig
from ..events import PipelineEvent
from .controller import TaskController
from .i18n import LocaleController, Translator
from .log_panel import LogPanel
from .model_manager import estimated_asr_bytes, is_asr_model_ready
from .models import ProgressModel
from .progress_panel import ProgressPanel
from .settings import AppSettings, apply_model_environment, should_update_current_output
from .settings_dialog import SettingsDialog
from .source_input import SourceInput
from .styles import FAILED, MUTED


class MainWindow(QMainWindow):
    def __init__(self, settings: AppSettings, controller: TaskController | None = None,
                 locale_controller: LocaleController | None = None):
        super().__init__()
        self.settings = settings
        self.controller = controller or TaskController(self)
        self.locale_controller = locale_controller or LocaleController(settings.locale, self)
        self.translator = Translator(self.locale_controller.locale)
        self.progress_model = ProgressModel()
        self._status_key = None
        self._status_values = {}
        self._start_mode = "build"
        self.resize(860, 760)
        self.setMinimumSize(760, 650)
        self._build()
        self._wire()
        self.retranslate_ui(self.translator)
        self._validate()

    def _build(self) -> None:
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(30, 24, 30, 24)
        layout.setSpacing(16)
        header = QHBoxLayout()
        titles = QVBoxLayout()
        self.title_label = QLabel()
        self.title_label.setObjectName("Title")
        self.subtitle_label = QLabel()
        self.subtitle_label.setObjectName("Subtitle")
        titles.addWidget(self.title_label)
        titles.addWidget(self.subtitle_label)
        self.settings_btn = QToolButton()
        self.settings_btn.setObjectName("IconButton")
        self.settings_btn.setText("⚙")
        self.settings_btn.setFixedSize(44, 44)
        self.settings_btn.clicked.connect(self._open_settings)
        header.addLayout(titles, 1)
        header.addWidget(self.settings_btn)
        layout.addLayout(header)

        card = QFrame()
        card.setObjectName("Card")
        form = QVBoxLayout(card)
        form.setContentsMargins(18, 18, 18, 18)
        form.setSpacing(12)
        self.source = SourceInput(self.translator)
        form.addWidget(self.source)
        row = QHBoxLayout()
        self.speaker = QLineEdit()
        self.language = QComboBox()
        for language in ("auto", "ja", "zh", "en"):
            self.language.addItem("", language)
        self.model = QComboBox()
        self.model.addItems(["large-v3-turbo", "small", "large-v3"])
        self.model.setCurrentText(self.settings.preferred_asr_model)
        row.addWidget(self.speaker, 2)
        row.addWidget(self.language, 1)
        row.addWidget(self.model, 1)
        form.addLayout(row)
        out = QHBoxLayout()

        self.output_label = QLabel()
        self.output_label.setObjectName("FieldChip")
        self.output_label.setAlignment(Qt.AlignCenter)
        self.output_label.setFixedWidth(140)

        self.output = QLineEdit(str(self.settings.output_root))

        self.output_browse = QPushButton()
        self.output_browse.clicked.connect(self._browse_output)

        out.addWidget(self.output_label)
        out.addWidget(self.output, 1)
        out.addWidget(self.output_browse)

        form.addLayout(out)
        layout.addWidget(card)

        actions = QHBoxLayout()
        self.status = QLabel("")
        self.status.setStyleSheet(f"color:{MUTED}")
        self.open_output = QPushButton()
        self.open_output.clicked.connect(self._open_output)
        self.stop = QPushButton()
        self.stop.setObjectName("Danger")
        self.stop.clicked.connect(self._stop)
        self.start = QPushButton()
        self.start.setObjectName("Primary")
        self.start.clicked.connect(self._start)
        actions.addWidget(self.status, 1)
        actions.addWidget(self.open_output)
        actions.addWidget(self.stop)
        actions.addWidget(self.start)
        layout.addLayout(actions)

        progress_card = QFrame()
        progress_card.setObjectName("Card")
        progress_layout = QVBoxLayout(progress_card)
        progress_layout.setContentsMargins(18, 16, 18, 16)
        self.processing_label = QLabel()
        progress_layout.addWidget(self.processing_label)
        self.progress = ProgressPanel(self.translator)
        progress_layout.addWidget(self.progress)
        layout.addWidget(progress_card)
        self.logs = LogPanel(self.translator)
        layout.addWidget(self.logs)
        layout.addStretch(1)
        self.setCentralWidget(root)

    def _wire(self) -> None:
        self.source.source_changed.connect(self._validate)
        self.speaker.textChanged.connect(self._validate)
        self.controller.event_received.connect(self._event)
        self.controller.task_completed.connect(self._completed)
        self.controller.task_failed.connect(self._failed)
        self.controller.task_cancelled.connect(self._cancelled)
        self.controller.running_changed.connect(self._running)
        self.locale_controller.locale_changed.connect(self._locale_changed)

    def _locale_changed(self, locale: str) -> None:
        self.translator = Translator(locale)
        self.settings = replace(self.settings, locale=locale)
        self.retranslate_ui(self.translator)

    def retranslate_ui(self, translator: Translator) -> None:
        self.setWindowTitle(translator.text("app.title"))
        self.title_label.setText(translator.text("app.title"))
        self.subtitle_label.setText(translator.text("main.subtitle"))
        self.settings_btn.setToolTip(translator.text("settings.tooltip"))
        self.source.retranslate_ui(translator)
        self.speaker.setPlaceholderText(translator.text("speaker.placeholder"))
        for index in range(self.language.count()):
            code = self.language.itemData(index)
            self.language.setItemText(index, translator.text(f"language.{code}"))
        self.output_label.setText(translator.text("output.directory"))
        self.output_browse.setText(translator.text("output.browse"))
        self.open_output.setText(translator.text("output.open"))
        self.stop.setText(translator.text("actions.stop"))
        self.start.setText(translator.text(f"actions.{self._start_mode}"))
        self.processing_label.setText(translator.text("processing.title"))
        self.progress.retranslate_ui(translator)
        self.logs.retranslate_ui(translator)
        if self._status_key:
            self.status.setText(translator.text(self._status_key, **self._status_values))

    def _set_status(self, key: str, **values: object) -> None:
        self._status_key = key
        self._status_values = values
        self.status.setText(self.translator.text(key, **values))

    def _set_start_mode(self, mode: str) -> None:
        self._start_mode = mode
        self.start.setText(self.translator.text(f"actions.{mode}"))

    def _validate(self, *_) -> None:
        self.start.setEnabled(bool(self.source.value() and self.speaker.text().strip()) and not self.controller.running)

    def _config(self) -> BuildConfig:
        return BuildConfig(
            output=Path(self.output.text()).expanduser(), speaker=self.speaker.text().strip(),
            language=str(self.language.currentData()), asr_model=self.model.currentText(),
        )

    def _start(self) -> None:
        config = self._config()
        if not is_asr_model_ready(config.asr_model, self.settings.model_root / "huggingface"):
            size = estimated_asr_bytes(config.asr_model)
            size_text = f"~{size / 1e9:.1f} GB" if size else self.translator.text("models.size_unknown")
            box = QMessageBox(self)
            box.setWindowTitle(self.translator.text("models.prepare_title"))
            box.setText(self.translator.text("models.not_installed", model=config.asr_model))
            box.setInformativeText(self.translator.text(
                "models.download_info", size=size_text, model_root=self.settings.model_root
            ))
            box.setStandardButtons(QMessageBox.Cancel | QMessageBox.Ok)
            box.button(QMessageBox.Ok).setText(self.translator.text("models.download_continue"))
            if box.exec() != QMessageBox.Ok:
                return
        self.progress_model = ProgressModel()
        self.progress.apply(self.progress_model)
        self._set_start_mode("build")
        self.status.setStyleSheet(f"color:{MUTED}")
        self._set_status("status.preparing")
        self.controller.start(self.source.value(), config)

    def _event(self, event: PipelineEvent) -> None:
        self.progress_model.consume(event)
        self.progress.apply(self.progress_model)
        self.logs.append_event(event)
        if event.stage and event.message:
            self._status_key = None
            self.status.setText(event.message)

    def _running(self, running: bool) -> None:
        self.source.setEnabled(not running)
        self.speaker.setEnabled(not running)
        self.language.setEnabled(not running)
        self.model.setEnabled(not running)
        self.output.setEnabled(not running)
        self.output_browse.setEnabled(not running)
        self.settings_btn.setEnabled(not running)
        self.stop.setEnabled(running)
        self._validate()

    def _completed(self, summary) -> None:
        self.status.setStyleSheet(f"color:{MUTED}")
        self._set_status("status.ready", accepted=summary.accepted, rejected=summary.rejected)
        self._set_start_mode("build")

    def _failed(self, title: str, message: str, detail: str) -> None:
        self.status.setStyleSheet(f"color:{FAILED}")
        self._status_key = None
        self.status.setText(message)
        self.logs.append_text(detail)
        self._set_start_mode("retry")
        box = QMessageBox(QMessageBox.Critical, title, message, QMessageBox.Ok, self)
        box.setDetailedText(detail)
        box.exec()

    def _cancelled(self) -> None:
        self._set_status("status.stopped")

    def _stop(self) -> None:
        self._set_status("status.stopping")
        self.controller.stop()

    def _browse_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, self.translator.text("output.choose"), self.output.text())
        if path:
            self.output.setText(path)

    def _open_output(self) -> None:
        path = Path(self.output.text()).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve())))

    def _open_settings(self) -> None:
        old_settings = self.settings
        current_output = Path(self.output.text()).expanduser()
        dialog = SettingsDialog(self.settings, self)
        if dialog.exec():
            new_settings = dialog.result_settings()
            self.settings = new_settings
            self.settings.save()
            apply_model_environment(self.settings.model_root)
            if should_update_current_output(current_output, old_settings.output_root):
                self.output.setText(str(self.settings.output_root))
            self.model.setCurrentText(self.settings.preferred_asr_model)
            self.locale_controller.set_locale(self.settings.locale)
