from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from PySide6.QtWidgets import QDialog, QFileDialog, QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

from .settings import AppSettings
from .i18n import Translator
from .styles import GREEN, MUTED
from .system_check import detect_system, recommendation_for


class FirstRunDialog(QDialog):
    def __init__(self, settings: AppSettings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.translator = Translator(settings.locale)
        self.setWindowTitle(self.translator.text("app.title"))
        self.setMinimumWidth(620)
        self.info = detect_system(settings.model_root, self.translator)
        self.recommendation = recommendation_for(self.info, self.translator)
        self._build()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(34, 30, 34, 30)
        layout.setSpacing(16)
        title = QLabel(self.translator.text("app.title"))
        title.setObjectName("Title")
        subtitle = QLabel(self.translator.text("first_run.subtitle"))
        subtitle.setObjectName("Subtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        card = QFrame()
        card.setObjectName("Card")
        rows = QVBoxLayout(card)
        rows.setContentsMargins(18, 16, 18, 16)
        gpu = self.info.gpu_name if self.info.cuda_available else self.translator.text("first_run.cpu_mode")
        rows.addWidget(self._status(
            self.translator.text("first_run.acceleration"), gpu, self.info.cuda_available, accent=True
        ))
        rows.addWidget(self._status(self.translator.text("first_run.memory"), f"{self.info.ram_gb:.1f} GB", self.info.ram_gb >= 8))
        ffmpeg = self.translator.text("first_run.ready" if self.info.ffmpeg_available else "first_run.not_found")
        rows.addWidget(self._status(self.translator.text("first_run.ffmpeg"), ffmpeg, self.info.ffmpeg_available))
        rows.addWidget(self._status(
            self.translator.text("first_run.model_storage"),
            self.translator.text("first_run.free_gb", free_gb=self.info.free_gb),
            self.info.free_gb >= 5,
        ))
        layout.addWidget(card)

        note = QLabel(self.translator.text("first_run.before_start"))
        note.setWordWrap(True)
        note.setStyleSheet(f"color:{MUTED}; line-height:1.4")
        layout.addWidget(note)

        self.path_label = QLabel(str(self.settings.model_root))
        self.path_label.setStyleSheet(f"color:{MUTED}")
        change = QPushButton(self.translator.text("first_run.change"))
        change.clicked.connect(self._change_root)
        row = QHBoxLayout()
        row.addWidget(QLabel(self.translator.text("first_run.model_storage")))
        row.addWidget(self.path_label, 1)
        row.addWidget(change)
        layout.addLayout(row)
        layout.addStretch(1)
        button = QPushButton(self.translator.text("first_run.continue"))
        button.setObjectName("Primary")
        button.clicked.connect(self.accept)
        layout.addWidget(button)

    def _status(self, name: str, value: str, good: bool, accent: bool = False) -> QLabel:
        icon = "●" if good else "○"
        color = GREEN if good else "#E6B450"
        label = QLabel(f"{icon}  {name}    {value}")
        label.setStyleSheet(f"color:{color if accent else '#FFFFFF'}")
        return label

    def _change_root(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self, self.translator.text("first_run.choose_storage"), str(self.settings.model_root)
        )
        if chosen:
            self.settings = replace(self.settings, model_root=Path(chosen))
            self.path_label.setText(chosen)

    def result_settings(self) -> AppSettings:
        model = self.settings.preferred_asr_model
        if not self.info.cuda_available and model == "large-v3-turbo":
            model = self.recommendation.asr_model
        return replace(self.settings, first_run_completed=True, preferred_asr_model=model)
