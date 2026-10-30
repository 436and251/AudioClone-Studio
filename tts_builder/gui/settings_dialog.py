from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QComboBox, QDialog, QFileDialog, QFormLayout, QFrame, QHBoxLayout, QLabel, QLineEdit, QPushButton, QScrollArea, QSizePolicy, QVBoxLayout, QWidget

from .model_manager import is_asr_model_ready
from .i18n import Translator
from .module_settings import ModuleSettings
from .settings import AppSettings, model_cache_paths, normalize_model_root
from .styles import FAILED, MUTED
from ..audio2score.exporters import find_musescore


class SettingsDialog(QDialog):
    def __init__(self, settings: AppSettings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.translator = Translator(settings.locale)
        self._browse_buttons = []
        self.setMinimumWidth(600)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 24)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.model_root = QLineEdit(str(settings.model_root))
        self.output_root = QLineEdit(str(settings.output_root))
        self.audio2score_output_root = QLineEdit(str(settings.audio2score_output_root))
        self.musescore = QLineEdit(str(settings.musescore) if settings.musescore else "")
        self.model = QComboBox()
        self.model.addItems(["large-v3-turbo", "small", "large-v3"])
        self.model.setCurrentText(settings.preferred_asr_model)
        self.model_root_label = QLabel()
        form.addRow(self.model_root_label, self._path_row(self.model_root))
        self.model_paths_hint = QLabel()
        self.model_paths_hint.setWordWrap(True)
        self.model_paths_hint.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.model_paths_hint.setStyleSheet(f"color:{MUTED}")
        form.addRow("", self.model_paths_hint)
        self.move_hint = QLabel()
        self.move_hint.setWordWrap(True)
        self.move_hint.setStyleSheet(f"color:{MUTED}")
        form.addRow("", self.move_hint)
        self.output_root_label = QLabel()
        form.addRow(self.output_root_label, self._path_row(self.output_root))
        self.audio2score_output_label = QLabel()
        form.addRow(self.audio2score_output_label, self._path_row(self.audio2score_output_root))
        self.musescore_label = QLabel()
        form.addRow(self.musescore_label, self._path_row(self.musescore))
        self.musescore_hint = QLabel()
        self.musescore_hint.setWordWrap(True)
        self.musescore_hint.setStyleSheet(f"color:{MUTED}")
        form.addRow("", self.musescore_hint)
        self.musescore_status = QLabel()
        self.musescore_status.setWordWrap(True)
        self.musescore_status.setTextFormat(Qt.TextFormat.PlainText)
        form.addRow("", self.musescore_status)
        self.musescore.textChanged.connect(self._update_musescore_status)
        self.asr_model_label = QLabel()
        form.addRow(self.asr_model_label, self.model)
        self.locale = QComboBox()
        for locale in ("en", "zh_CN", "ja"):
            self.locale.addItem("", locale)
        self.locale.setCurrentIndex(self.locale.findData(settings.locale))
        self.locale_label = QLabel()
        form.addRow(self.locale_label, self.locale)
        self.ready = QLabel()
        self.model_status_label = QLabel()
        form.addRow(self.model_status_label, self.ready)
        self.model.currentTextChanged.connect(self._update_ready)
        self.model_root.textChanged.connect(self._update_ready)
        self.model_root.textChanged.connect(self._update_model_paths_hint)
        content_layout.addLayout(form)
        self.module_settings = ModuleSettings(
            settings.training_modules, self.translator, self
        )
        content_layout.addWidget(self.module_settings)
        content_layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(content)
        layout.addWidget(scroll, 1)
        self.validation_error = QLabel()
        self.validation_error.setWordWrap(True)
        self.validation_error.setStyleSheet(f"color:{FAILED}")
        self.validation_error.hide()
        layout.addWidget(self.validation_error)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.cancel = QPushButton()
        self.save_button = QPushButton()
        self.save_button.setObjectName("Primary")
        self.cancel.clicked.connect(self.reject)
        self.save_button.clicked.connect(self._save)
        buttons.addWidget(self.cancel)
        buttons.addWidget(self.save_button)
        layout.addLayout(buttons)
        self.locale.currentIndexChanged.connect(self._locale_changed)
        self.retranslate_ui(self.translator)
        self._update_ready()
        self._update_model_paths_hint()
        available = QApplication.primaryScreen().availableGeometry()
        self.resize(min(960, int(available.width() * .9)), min(850, int(available.height() * .9)))

    def _path_row(self, edit: QLineEdit):
        box = QHBoxLayout()
        box.setContentsMargins(0, 0, 0, 0)
        button = QPushButton()
        self._browse_buttons.append(button)
        button.clicked.connect(lambda: self._browse(edit))
        box.addWidget(edit, 1)
        box.addWidget(button)
        container = QWidget()
        container.setLayout(box)
        return container

    def _locale_changed(self, *_):
        self.retranslate_ui(Translator(self.locale.currentData()))

    def retranslate_ui(self, translator: Translator) -> None:
        self.translator = translator
        self.setWindowTitle(translator.text("settings.title"))
        self.model_root_label.setText(translator.text("settings.model_root"))
        self.move_hint.setText(translator.text("settings.model_move_hint"))
        self.output_root_label.setText(translator.text("settings.output_root"))
        self.audio2score_output_label.setText(translator.text("settings.audio2score_output_root"))
        self.musescore_label.setText(translator.text("audio2score.musescore_path"))
        self.musescore.setPlaceholderText(translator.text("audio2score.musescore_auto"))
        self.musescore_hint.setText(translator.text("audio2score.musescore_hint"))
        self.asr_model_label.setText(translator.text("settings.asr_model"))
        self.locale_label.setText(translator.text("settings.language"))
        self.model_status_label.setText(translator.text("settings.model_status"))
        self.cancel.setText(translator.text("settings.cancel"))
        self.save_button.setText(translator.text("settings.save"))
        self.module_settings.retranslate_ui(translator)
        for button in self._browse_buttons:
            button.setText(translator.text("source.browse"))
        for index in range(self.locale.count()):
            code = self.locale.itemData(index)
            self.locale.setItemText(index, translator.text(f"locale.{code}"))
        self._update_ready()
        self._update_model_paths_hint()
        self._update_musescore_status()

    def _update_musescore_status(self, *_):
        explicit = self.musescore.text().strip()
        try:
            executable = find_musescore(Path(explicit) if explicit else None)
        except (OSError, ValueError):
            self.musescore_status.setText(self.translator.text("audio2score.musescore_invalid"))
            self.musescore_status.setStyleSheet(f"color:{FAILED}")
            return
        key = ("audio2score.musescore_selected" if explicit else "audio2score.musescore_found") if executable else "audio2score.musescore_not_found"
        self.musescore_status.setText(self.translator.text(key, path=executable))
        self.musescore_status.setStyleSheet(f"color:{MUTED}")

    def _browse(self, edit: QLineEdit) -> None:
        if edit is self.musescore:
            chosen, _ = QFileDialog.getOpenFileName(self, "MuseScore", edit.text(), "MuseScore (*.exe);;All files (*)")
        else:
            chosen = QFileDialog.getExistingDirectory(self, self.translator.text("settings.choose_folder"), edit.text())
        if chosen:
            chosen_path = Path(chosen)
            if edit is self.model_root:
                chosen_path = normalize_model_root(chosen_path)
            edit.setText(str(chosen_path))

    def _save(self) -> None:
        message = self._validation_message()
        if message is not None:
            self.validation_error.setText(message)
            self.validation_error.show()
            return
        self.validation_error.hide()
        self.accept()

    def _validation_message(self) -> str | None:
        for edit, key in (
            (self.model_root, "settings.model_root_required"),
            (self.output_root, "settings.output_root_required"),
            (self.audio2score_output_root, "settings.audio2score_output_required"),
        ):
            if not edit.text().strip() or not Path(edit.text()).expanduser().is_absolute():
                return self.translator.text(key)
        if self.musescore.text().strip() and not Path(self.musescore.text().strip()).expanduser().is_file():
            return self.translator.text("audio2score.musescore_invalid")
        return self.module_settings.validation_error()

    def _update_ready(self, *_) -> None:
        root = normalize_model_root(Path(self.model_root.text()).expanduser())
        hf_home, _ = model_cache_paths(root)
        key = "settings.ready" if is_asr_model_ready(self.model.currentText(), hf_home) else "settings.not_downloaded"
        self.ready.setText(self.translator.text(key))

    def _update_model_paths_hint(self, *_) -> None:
        root = normalize_model_root(Path(self.model_root.text()).expanduser())
        hf_home, torch_home = model_cache_paths(root)
        self.model_paths_hint.setText(self.translator.text(
            "settings.cache_paths", hugging_face=hf_home, demucs=torch_home
        ))

    def result_settings(self) -> AppSettings:
        return replace(
            self.settings,
            model_root=normalize_model_root(Path(self.model_root.text()).expanduser()),
            output_root=Path(self.output_root.text()).expanduser(),
            audio2score_output_root=Path(self.audio2score_output_root.text()).expanduser(),
            preferred_asr_model=self.model.currentText(),
            locale=self.locale.currentData(),
            training_modules=self.module_settings.modules(),
            musescore=Path(self.musescore.text().strip()).expanduser() if self.musescore.text().strip() else None,
        )
