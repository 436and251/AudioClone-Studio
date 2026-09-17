from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from PySide6.QtWidgets import QComboBox, QDialog, QFileDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout

from .model_manager import is_asr_model_ready
from .i18n import Translator
from .settings import AppSettings, model_cache_paths, normalize_model_root
from .styles import MUTED


class SettingsDialog(QDialog):
    def __init__(self, settings: AppSettings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.translator = Translator(settings.locale)
        self._browse_buttons = []
        self.setMinimumWidth(600)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 24)
        form = QFormLayout()
        self.model_root = QLineEdit(str(settings.model_root))
        self.output_root = QLineEdit(str(settings.output_root))
        self.model = QComboBox()
        self.model.addItems(["large-v3-turbo", "small", "large-v3"])
        self.model.setCurrentText(settings.preferred_asr_model)
        self.model_root_label = QLabel()
        form.addRow(self.model_root_label, self._path_row(self.model_root))
        self.model_paths_hint = QLabel()
        self.model_paths_hint.setWordWrap(True)
        self.model_paths_hint.setStyleSheet(f"color:{MUTED}")
        form.addRow("", self.model_paths_hint)
        self.move_hint = QLabel()
        self.move_hint.setStyleSheet(f"color:{MUTED}")
        form.addRow("", self.move_hint)
        self.output_root_label = QLabel()
        form.addRow(self.output_root_label, self._path_row(self.output_root))
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
        layout.addLayout(form)
        layout.addStretch(1)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.cancel = QPushButton()
        self.save_button = QPushButton()
        self.save_button.setObjectName("Primary")
        self.cancel.clicked.connect(self.reject)
        self.save_button.clicked.connect(self.accept)
        buttons.addWidget(self.cancel)
        buttons.addWidget(self.save_button)
        layout.addLayout(buttons)
        self.locale.currentIndexChanged.connect(self._locale_changed)
        self.retranslate_ui(self.translator)
        self._update_ready()
        self._update_model_paths_hint()

    def _path_row(self, edit: QLineEdit):
        box = QHBoxLayout()
        box.setContentsMargins(0, 0, 0, 0)
        button = QPushButton()
        self._browse_buttons.append(button)
        button.clicked.connect(lambda: self._browse(edit))
        box.addWidget(edit, 1)
        box.addWidget(button)
        from PySide6.QtWidgets import QWidget
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
        self.asr_model_label.setText(translator.text("settings.asr_model"))
        self.locale_label.setText(translator.text("settings.language"))
        self.model_status_label.setText(translator.text("settings.model_status"))
        self.cancel.setText(translator.text("settings.cancel"))
        self.save_button.setText(translator.text("settings.save"))
        for button in self._browse_buttons:
            button.setText(translator.text("source.browse"))
        for index in range(self.locale.count()):
            code = self.locale.itemData(index)
            self.locale.setItemText(index, translator.text(f"locale.{code}"))
        self._update_ready()
        self._update_model_paths_hint()

    def _browse(self, edit: QLineEdit) -> None:
        chosen = QFileDialog.getExistingDirectory(self, self.translator.text("settings.choose_folder"), edit.text())
        if chosen:
            chosen_path = Path(chosen)
            if edit is self.model_root:
                chosen_path = normalize_model_root(chosen_path)
            edit.setText(str(chosen_path))

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
            preferred_asr_model=self.model.currentText(),
            locale=self.locale.currentData(),
        )
