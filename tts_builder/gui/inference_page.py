from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QFileDialog, QFormLayout, QHBoxLayout, QLabel, QComboBox, QLineEdit,
    QPlainTextEdit, QPushButton, QVBoxLayout, QWidget,
)

from .audio_player import AudioPlayer
from .i18n import LocaleController, Translator
from .styles import FAILED, MUTED


@dataclass(frozen=True)
class InferenceInput:
    text: str | None
    text_file: Path | None
    language: str
    device: str


class InferencePage(QWidget):
    inference_requested = Signal(object)

    def __init__(self, locale_controller: LocaleController, parent=None, *, player=None):
        super().__init__(parent)
        self.locale_controller = locale_controller
        self.translator = Translator(locale_controller.locale)
        self.player = player or AudioPlayer(self)
        self.model: Path | None = None
        self.result: Path | None = None
        self._busy = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        self.title = QLabel()
        self.title.setObjectName("Subtitle")
        layout.addWidget(self.title)
        self.model_hint = QLabel()
        self.model_hint.setWordWrap(True)
        self.model_hint.setStyleSheet(f"color:{MUTED}")
        layout.addWidget(self.model_hint)
        self.busy_hint = QLabel()
        self.busy_hint.setObjectName("OperationStatus")
        self.busy_hint.hide()
        layout.addWidget(self.busy_hint)

        form = QFormLayout()
        self.text = QPlainTextEdit()
        self.text.setMaximumHeight(130)
        self.text.textChanged.connect(self._update_actions)
        self.text_label = self._label("inference.text")
        form.addRow(self.text_label, self.text)
        source_row = QHBoxLayout()
        self.txt_edit = QLineEdit()
        self.txt_edit.textChanged.connect(self._update_actions)
        self.browse_button = QPushButton()
        self.browse_button.clicked.connect(self._browse)
        source_row.addWidget(self.txt_edit, 1)
        source_row.addWidget(self.browse_button)
        self.txt_label = self._label("inference.text_file")
        form.addRow(self.txt_label, source_row)
        self.language = QComboBox()
        for code in ("zh", "ja", "en", "mixed"):
            self.language.addItem("", code)
        self.language_label = self._label("inference.language")
        form.addRow(self.language_label, self.language)
        self.device = QComboBox()
        self.device.addItem("CUDA 0", "cuda:0")
        self.device.addItem("CPU", "cpu")
        self.device_label = self._label("inference.device")
        form.addRow(self.device_label, self.device)
        layout.addLayout(form)

        self.error = QLabel()
        self.error.setWordWrap(True)
        self.error.setStyleSheet(f"color:{FAILED}")
        self.error.hide()
        layout.addWidget(self.error)
        actions = QHBoxLayout()
        self.play_button = QPushButton()
        self.play_button.clicked.connect(self._play)
        self.open_button = QPushButton()
        self.open_button.clicked.connect(self._open_directory)
        self.start_button = QPushButton()
        self.start_button.setObjectName("Primary")
        self.start_button.clicked.connect(self._request)
        actions.addWidget(self.play_button)
        actions.addWidget(self.open_button)
        actions.addStretch(1)
        actions.addWidget(self.start_button)
        layout.addLayout(actions)
        self.toast = QLabel()
        self.toast.setWordWrap(True)
        self.toast.hide()
        layout.addWidget(self.toast)
        layout.addStretch(1)
        self.toast_timer = QTimer(self)
        self.toast_timer.setSingleShot(True)
        self.toast_timer.setInterval(3500)
        self.toast_timer.timeout.connect(self.toast.hide)

        locale_controller.locale_changed.connect(self._locale_changed)
        self.retranslate_ui(self.translator)
        self._update_actions()

    def _label(self, key: str) -> QLabel:
        label = QLabel()
        label.setProperty("translation_key", key)
        return label

    def set_model(self, path: Path | None) -> None:
        self.model = Path(path).resolve() if path is not None else None
        self._update_model_hint()
        self._update_actions()

    def set_busy(self, busy: bool) -> None:
        self._busy = busy
        self.busy_hint.setVisible(busy)
        self.busy_hint.setText(self.translator.text("inference.running"))
        self._update_actions()

    def set_result(self, path: Path) -> None:
        self.result = Path(path).resolve()
        self.error.hide()
        self.toast.setText(
            self.translator.text("inference.output_toast", path=str(self.result.parent))
        )
        self.toast.show()
        self.toast_timer.start()
        self._update_actions()

    def set_error(self, message: str) -> None:
        self.error.setText(message)
        self.error.show()

    def retranslate_ui(self, translator: Translator) -> None:
        self.translator = translator
        self.title.setText(translator.text("inference.title"))
        for label in (
            self.text_label, self.txt_label, self.language_label, self.device_label
        ):
            label.setText(translator.text(label.property("translation_key")))
        self.browse_button.setText(translator.text("inference.browse"))
        for index in range(self.language.count()):
            code = self.language.itemData(index)
            self.language.setItemText(index, translator.text(f"language.{code}"))
        self.start_button.setText(translator.text(
            "inference.running" if self._busy else "inference.start"
        ))
        self.play_button.setText(translator.text("inference.play"))
        self.open_button.setText(translator.text("inference.open"))
        self.busy_hint.setText(translator.text("inference.running"))
        self._update_model_hint()

    def _update_model_hint(self) -> None:
        key = "inference.no_model" if self.model is None else "inference.model_ready"
        values = {} if self.model is None else {"path": str(self.model)}
        self.model_hint.setText(self.translator.text(key, **values))

    def _update_actions(self, *_):
        inline = bool(self.text.toPlainText().strip())
        source = Path(self.txt_edit.text().strip()) if self.txt_edit.text().strip() else None
        valid_file = source is not None and source.is_file() and source.suffix.lower() == ".txt"
        self.start_button.setEnabled(
            self.model is not None
            and not self._busy
            and (inline != valid_file)
            and (source is None or valid_file)
        )
        available = self.result is not None and self.result.is_file() and not self._busy
        self.play_button.setEnabled(available)
        self.open_button.setEnabled(available)
        self.start_button.setText(self.translator.text(
            "inference.running" if self._busy else "inference.start"
        ))

    def _request(self) -> None:
        text = self.text.toPlainText().strip() or None
        source = self.txt_edit.text().strip()
        self.error.hide()
        self.inference_requested.emit(InferenceInput(
            text,
            Path(source).resolve() if source else None,
            str(self.language.currentData()),
            str(self.device.currentData()),
        ))

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, self.translator.text("inference.choose_text"), "", "Text files (*.txt)"
        )
        if path:
            self.txt_edit.setText(path)

    def _play(self) -> None:
        if self.result is not None:
            self.player.play(self.result)

    def _open_directory(self) -> None:
        if self.result is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.result.parent)))

    def _locale_changed(self, locale: str) -> None:
        self.retranslate_ui(Translator(locale))
