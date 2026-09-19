from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QFormLayout, QFrame, QHBoxLayout, QLabel,
    QLineEdit, QListWidget, QListWidgetItem, QPlainTextEdit, QPushButton,
    QVBoxLayout, QWidget,
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
    model_selected = Signal(object)

    def __init__(self, locale_controller: LocaleController, parent=None, *, player=None):
        super().__init__(parent)
        self.locale_controller = locale_controller
        self.translator = Translator(locale_controller.locale)
        self.player = player or AudioPlayer(self)
        self.model: Path | None = None
        self.result: Path | None = None
        self._results: dict[Path, Path] = {}
        self._busy = False

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 14, 0, 0)
        outer.setSpacing(14)

        history_card = QFrame()
        history_card.setObjectName("Card")
        history_card.setMinimumWidth(210)
        history_card.setMaximumWidth(260)
        history_layout = QVBoxLayout(history_card)
        self.history_title = QLabel()
        self.history_title.setObjectName("CandidateTitle")
        history_layout.addWidget(self.history_title)
        self.history = QListWidget()
        self.history.setObjectName("ModelHistory")
        self.history.currentItemChanged.connect(self._history_changed)
        history_layout.addWidget(self.history, 1)
        outer.addWidget(history_card)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        outer.addWidget(content, 1)

        hero = QFrame()
        hero.setObjectName("InferenceHero")
        hero_layout = QHBoxLayout(hero)
        hero_layout.setContentsMargins(18, 16, 18, 16)
        self.hero_icon = QLabel("♫")
        self.hero_icon.setObjectName("InferenceIcon")
        hero_layout.addWidget(self.hero_icon, 0, Qt.AlignTop)
        hero_text = QVBoxLayout()
        self.title = QLabel()
        self.title.setObjectName("CandidateTitle")
        hero_text.addWidget(self.title)
        self.model_hint = QLabel()
        self.model_hint.setWordWrap(True)
        self.model_hint.setStyleSheet(f"color:{MUTED}")
        hero_text.addWidget(self.model_hint)
        self.busy_hint = QLabel()
        self.busy_hint.setObjectName("InferenceBusy")
        self.busy_hint.hide()
        hero_text.addWidget(self.busy_hint)
        hero_layout.addLayout(hero_text, 1)
        layout.addWidget(hero)

        form_card = QFrame()
        form_card.setObjectName("Card")
        form_layout = QVBoxLayout(form_card)
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
        form_layout.addLayout(form)
        layout.addWidget(form_card)

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

    def set_models(
        self,
        models: tuple[Path, ...],
        *,
        results: dict[Path, Path] | None = None,
        selected: Path | None = None,
    ) -> None:
        resolved = tuple(Path(path).resolve() for path in models)
        self._results = {
            Path(model).resolve(): Path(audio).resolve()
            for model, audio in (results or {}).items()
        }
        preferred = Path(selected).resolve() if selected is not None else None
        self.history.blockSignals(True)
        self.history.clear()
        for model in resolved:
            item = QListWidgetItem(model.name)
            item.setData(Qt.UserRole, str(model))
            self.history.addItem(item)
        if resolved:
            self.history.setCurrentRow(
                resolved.index(preferred) if preferred in resolved else 0
            )
        self.history.blockSignals(False)
        self._select_model(
            resolved[self.history.currentRow()] if resolved else None,
            emit=bool(resolved),
        )

    def set_model(self, path: Path | None) -> None:
        model = Path(path).resolve() if path is not None else None
        if model is not None:
            for index in range(self.history.count()):
                if Path(self.history.item(index).data(Qt.UserRole)) == model:
                    self.history.setCurrentRow(index)
                    self._select_model(model, emit=False)
                    return
            item = QListWidgetItem(model.name)
            item.setData(Qt.UserRole, str(model))
            self.history.insertItem(0, item)
            self.history.setCurrentItem(item)
            return
        self.history.clearSelection()
        self._select_model(None, emit=False)

    def set_busy(self, busy: bool) -> None:
        self._busy = busy
        self.busy_hint.setVisible(busy)
        self.busy_hint.setText(self.translator.text("inference.running"))
        self._update_actions()

    def set_result(self, path: Path | None) -> None:
        self.result = Path(path).resolve() if path is not None else None
        if self.model is not None:
            if self.result is None:
                self._results.pop(self.model, None)
            else:
                self._results[self.model] = self.result
        self.error.hide()
        if self.result is not None:
            self.toast.setText(self.translator.text(
                "inference.output_toast", path=str(self.result.parent)
            ))
            self.toast.show()
            self.toast_timer.start()
        self._update_actions()

    def set_error(self, message: str) -> None:
        self.error.setText(message)
        self.error.show()

    def retranslate_ui(self, translator: Translator) -> None:
        self.translator = translator
        self.history_title.setText(translator.text("inference.history"))
        for label in (
            self.text_label, self.txt_label, self.language_label, self.device_label
        ):
            label.setText(translator.text(label.property("translation_key")))
        self.browse_button.setText(translator.text("inference.browse"))
        for index in range(self.language.count()):
            code = self.language.itemData(index)
            self.language.setItemText(index, translator.text(f"language.{code}"))
        self.play_button.setText(translator.text("inference.play"))
        self.open_button.setText(translator.text("inference.open"))
        self.busy_hint.setText(translator.text("inference.running"))
        self._update_model_hint()
        self._update_actions()

    def _history_changed(self, current, _previous) -> None:
        if current is not None:
            self._select_model(Path(current.data(Qt.UserRole)), emit=True)

    def _select_model(self, model: Path | None, *, emit: bool) -> None:
        self.model = Path(model).resolve() if model is not None else None
        result = self._results.get(self.model) if self.model is not None else None
        self.result = result if result is not None and result.is_file() else None
        self.error.hide()
        self._update_model_hint()
        self._update_actions()
        if emit and self.model is not None:
            self.model_selected.emit(self.model)

    def _update_model_hint(self) -> None:
        ready = self.model is not None
        self.hero_icon.setText("♫" if ready else "○")
        self.title.setText(self.translator.text(
            "inference.ready_title" if ready else "inference.empty_title"
        ))
        key = "inference.model_ready" if ready else "inference.no_model"
        values = {"name": self.model.name} if ready else {}
        self.model_hint.setText(self.translator.text(key, **values))

    def _update_actions(self, *_):
        inline = bool(self.text.toPlainText().strip())
        source = Path(self.txt_edit.text().strip()) if self.txt_edit.text().strip() else None
        self.start_button.setEnabled(
            self.model is not None
            and not self._busy
            and (inline or source is not None)
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
        if text and source:
            self.set_error(self.translator.text("inference.source_conflict"))
            return
        if source and (
            not Path(source).is_file() or Path(source).suffix.lower() != ".txt"
        ):
            self.set_error(self.translator.text("inference.invalid_text_file"))
            return
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
