from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import QFileDialog, QHBoxLayout, QLineEdit, QPushButton, QVBoxLayout, QWidget

from .i18n import Translator


class SourceInput(QWidget):
    source_changed = Signal(str)

    def __init__(self, translator: Translator | None = None, parent=None):
        super().__init__(parent)
        self.translator = translator or Translator("en")
        self.setAcceptDrops(True)
        self.edit = QLineEdit()
        self.edit.textChanged.connect(self.source_changed)
        self.browse = QPushButton()
        self.browse.clicked.connect(self._browse)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.edit, 1)
        row.addWidget(self.browse)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(row)
        self.retranslate_ui(self.translator)

    def retranslate_ui(self, translator: Translator) -> None:
        self.translator = translator
        self.edit.setPlaceholderText(translator.text("source.placeholder"))
        self.browse.setText(translator.text("source.browse"))

    def value(self) -> str:
        return self.edit.text().strip()

    def set_value(self, value: str) -> None:
        self.edit.setText(value)

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            self.translator.text("source.choose_media"),
            "",
            self.translator.text("source.media_filter"),
        )
        if path:
            self.set_value(path)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls() and len(event.mimeData().urls()) == 1:
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        url = event.mimeData().urls()[0]
        if url.isLocalFile() and Path(url.toLocalFile()).is_file():
            self.set_value(url.toLocalFile())
            event.acceptProposedAction()
