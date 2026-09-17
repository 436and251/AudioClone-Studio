from __future__ import annotations

from datetime import datetime

from PySide6.QtWidgets import QPlainTextEdit, QToolButton, QVBoxLayout, QWidget

from ..events import PipelineEvent
from .i18n import Translator


class LogPanel(QWidget):
    def __init__(self, translator: Translator | None = None, parent=None):
        super().__init__(parent)
        self.translator = translator or Translator("en")
        self.toggle = QToolButton()
        self.toggle.setCheckable(True)
        self.toggle.toggled.connect(self._toggle)
        self.view = QPlainTextEdit()
        self.view.setReadOnly(True)
        self.view.setMaximumBlockCount(800)
        self.view.setVisible(False)
        self.view.setMaximumHeight(170)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toggle)
        layout.addWidget(self.view)
        self.retranslate_ui(self.translator)

    def retranslate_ui(self, translator: Translator) -> None:
        self.translator = translator
        marker = "▾" if self.toggle.isChecked() else "▸"
        self.toggle.setText(f"{marker} {translator.text('logs.activity')}")

    def _toggle(self, checked: bool) -> None:
        marker = "▾" if checked else "▸"
        self.toggle.setText(f"{marker} {self.translator.text('logs.activity')}")
        self.view.setVisible(checked)

    def append_event(self, event: PipelineEvent) -> None:
        if not event.message:
            return
        stage = f"[{event.stage}] " if event.stage else ""
        self.append_text(f"{stage}{event.message}")

    def append_text(self, text: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        self.view.appendPlainText(f"{stamp}  {text}")
