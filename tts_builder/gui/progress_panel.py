from __future__ import annotations

from PySide6.QtWidgets import QGridLayout, QLabel, QProgressBar, QVBoxLayout, QWidget

from .models import ProgressModel, STAGES
from .styles import FAILED, GREEN, MUTED, PENDING
from .i18n import Translator


class ProgressPanel(QWidget):
    def __init__(self, translator: Translator | None = None, parent=None):
        super().__init__(parent)
        self.translator = translator or Translator("en")
        self._model = ProgressModel()
        self.rows = {}
        self.names = {}
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(8)
        for row, stage in enumerate(STAGES):
            icon = QLabel("○")
            name = QLabel()
            status = QLabel()
            status.setStyleSheet(f"color:{MUTED}")
            grid.addWidget(icon, row, 0)
            grid.addWidget(name, row, 1)
            grid.addWidget(status, row, 2)
            self.rows[stage] = (icon, status)
            self.names[stage] = name
        self.total = QProgressBar()
        self.total.setRange(0, 100)
        self.total.setValue(0)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(grid)
        layout.addSpacing(8)
        layout.addWidget(self.total)
        self.retranslate_ui(self.translator)

    def retranslate_ui(self, translator: Translator) -> None:
        self.translator = translator
        for stage, name in self.names.items():
            name.setText(translator.text(f"progress.{stage}"))
        self.apply(self._model)

    def apply(self, model: ProgressModel) -> None:
        self._model = model
        for stage, state in model.stages.items():
            icon, label = self.rows[stage]
            if state.status in {"completed", "cached"}:
                icon.setText("✓")
                icon.setStyleSheet(f"color:{GREEN};font-weight:700")
                key = "progress.cached" if state.status == "cached" else "progress.done"
                label.setText(state.message or self.translator.text(key))
            elif state.status == "running":
                icon.setText("●")
                icon.setStyleSheet(f"color:{GREEN}")
                label.setText(state.message or (f"{state.percent}%" if state.percent else self.translator.text("progress.working")))
            elif state.status == "failed":
                icon.setText("!")
                icon.setStyleSheet(f"color:{FAILED};font-weight:700")
                label.setText(state.message or self.translator.text("progress.failed"))
            else:
                icon.setText("○")
                icon.setStyleSheet(f"color:{PENDING}")
                label.setText(self.translator.text("progress.waiting"))
            label.setStyleSheet(f"color:{MUTED}")
        self.total.setValue(model.overall_percent)
