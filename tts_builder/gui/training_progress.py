from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtWidgets import QGridLayout, QLabel, QProgressBar, QWidget

from ..training_modules.models import ModuleEvent
from .i18n import Translator
from .styles import FAILED, MUTED


_STAGES = ("preprocess", "s2", "s1", "evaluate")


@dataclass
class ProgressRow:
    label: QLabel
    bar: QProgressBar
    status: QLabel
    state: str = "waiting"


class TrainingProgress(QWidget):
    def __init__(self, translator: Translator, parent=None) -> None:
        super().__init__(parent)
        self.translator = translator
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.rows: dict[str, ProgressRow] = {}
        for index, stage in enumerate(_STAGES):
            label = QLabel()
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            status = QLabel()
            status.setStyleSheet(f"color:{MUTED}")
            layout.addWidget(label, index, 0)
            layout.addWidget(bar, index, 1)
            layout.addWidget(status, index, 2)
            self.rows[stage] = ProgressRow(label, bar, status)
        self.retranslate_ui(translator)

    def consume(self, event: ModuleEvent) -> None:
        stage = _stage(event.stage)
        if stage not in self.rows:
            return
        row = self.rows[stage]
        states = {
            "stage_started": "running",
            "stage_progress": "running",
            "stage_cache_hit": "cached",
            "stage_completed": "done",
            "stage_failed": "failed",
            "stage_cancelled": "stopped",
        }
        if event.type in states:
            row.state = states[event.type]
        if event.type == "stage_progress" and event.current is not None and event.total:
            row.bar.setRange(0, 100)
            row.bar.setValue(min(100, round(event.current / event.total * 100)))
        elif row.state == "running":
            row.bar.setRange(0, 100)
            if event.type == "stage_started":
                row.bar.setValue(0)
        elif row.state in {"cached", "done"}:
            row.bar.setRange(0, 100)
            row.bar.setValue(100)
        elif row.state in {"failed", "stopped"}:
            row.bar.setRange(0, 100)
        self._apply_status(row)

    def retranslate_ui(self, translator: Translator) -> None:
        self.translator = translator
        for stage, row in self.rows.items():
            row.label.setText(translator.text(f"training.stage.{stage}"))
            self._apply_status(row)

    def _apply_status(self, row: ProgressRow) -> None:
        row.status.setText(self.translator.text(f"training.state.{row.state}"))
        row.status.setStyleSheet(f"color:{FAILED if row.state == 'failed' else MUTED}")


def _stage(value: str | None) -> str | None:
    return value.rsplit(".", 1)[-1] if value else None
