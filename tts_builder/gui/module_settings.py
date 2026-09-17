from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFileDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QPushButton, QVBoxLayout, QWidget,
)

from ..training_modules.models import ProbeResult
from ..training_modules.probe import probe_module
from .i18n import Translator
from .settings import TrainingModuleSetting
from .styles import MUTED


class ModuleSettings(QWidget):
    validity_changed = Signal(bool)

    def __init__(
        self,
        modules: Sequence[TrainingModuleSetting],
        translator: Translator,
        parent=None,
        *,
        probe: Callable[[TrainingModuleSetting], ProbeResult] = probe_module,
    ):
        super().__init__(parent)
        self.translator = translator
        self._probe = probe
        self._loading = False
        self._drafts = [
            {
                "name": module.name,
                "project_root": str(module.project_root),
                "python_executable": str(module.python_executable),
                "module_name": module.module_name,
            }
            for module in modules
        ]

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 8, 0, 0)
        self.title = QLabel()
        self.title.setObjectName("Subtitle")
        layout.addWidget(self.title)
        body = QHBoxLayout()
        self.module_list = QListWidget()
        self.module_list.setMinimumWidth(170)
        body.addWidget(self.module_list, 1)

        form = QFormLayout()
        self.name_edit = QLineEdit()
        self.project_edit = QLineEdit()
        self.python_edit = QLineEdit()
        self.module_edit = QLineEdit()
        self.name_label = QLabel()
        self.project_label = QLabel()
        self.python_label = QLabel()
        self.module_label = QLabel()
        self.project_browse = QPushButton()
        self.python_browse = QPushButton()
        form.addRow(self.name_label, self.name_edit)
        form.addRow(self.project_label, self._path_row(self.project_edit, self.project_browse))
        form.addRow(self.python_label, self._path_row(self.python_edit, self.python_browse))
        form.addRow(self.module_label, self.module_edit)
        body.addLayout(form, 2)
        layout.addLayout(body)

        actions = QHBoxLayout()
        self.add_button = QPushButton()
        self.remove_button = QPushButton()
        self.check_button = QPushButton()
        self.connection_status = QLabel()
        self.connection_status.setStyleSheet(f"color:{MUTED}")
        actions.addWidget(self.add_button)
        actions.addWidget(self.remove_button)
        actions.addStretch(1)
        actions.addWidget(self.connection_status)
        actions.addWidget(self.check_button)
        layout.addLayout(actions)

        self.module_list.currentRowChanged.connect(self._load_current)
        for edit in (self.name_edit, self.project_edit, self.python_edit, self.module_edit):
            edit.textChanged.connect(self._update_current)
        self.add_button.clicked.connect(self._add)
        self.remove_button.clicked.connect(self._remove)
        self.project_browse.clicked.connect(self._browse_project)
        self.python_browse.clicked.connect(self._browse_python)
        self.check_button.clicked.connect(self._check_connection)

        self._refresh_list()
        self.retranslate_ui(translator)
        self._load_current(0 if self._drafts else -1)

    @staticmethod
    def _path_row(edit: QLineEdit, button: QPushButton) -> QWidget:
        container = QWidget()
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(edit, 1)
        row.addWidget(button)
        return container

    def retranslate_ui(self, translator: Translator) -> None:
        self.translator = translator
        self.title.setText(translator.text("modules.title"))
        self.name_label.setText(translator.text("modules.name"))
        self.project_label.setText(translator.text("modules.project"))
        self.python_label.setText(translator.text("modules.python"))
        self.module_label.setText(translator.text("modules.entry"))
        self.project_browse.setText(translator.text("source.browse"))
        self.python_browse.setText(translator.text("source.browse"))
        self.add_button.setText(translator.text("modules.add"))
        self.remove_button.setText(translator.text("modules.remove"))
        self.check_button.setText(translator.text("modules.check"))
        self._refresh_list()

    def modules(self) -> tuple[TrainingModuleSetting, ...]:
        if not self.is_valid():
            raise ValueError("training module settings are incomplete")
        return tuple(self._setting(draft) for draft in self._drafts)

    def is_valid(self) -> bool:
        return all(self._draft_valid(draft) for draft in self._drafts)

    def _refresh_list(self) -> None:
        current = self.module_list.currentRow()
        self.module_list.clear()
        for draft in self._drafts:
            self.module_list.addItem(draft["name"] or self.translator.text("modules.new"))
        if self._drafts:
            self.module_list.setCurrentRow(max(0, min(current, len(self._drafts) - 1)))

    def _load_current(self, row: int) -> None:
        self._loading = True
        draft = self._drafts[row] if 0 <= row < len(self._drafts) else None
        edits = (
            (self.name_edit, "name"),
            (self.project_edit, "project_root"),
            (self.python_edit, "python_executable"),
            (self.module_edit, "module_name"),
        )
        for edit, key in edits:
            edit.setText(draft[key] if draft else "")
            edit.setEnabled(draft is not None)
        self._loading = False
        self.remove_button.setEnabled(draft is not None)
        self.connection_status.clear()
        self._emit_validity()

    def _update_current(self, *_) -> None:
        if self._loading:
            return
        row = self.module_list.currentRow()
        if not 0 <= row < len(self._drafts):
            return
        draft = self._drafts[row]
        draft.update(
            name=self.name_edit.text().strip(),
            project_root=self.project_edit.text().strip(),
            python_executable=self.python_edit.text().strip(),
            module_name=self.module_edit.text().strip(),
        )
        self.module_list.item(row).setText(
            draft["name"] or self.translator.text("modules.new")
        )
        self.connection_status.clear()
        self._emit_validity()

    def _add(self) -> None:
        self._drafts.append(
            {"name": "", "project_root": "", "python_executable": "", "module_name": ""}
        )
        self._refresh_list()
        self.module_list.setCurrentRow(len(self._drafts) - 1)

    def _remove(self) -> None:
        row = self.module_list.currentRow()
        if 0 <= row < len(self._drafts):
            self._drafts.pop(row)
            self._refresh_list()
            self._load_current(self.module_list.currentRow())

    def _browse_project(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self, self.translator.text("modules.choose_project"), self.project_edit.text()
        )
        if chosen:
            self.project_edit.setText(chosen)

    def _browse_python(self) -> None:
        chosen, _ = QFileDialog.getOpenFileName(
            self, self.translator.text("modules.choose_python"), self.python_edit.text()
        )
        if chosen:
            self.python_edit.setText(chosen)

    def _check_connection(self) -> None:
        row = self.module_list.currentRow()
        if not 0 <= row < len(self._drafts) or not self._draft_valid(self._drafts[row]):
            return
        result = self._probe(self._setting(self._drafts[row]))
        if result.available and result.descriptor is not None:
            framework = result.descriptor.frameworks[0].display_name
            self.connection_status.setText(
                self.translator.text("modules.connected", framework=framework)
            )
        else:
            self.connection_status.setText(self.translator.text("modules.unavailable"))

    def _emit_validity(self) -> None:
        valid = self.is_valid()
        self.check_button.setEnabled(
            0 <= self.module_list.currentRow() < len(self._drafts)
            and self._draft_valid(self._drafts[self.module_list.currentRow()])
        )
        self.validity_changed.emit(valid)

    @staticmethod
    def _setting(draft: dict[str, str]) -> TrainingModuleSetting:
        return TrainingModuleSetting(
            draft["name"],
            Path(draft["project_root"]),
            Path(draft["python_executable"]),
            draft["module_name"],
        )

    @staticmethod
    def _draft_valid(draft: dict[str, str]) -> bool:
        project = Path(draft["project_root"])
        python = Path(draft["python_executable"])
        return bool(
            draft["name"]
            and draft["module_name"]
            and project.is_absolute()
            and project.is_dir()
            and python.is_absolute()
            and python.is_file()
        )
