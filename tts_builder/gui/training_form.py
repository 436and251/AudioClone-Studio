from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Sequence

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QFrame,
    QGridLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QSpinBox, QToolButton,
    QVBoxLayout, QWidget,
)

from ..training_modules.models import FieldDescriptor, FrameworkDescriptor, ModuleDescriptor
from .i18n import Translator
from .settings import TrainingModuleSetting


_STAGES = ("preprocess", "s2", "s1", "evaluate")
_PROJECT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")
ModuleBinding = tuple[TrainingModuleSetting | None, ModuleDescriptor]


@dataclass(frozen=True, slots=True)
class TrainingSelection:
    setting: TrainingModuleSetting | None
    module: ModuleDescriptor
    framework: FrameworkDescriptor
    project_dir: Path
    values: dict[str, object]
    stages: tuple[str, ...]
    training_data: Path


class TrainingForm(QWidget):
    validity_changed = Signal(bool)

    def __init__(
        self,
        bindings: Sequence[ModuleBinding],
        translator: Translator,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.bindings = tuple(bindings)
        self.translator = translator
        self.advanced_fields: dict[str, QWidget] = {}
        self._field_descriptors: dict[str, FieldDescriptor] = {}
        self._advanced_labels: dict[str, QLabel] = {}
        self.advanced_groups: dict[str, QFrame] = {}
        self.advanced_group_titles: dict[str, QLabel] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.common_layout = QGridLayout()
        self.common_layout.setHorizontalSpacing(12)
        self.common_layout.setVerticalSpacing(10)
        self.common_layout.setColumnStretch(1, 1)
        self.common_layout.setColumnStretch(3, 1)
        self.framework_label = QLabel()
        self.project_name_label = QLabel()
        self.dataset_label = QLabel()
        self.output_label = QLabel()
        self.device_label = QLabel()
        self.precision_label = QLabel()
        self.stages_label = QLabel()

        self.framework_combo = QComboBox()
        for binding_index, (_, module) in enumerate(self.bindings):
            for framework_index, framework in enumerate(module.frameworks):
                self.framework_combo.addItem(
                    framework.display_name, (binding_index, framework_index)
                )
        self.project_name = QLineEdit()
        self.dataset_edit = QLineEdit()
        self.output_edit = QLineEdit()
        self.output_edit.setReadOnly(True)
        self.device = QComboBox()
        self.device.addItems(["cuda:0", "cpu"])
        self.precision = QComboBox()
        self.precision.addItems(["fp16", "fp32"])
        self.common_layout.addWidget(self.framework_label, 0, 0)
        self.common_layout.addWidget(self.framework_combo, 0, 1)
        self.common_layout.addWidget(self.project_name_label, 0, 2)
        self.common_layout.addWidget(self.project_name, 0, 3)
        self.common_layout.addWidget(self.dataset_label, 1, 0)
        self.common_layout.addWidget(
            self._path_row(self.dataset_edit, self._browse_dataset), 1, 1, 1, 3
        )
        self.common_layout.addWidget(self.output_label, 2, 0)
        self.common_layout.addWidget(self.output_edit, 2, 1, 1, 3)
        self.common_layout.addWidget(self.device_label, 3, 0)
        self.common_layout.addWidget(self.device, 3, 1)
        self.common_layout.addWidget(self.precision_label, 3, 2)
        self.common_layout.addWidget(self.precision, 3, 3)

        stages = QWidget()
        stage_layout = QHBoxLayout(stages)
        stage_layout.setContentsMargins(0, 0, 0, 0)
        self.stage_boxes = {stage: QCheckBox() for stage in _STAGES}
        for box in self.stage_boxes.values():
            box.setChecked(True)
            stage_layout.addWidget(box)
        stage_layout.addStretch(1)
        self.common_layout.addWidget(self.stages_label, 4, 0)
        self.common_layout.addWidget(stages, 4, 1, 1, 3)

        layout.addLayout(self.common_layout)

        self.advanced_toggle = QToolButton()
        self.advanced_toggle.setObjectName("AdvancedToggle")
        self.advanced_toggle.setCheckable(True)
        self.advanced_toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.advanced_toggle.setArrowType(Qt.RightArrow)
        layout.addWidget(self.advanced_toggle)
        self.advanced_content = QWidget()
        self.advanced_content.setObjectName("AdvancedContent")
        self.advanced_layout = QGridLayout(self.advanced_content)
        self.advanced_layout.setContentsMargins(12, 12, 12, 12)
        self.advanced_layout.setHorizontalSpacing(12)
        self.advanced_content.hide()
        layout.addWidget(self.advanced_content)

        self.framework_combo.currentIndexChanged.connect(self._framework_changed)
        self.advanced_toggle.toggled.connect(self._toggle_advanced)
        for edit in (self.project_name, self.dataset_edit):
            edit.textChanged.connect(self._changed)
        for combo in (self.device, self.precision):
            combo.currentIndexChanged.connect(self._changed)
        for box in self.stage_boxes.values():
            box.toggled.connect(self._changed)
        self._framework_changed()
        self.retranslate_ui(translator)

    def _path_row(self, edit: QLineEdit, callback) -> QWidget:
        container = QWidget()
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        button = QPushButton("…")
        button.setFixedWidth(42)
        button.clicked.connect(callback)
        row.addWidget(edit, 1)
        row.addWidget(button)
        return container

    def selection(self) -> tuple[TrainingModuleSetting | None, ModuleDescriptor, FrameworkDescriptor]:
        binding_index, framework_index = self.framework_combo.currentData()
        setting, module = self.bindings[binding_index]
        return setting, module, module.frameworks[framework_index]

    def selected_stages(self) -> tuple[str, ...]:
        return tuple(stage for stage in _STAGES if self.stage_boxes[stage].isChecked())

    def training_data_path(self) -> Path:
        return Path(self.dataset_edit.text()).resolve()

    def snapshot(self) -> TrainingSelection:
        setting, module, framework = self.selection()
        if setting is None:
            raise ValueError("training module is unavailable")
        return TrainingSelection(
            setting=setting,
            module=module,
            framework=framework,
            project_dir=Path(setting.project_root).resolve(),
            values=self.values(),
            stages=self.selected_stages(),
            training_data=self.training_data_path(),
        )

    def values(self) -> dict[str, object]:
        return {
            "project_name": self.project_name.text().strip(),
            "output_root": Path(self.output_edit.text()).resolve(),
            "device": self.device.currentText(),
            "precision": self.precision.currentText(),
            "parameters": {
                key: _widget_value(self.advanced_fields[key], descriptor.kind)
                for key, descriptor in self._field_descriptors.items()
            },
            "reference": None,
        }

    def is_valid(self) -> bool:
        if not self.bindings or self.framework_combo.currentIndex() < 0:
            return False
        setting, _, framework = self.selection()
        if setting is None:
            return False
        project = Path(setting.project_root)
        training_data = Path(self.dataset_edit.text())
        output = Path(self.output_edit.text())
        if (
            _PROJECT_NAME.fullmatch(self.project_name.text().strip()) is None
            or not project.is_absolute()
            or not project.is_dir()
            or not _valid_training_data(training_data, framework)
            or not _contained_path(output, project)
            or not self.selected_stages()
        ):
            return False
        return all(
            descriptor.kind != "path"
            or _contained_path(Path(self.advanced_fields[key].text()), project)
            for key, descriptor in self._field_descriptors.items()
        )

    def prefill_dataset(self, path: Path) -> None:
        dataset = Path(path).resolve()
        self.dataset_edit.setText(str(dataset))
        self.project_name.setText(dataset.parent.name)
        self._sync_output_root()

    def retranslate_ui(self, translator: Translator) -> None:
        self.translator = translator
        labels = {
            self.framework_label: "training.framework",
            self.project_name_label: "training.project_name",
            self.dataset_label: "training.dataset",
            self.output_label: "training.output",
            self.device_label: "training.device",
            self.precision_label: "training.precision",
            self.stages_label: "training.stages",
        }
        for label, key in labels.items():
            label.setText(translator.text(key))
        for stage, box in self.stage_boxes.items():
            box.setText(translator.text(f"training.stage.{stage}"))
        self.advanced_toggle.setText(translator.text("training.advanced"))
        for key, label in self.advanced_group_titles.items():
            label.setText(translator.text(f"training.advanced_group.{key}"))
        for key, label in self._advanced_labels.items():
            label.setText(self._field_descriptors[key].labels[translator.locale])

    def _framework_changed(self, *_):
        while self.advanced_layout.count():
            item = self.advanced_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self.advanced_fields.clear()
        self._field_descriptors.clear()
        self._advanced_labels.clear()
        self.advanced_groups.clear()
        self.advanced_group_titles.clear()
        if not self.bindings or self.framework_combo.currentIndex() < 0:
            self._changed()
            return
        _, _, framework = self.selection()
        self._sync_output_root()
        available = set()
        if "preprocess" in framework.capabilities:
            available.add("preprocess")
        if "train" in framework.capabilities:
            available.update(("s2", "s1"))
        if "evaluate" in framework.capabilities:
            available.add("evaluate")
        for stage, box in self.stage_boxes.items():
            box.setVisible(stage in available)
            box.setEnabled(stage in available)
            box.setChecked(stage in available)
        grouped = {
            key: [field for field in framework.fields if _advanced_group(field.key) == key]
            for key in ("general", "s1", "s2")
        }
        for column, key in enumerate(key for key, fields in grouped.items() if fields):
            frame = QFrame()
            frame.setObjectName("AdvancedGroup")
            group_layout = QVBoxLayout(frame)
            title = QLabel(self.translator.text(f"training.advanced_group.{key}"))
            title.setObjectName("AdvancedGroupTitle")
            group_layout.addWidget(title)
            fields_layout = QFormLayout()
            fields_layout.setContentsMargins(0, 0, 0, 0)
            fields_layout.setVerticalSpacing(10)
            for descriptor in grouped[key]:
                widget = _field_widget(descriptor)
                label = QLabel(descriptor.labels[self.translator.locale])
                self.advanced_fields[descriptor.key] = widget
                self._field_descriptors[descriptor.key] = descriptor
                self._advanced_labels[descriptor.key] = label
                fields_layout.addRow(label, widget)
                _connect_change(widget, self._changed)
            group_layout.addLayout(fields_layout)
            group_layout.addStretch(1)
            self.advanced_groups[key] = frame
            self.advanced_group_titles[key] = title
            self.advanced_layout.addWidget(frame, 0, column)
            self.advanced_layout.setColumnStretch(column, 1)
        self._changed()

    def _toggle_advanced(self, expanded: bool) -> None:
        self.advanced_toggle.setArrowType(
            Qt.DownArrow if expanded else Qt.RightArrow
        )
        self.advanced_content.setVisible(expanded)

    def _changed(self, *_):
        self.validity_changed.emit(self.is_valid())

    def _sync_output_root(self) -> None:
        if not self.bindings or self.framework_combo.currentIndex() < 0:
            self.output_edit.clear()
            return
        setting, _, _ = self.selection()
        self.output_edit.setText(
            str((Path(setting.project_root) / "runs").resolve()) if setting else ""
        )

    def _browse_dataset(self):
        _, _, framework = self.selection()
        if framework.training_data.kind == "directory":
            path = QFileDialog.getExistingDirectory(self, "", self.dataset_edit.text())
        else:
            patterns = " ".join(f"*{extension}" for extension in framework.training_data.extensions)
            file_filter = f"Training data ({patterns});;All files (*)" if patterns else "All files (*)"
            path, _ = QFileDialog.getOpenFileName(
                self, "", self.dataset_edit.text(), file_filter
            )
        if path:
            self.dataset_edit.setText(str(Path(path).resolve()))

def _field_widget(field: FieldDescriptor) -> QWidget:
    if field.kind == "integer":
        widget = QSpinBox()
        minimum = int(field.constraints.get("minimum", -2_147_483_648))
        maximum = int(field.constraints.get("maximum", 2_147_483_647))
        if "exclusive_minimum" in field.constraints:
            minimum = int(field.constraints["exclusive_minimum"]) + 1
        if "exclusive_maximum" in field.constraints:
            maximum = int(field.constraints["exclusive_maximum"]) - 1
        widget.setRange(minimum, maximum)
        widget.setValue(field.default)
        return widget
    if field.kind == "number":
        widget = QDoubleSpinBox()
        minimum = float(field.constraints.get("minimum", -1e100))
        maximum = float(field.constraints.get("maximum", 1e100))
        if "exclusive_minimum" in field.constraints:
            minimum = float(field.constraints["exclusive_minimum"]) + 1e-10
        if "exclusive_maximum" in field.constraints:
            maximum = float(field.constraints["exclusive_maximum"]) - 1e-10
        widget.setDecimals(10)
        widget.setRange(minimum, maximum)
        widget.setValue(field.default)
        return widget
    if field.kind == "boolean":
        widget = QCheckBox()
        widget.setChecked(field.default)
        return widget
    if field.kind == "enum":
        widget = QComboBox()
        widget.addItems(field.constraints["choices"])
        widget.setCurrentText(field.default)
        return widget
    widget = QLineEdit(str(field.default))
    return widget


def _advanced_group(key: str) -> str:
    prefix = key.partition(".")[0]
    return prefix if prefix in {"s1", "s2"} else "general"


def _connect_change(widget: QWidget, callback) -> None:
    if isinstance(widget, QLineEdit):
        widget.textChanged.connect(callback)
    elif isinstance(widget, QCheckBox):
        widget.toggled.connect(callback)
    elif isinstance(widget, QComboBox):
        widget.currentIndexChanged.connect(callback)
    else:
        widget.valueChanged.connect(callback)


def _widget_value(widget: QWidget, kind: str) -> object:
    if kind == "boolean":
        return widget.isChecked()
    if kind == "enum":
        return widget.currentText()
    if kind in {"integer", "number"}:
        return widget.value()
    return widget.text()


def _contained_path(path: Path, root: Path) -> bool:
    if not path.is_absolute():
        return False
    resolved = path.resolve()
    root = root.resolve()
    return resolved == root or resolved.is_relative_to(root)


def _valid_training_data(
    path: Path,
    framework: FrameworkDescriptor,
) -> bool:
    if not path.is_absolute():
        return False
    descriptor = framework.training_data
    resolved = path.resolve()
    if descriptor.kind == "directory":
        return resolved.is_dir()
    extensions = {extension.casefold() for extension in descriptor.extensions}
    return resolved.is_file() and (
        not extensions or resolved.suffix.casefold() in extensions
    )
