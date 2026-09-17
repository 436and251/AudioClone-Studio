from __future__ import annotations

from pathlib import Path
import re
from typing import Sequence

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

from ..training_modules.models import FieldDescriptor, FrameworkDescriptor, ModuleDescriptor
from .i18n import Translator
from .settings import TrainingModuleSetting


_STAGES = ("preprocess", "s2", "s1", "evaluate")
_PROJECT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")
ModuleBinding = tuple[TrainingModuleSetting | None, ModuleDescriptor]


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

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        common = QFormLayout()
        self.framework_label = QLabel()
        self.project_name_label = QLabel()
        self.project_label = QLabel()
        self.dataset_label = QLabel()
        self.output_label = QLabel()
        self.device_label = QLabel()
        self.precision_label = QLabel()
        self.reference_label = QLabel()
        self.reference_text_label = QLabel()
        self.reference_language_label = QLabel()
        self.stages_label = QLabel()

        self.framework_combo = QComboBox()
        for binding_index, (_, module) in enumerate(self.bindings):
            for framework_index, framework in enumerate(module.frameworks):
                self.framework_combo.addItem(
                    framework.display_name, (binding_index, framework_index)
                )
        self.project_name = QLineEdit()
        self.project_edit = QLineEdit()
        self.dataset_edit = QLineEdit()
        self.output_edit = QLineEdit()
        self.device = QComboBox()
        self.device.addItems(["cuda:0", "cpu"])
        self.precision = QComboBox()
        self.precision.addItems(["fp16", "fp32"])
        common.addRow(self.framework_label, self.framework_combo)
        common.addRow(self.project_name_label, self.project_name)
        common.addRow(
            self.project_label, self._path_row(self.project_edit, self._browse_project)
        )
        common.addRow(
            self.dataset_label, self._path_row(self.dataset_edit, self._browse_dataset)
        )
        common.addRow(
            self.output_label, self._path_row(self.output_edit, self._browse_output)
        )
        common.addRow(self.device_label, self.device)
        common.addRow(self.precision_label, self.precision)

        stages = QWidget()
        stage_layout = QHBoxLayout(stages)
        stage_layout.setContentsMargins(0, 0, 0, 0)
        self.stage_boxes = {stage: QCheckBox() for stage in _STAGES}
        for box in self.stage_boxes.values():
            box.setChecked(True)
            stage_layout.addWidget(box)
        stage_layout.addStretch(1)
        common.addRow(self.stages_label, stages)

        self.reference_audio = QLineEdit()
        self.reference_text = QLineEdit()
        self.reference_language = QComboBox()
        for language in ("ja", "zh", "en"):
            self.reference_language.addItem("", language)
        common.addRow(
            self.reference_label,
            self._path_row(self.reference_audio, self._browse_reference),
        )
        common.addRow(self.reference_text_label, self.reference_text)
        common.addRow(self.reference_language_label, self.reference_language)
        layout.addLayout(common)

        self.advanced_group = QGroupBox()
        self.advanced_layout = QFormLayout(self.advanced_group)
        layout.addWidget(self.advanced_group)

        self.framework_combo.currentIndexChanged.connect(self._framework_changed)
        for edit in (
            self.project_name, self.project_edit, self.dataset_edit, self.output_edit,
            self.reference_audio, self.reference_text,
        ):
            edit.textChanged.connect(self._changed)
        for combo in (self.device, self.precision, self.reference_language):
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

    def values(self) -> dict[str, object]:
        reference = None
        if self.stage_boxes["evaluate"].isChecked():
            reference = {
                "audio": str(Path(self.reference_audio.text()).resolve()),
                "text": self.reference_text.text().strip() or None,
                "language": self.reference_language.currentData(),
            }
        return {
            "project_name": self.project_name.text().strip(),
            "output_root": Path(self.output_edit.text()).resolve(),
            "device": self.device.currentText(),
            "precision": self.precision.currentText(),
            "parameters": {
                key: _widget_value(self.advanced_fields[key], descriptor.kind)
                for key, descriptor in self._field_descriptors.items()
            },
            "reference": reference,
        }

    def is_valid(self) -> bool:
        if not self.bindings or self.framework_combo.currentIndex() < 0:
            return False
        setting, _, framework = self.selection()
        project = Path(self.project_edit.text())
        training_data = Path(self.dataset_edit.text())
        output = Path(self.output_edit.text())
        if (
            setting is None
            or _PROJECT_NAME.fullmatch(self.project_name.text().strip()) is None
            or not project.is_absolute()
            or not project.is_dir()
            or not _valid_training_data(training_data, project, framework)
            or not _contained_path(output, project)
            or not self.selected_stages()
        ):
            return False
        if self.stage_boxes["evaluate"].isChecked() and not _contained_file(
            Path(self.reference_audio.text()), project
        ):
            return False
        return all(
            descriptor.kind != "path"
            or _contained_path(Path(self.advanced_fields[key].text()), project)
            for key, descriptor in self._field_descriptors.items()
        )

    def prefill_dataset(self, path: Path) -> None:
        dataset = Path(path).resolve()
        project = dataset.parent
        self.dataset_edit.setText(str(dataset))
        self.project_edit.setText(str(project))
        self.output_edit.setText(str((project / "runs").resolve()))
        self.project_name.setText(project.name)

    def retranslate_ui(self, translator: Translator) -> None:
        self.translator = translator
        labels = {
            self.framework_label: "training.framework",
            self.project_name_label: "training.project_name",
            self.project_label: "training.project_dir",
            self.dataset_label: "training.dataset",
            self.output_label: "training.output",
            self.device_label: "training.device",
            self.precision_label: "training.precision",
            self.stages_label: "training.stages",
            self.reference_label: "training.reference_audio",
            self.reference_text_label: "training.reference_text",
            self.reference_language_label: "training.reference_language",
        }
        for label, key in labels.items():
            label.setText(translator.text(key))
        for stage, box in self.stage_boxes.items():
            box.setText(translator.text(f"training.stage.{stage}"))
        for index in range(self.reference_language.count()):
            code = self.reference_language.itemData(index)
            self.reference_language.setItemText(index, translator.text(f"language.{code}"))
        self.advanced_group.setTitle(translator.text("training.advanced"))
        for key, label in self._advanced_labels.items():
            label.setText(self._field_descriptors[key].labels[translator.locale])

    def _framework_changed(self, *_):
        while self.advanced_layout.rowCount():
            self.advanced_layout.removeRow(0)
        self.advanced_fields.clear()
        self._field_descriptors.clear()
        self._advanced_labels.clear()
        if not self.bindings or self.framework_combo.currentIndex() < 0:
            self._changed()
            return
        _, _, framework = self.selection()
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
        for descriptor in framework.fields:
            widget = _field_widget(descriptor)
            label = QLabel(descriptor.labels[self.translator.locale])
            self.advanced_fields[descriptor.key] = widget
            self._field_descriptors[descriptor.key] = descriptor
            self._advanced_labels[descriptor.key] = label
            self.advanced_layout.addRow(label, widget)
            _connect_change(widget, self._changed)
        self._changed()

    def _changed(self, *_):
        self.validity_changed.emit(self.is_valid())

    def _browse_project(self):
        self._choose_directory(self.project_edit)

    def _browse_output(self):
        self._choose_directory(self.output_edit)

    def _choose_directory(self, edit: QLineEdit):
        path = QFileDialog.getExistingDirectory(self, "", edit.text())
        if path:
            edit.setText(path)

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

    def _browse_reference(self):
        path, _ = QFileDialog.getOpenFileName(self, "", self.reference_audio.text(), "WAV (*.wav);;All files (*)")
        if path:
            self.reference_audio.setText(path)


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


def _contained_file(path: Path, root: Path) -> bool:
    return _contained_path(path, root) and path.resolve().is_file()


def _valid_training_data(
    path: Path,
    root: Path,
    framework: FrameworkDescriptor,
) -> bool:
    if not _contained_path(path, root):
        return False
    descriptor = framework.training_data
    resolved = path.resolve()
    if descriptor.kind == "directory":
        return resolved.is_dir()
    extensions = {extension.casefold() for extension in descriptor.extensions}
    return resolved.is_file() and (
        not extensions or resolved.suffix.casefold() in extensions
    )
