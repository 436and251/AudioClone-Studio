import os
from dataclasses import FrozenInstanceError
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from tts_builder.gui.app import create_application
from tts_builder.gui.i18n import Translator
from tts_builder.gui.settings import TrainingModuleSetting
from tts_builder.training_modules.models import (
    FieldDescriptor,
    FrameworkDescriptor,
    ModuleDescriptor,
    TrainingDataDescriptor,
)


def _binding(tmp_path: Path):
    module_root = tmp_path / "module"
    python = tmp_path / "venv" / "python.exe"
    module_root.mkdir()
    python.parent.mkdir()
    python.write_bytes(b"")
    setting = TrainingModuleSetting("GPT-SoVITS", module_root, python, "voice_pipeline")
    labels = {"en": "Value", "zh_CN": "数值", "ja": "値"}
    fields = (
        FieldDescriptor("steps", "integer", 10, {"minimum": 1, "maximum": 20}, labels),
        FieldDescriptor("rate", "number", 0.1, {"exclusive_minimum": 0}, labels),
        FieldDescriptor("resume", "boolean", True, {}, labels),
        FieldDescriptor("mode", "enum", "safe", {"choices": ["safe", "fast"]}, labels),
        FieldDescriptor("resource", "path", str(module_root), {}, labels),
        FieldDescriptor("note", "string", "hello", {}, labels),
    )
    framework = FrameworkDescriptor(
        "v2ProPlus", "GPT-SoVITS v2ProPlus",
        ("preprocess", "train", "evaluate", "listen", "promote", "infer"),
        TrainingDataDescriptor("file", (".list",)), fields,
    )
    return setting, ModuleDescriptor(2, "gpt-sovits", "1.0", (framework,))


def _project(tmp_path: Path):
    project = tmp_path / "Acane"
    dataset = project / "dataset.list"
    project.mkdir()
    dataset.write_text("clip.wav|Acane|ja|test\n", encoding="utf-8")
    return project, dataset


def test_descriptor_form_maps_fields_and_preserves_canonical_stages(tmp_path):
    from PySide6.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QLineEdit, QSpinBox
    from tts_builder.gui.training_form import TrainingForm

    create_application([])
    binding = _binding(tmp_path)
    form = TrainingForm((binding,), Translator("en"))
    project, dataset = _project(tmp_path)
    form.prefill_dataset(dataset)
    reference = project / "reference.wav"
    reference.write_bytes(b"wav")
    form.reference_audio.setText(str(reference))
    form.advanced_fields["resource"].setText(str(binding[0].project_root))

    assert isinstance(form.advanced_fields["steps"], QSpinBox)
    assert isinstance(form.advanced_fields["rate"], QDoubleSpinBox)
    assert isinstance(form.advanced_fields["resume"], QCheckBox)
    assert isinstance(form.advanced_fields["mode"], QComboBox)
    assert isinstance(form.advanced_fields["resource"], QLineEdit)
    assert isinstance(form.advanced_fields["note"], QLineEdit)
    assert form.advanced_fields["steps"].minimum() == 1
    assert form.advanced_fields["steps"].maximum() == 20
    assert form.selected_stages() == ("preprocess", "s2", "s1", "evaluate")
    assert form.is_valid()

    form.stage_boxes["s2"].setChecked(False)
    form.stage_boxes["preprocess"].setChecked(False)
    form.stage_boxes["preprocess"].setChecked(True)
    assert form.selected_stages() == ("preprocess", "s1", "evaluate")
    form.advanced_fields["resource"].setText("relative/path")
    assert not form.is_valid()


def test_prefill_uses_external_dataset_and_bound_module_output(tmp_path):
    from tts_builder.gui.training_form import TrainingForm

    create_application([])
    binding = _binding(tmp_path)
    form = TrainingForm((binding,), Translator("zh_CN"))
    project, dataset = _project(tmp_path)

    form.prefill_dataset(dataset)

    assert Path(form.dataset_edit.text()) == dataset.resolve()
    assert not hasattr(form, "project_edit")
    assert Path(form.output_edit.text()) == (binding[0].project_root / "runs").resolve()
    assert form.project_name.text() == "Acane"
    assert form.is_valid() is False  # evaluation still needs a reference audio


@pytest.mark.parametrize(
    ("locale", "label"),
    [("zh_CN", "训练数据"), ("en", "Training data"), ("ja", "トレーニングデータ")],
)
def test_training_data_label_is_localized(tmp_path, locale, label):
    from tts_builder.gui.training_form import TrainingForm

    create_application([])
    form = TrainingForm((_binding(tmp_path),), Translator(locale))

    assert form.dataset_label.text() == label


def test_training_data_browse_and_validation_follow_descriptor(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog
    from tts_builder.gui.training_form import TrainingForm

    create_application([])
    file_binding = _binding(tmp_path)
    directory = tmp_path / "directory-data"
    directory.mkdir()
    file_dataset = _project(tmp_path)[1]
    directory_framework = FrameworkDescriptor(
        "directory-framework",
        "Directory Framework",
        ("preprocess",),
        TrainingDataDescriptor("directory", ()),
        (),
    )
    directory_module = ModuleDescriptor(
        2, "directory-module", "1.0", (directory_framework,)
    )
    calls = []
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        lambda *args: (calls.append("file") or str(file_dataset), ""),
    )
    monkeypatch.setattr(
        QFileDialog,
        "getExistingDirectory",
        lambda *args: calls.append("directory") or str(directory),
    )
    form = TrainingForm(
        (file_binding, (file_binding[0], directory_module)), Translator("en")
    )

    form._browse_dataset()
    assert calls == ["file"]
    assert form.training_data_path().suffix == ".list"

    form.framework_combo.setCurrentIndex(1)
    form._browse_dataset()
    assert calls == ["file", "directory"]
    assert form.training_data_path() == directory.resolve()


def test_advanced_settings_start_closed_and_toggle_without_bottom_border(tmp_path):
    from tts_builder.gui.styles import APP_QSS
    from tts_builder.gui.training_form import TrainingForm

    create_application([])
    binding = _binding(tmp_path)
    form = TrainingForm((binding,), Translator("en"))

    assert form.advanced_toggle.isCheckable()
    assert not form.advanced_toggle.isChecked()
    assert form.advanced_content.isHidden()
    form._framework_changed()
    assert form.advanced_content.isHidden()

    form.advanced_toggle.click()
    assert not form.advanced_content.isHidden()
    assert form.advanced_content.objectName() == "AdvancedContent"
    block = APP_QSS.split("QToolButton#AdvancedToggle", 1)[1].split("}", 1)[0]
    assert "border-bottom" not in block
    advanced = APP_QSS.split("QWidget#AdvancedContent", 1)[1].split("}", 1)[0]
    assert "border-radius" in advanced
    assert "background" in advanced


def test_short_training_fields_share_rows(tmp_path):
    from tts_builder.gui.training_form import TrainingForm

    create_application([])
    form = TrainingForm((_binding(tmp_path),), Translator("en"))

    assert form.common_layout.indexOf(form.framework_combo) >= 0
    assert form.common_layout.indexOf(form.project_name) >= 0
    assert form.common_layout.indexOf(form.device) >= 0
    assert form.common_layout.indexOf(form.precision) >= 0
    assert form.common_layout.getItemPosition(
        form.common_layout.indexOf(form.device)
    )[0] == form.common_layout.getItemPosition(
        form.common_layout.indexOf(form.precision)
    )[0]


def test_advanced_fields_are_grouped_general_s1_s2_in_parallel(tmp_path):
    from tts_builder.gui.training_form import TrainingForm

    create_application([])
    setting, module = _binding(tmp_path)
    labels = {"en": "Value", "zh_CN": "数值", "ja": "値"}
    framework = FrameworkDescriptor(
        "grouped", "Grouped", module.frameworks[0].capabilities,
        module.frameworks[0].training_data,
        (
            FieldDescriptor("preprocess.resume", "boolean", True, {}, labels),
            FieldDescriptor("s1.batch_size", "integer", 2, {"minimum": 1}, labels),
            FieldDescriptor("s2.batch_size", "integer", 2, {"minimum": 1}, labels),
        ),
    )
    grouped_module = ModuleDescriptor(2, "grouped", "1.0", (framework,))
    form = TrainingForm(((setting, grouped_module),), Translator("zh_CN"))

    assert tuple(form.advanced_groups) == ("general", "s1", "s2")
    assert [form.advanced_group_titles[key].text() for key in form.advanced_groups] == [
        "通用", "S1", "S2",
    ]
    positions = [
        form.advanced_layout.getItemPosition(
            form.advanced_layout.indexOf(form.advanced_groups[key])
        )
        for key in form.advanced_groups
    ]
    assert [position[0] for position in positions] == [0, 0, 0]
    assert [position[1] for position in positions] == [0, 1, 2]


def test_advanced_controls_use_rounded_dark_input_style():
    from tts_builder.gui.styles import APP_QSS

    inputs = APP_QSS.split("QLineEdit, QComboBox", 1)[1].split("}", 1)[0]
    assert "QSpinBox" in inputs
    assert "QDoubleSpinBox" in inputs
    group = APP_QSS.split("QFrame#AdvancedGroup", 1)[1].split("}", 1)[0]
    assert "border: none" in group
    assert "border-radius" in group
    assert "background" in group


def test_combo_and_spin_wheels_scroll_page_without_changing_values(tmp_path):
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QWheelEvent
    from PySide6.QtWidgets import QApplication, QScrollArea
    from tts_builder.gui.training_form import TrainingForm

    app = create_application([])
    form = TrainingForm((_binding(tmp_path),), Translator("en"))
    form.advanced_toggle.click()
    form.setMinimumHeight(1000)
    area = QScrollArea()
    area.resize(520, 260)
    area.setWidget(form)
    area.show()
    app.processEvents()
    bar = area.verticalScrollBar()
    bar.setValue(100)
    before_scroll = bar.value()
    widgets = [
        form.device,
        form.precision,
        form.reference_language,
        form.advanced_fields["steps"],
        form.advanced_fields["rate"],
        form.advanced_fields["mode"],
    ]
    values = [
        widget.currentIndex() if hasattr(widget, "currentIndex") else widget.value()
        for widget in widgets
    ]
    event = QWheelEvent(
        QPointF(5, 5),
        QPointF(5, 5),
        QPoint(),
        QPoint(0, -120),
        Qt.NoButton,
        Qt.NoModifier,
        Qt.ScrollUpdate,
        False,
    )

    QApplication.sendEvent(form.device, event)
    for widget in widgets[1:]:
        QApplication.sendEvent(
            widget,
            QWheelEvent(
                QPointF(5, 5), QPointF(5, 5), QPoint(), QPoint(0, -120),
                Qt.NoButton, Qt.NoModifier, Qt.ScrollUpdate, False,
            ),
        )

    assert [
        widget.currentIndex() if hasattr(widget, "currentIndex") else widget.value()
        for widget in widgets
    ] == values
    assert bar.value() > before_scroll
    area.close()


def test_snapshot_is_a_frozen_start_time_selection(tmp_path):
    from tts_builder.gui.training_form import TrainingForm

    create_application([])
    binding = _binding(tmp_path)
    form = TrainingForm((binding,), Translator("en"))
    project, dataset = _project(tmp_path)
    form.prefill_dataset(dataset)
    reference = project / "reference.wav"
    reference.write_bytes(b"wav")
    form.reference_audio.setText(str(reference))
    form.advanced_fields["resource"].setText(str(binding[0].project_root))

    snapshot = form.snapshot()

    assert snapshot.project_dir == binding[0].project_root.resolve()
    assert snapshot.training_data == dataset.resolve()
    assert snapshot.values["project_name"] == "Acane"
    assert snapshot.values["parameters"]["steps"] == 10
    with pytest.raises(FrozenInstanceError):
        snapshot.project_dir = tmp_path
