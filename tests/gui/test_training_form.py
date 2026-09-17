import os
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
        ("preprocess", "train", "evaluate", "listen", "promote"), fields,
    )
    return setting, ModuleDescriptor(1, "gpt-sovits", "1.0", (framework,))


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
    form = TrainingForm((_binding(tmp_path),), Translator("en"))
    project, dataset = _project(tmp_path)
    form.prefill_dataset(dataset)
    reference = project / "reference.wav"
    reference.write_bytes(b"wav")
    form.reference_audio.setText(str(reference))
    form.advanced_fields["resource"].setText(str(project))

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


def test_prefill_uses_existing_dataset_without_copying_it(tmp_path):
    from tts_builder.gui.training_form import TrainingForm

    create_application([])
    form = TrainingForm((_binding(tmp_path),), Translator("zh_CN"))
    project, dataset = _project(tmp_path)

    form.prefill_dataset(dataset)

    assert Path(form.dataset_edit.text()) == dataset.resolve()
    assert Path(form.project_edit.text()) == project.resolve()
    assert Path(form.output_edit.text()) == (project / "runs").resolve()
    assert form.project_name.text() == "Acane"
