import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from tts_builder.gui.app import create_application
from tts_builder.gui.i18n import Translator
from tts_builder.gui.settings import AppSettings, TrainingModuleSetting
from tts_builder.gui.settings_dialog import SettingsDialog
from tts_builder.training_modules.models import (
    FrameworkDescriptor,
    ModuleDescriptor,
    ProbeResult,
)


def _paths(tmp_path: Path):
    project = tmp_path / "voice-pipeline"
    python = tmp_path / "venv" / "python.exe"
    project.mkdir()
    python.parent.mkdir()
    python.write_bytes(b"")
    return project, python


def _setting(tmp_path: Path):
    project, python = _paths(tmp_path)
    return TrainingModuleSetting("GPT-SoVITS", project, python, "voice_pipeline")


def test_settings_dialog_adds_edits_and_removes_explicit_module(tmp_path):
    app = create_application([])
    dialog = SettingsDialog(AppSettings(
        model_root=tmp_path / "models", output_root=tmp_path / "out"
    ))
    editor = dialog.module_settings

    editor.add_button.click()
    assert not dialog.save_button.isEnabled()
    project, python = _paths(tmp_path)
    editor.name_edit.setText("GPT-SoVITS")
    editor.project_edit.setText(str(project))
    editor.python_edit.setText(str(python))
    editor.module_edit.setText("voice_pipeline")
    app.processEvents()

    assert dialog.save_button.isEnabled()
    assert dialog.result_settings().training_modules == (
        TrainingModuleSetting("GPT-SoVITS", project, python, "voice_pipeline"),
    )
    editor.name_edit.setText("GPT-SoVITS Local")
    app.processEvents()
    assert editor.module_list.item(0).text() == "GPT-SoVITS Local"

    editor.remove_button.click()
    app.processEvents()
    assert dialog.save_button.isEnabled()
    assert dialog.result_settings().training_modules == ()
    dialog.close()


def test_module_editor_browses_project_and_python_separately(tmp_path, monkeypatch):
    from tts_builder.gui.module_settings import ModuleSettings

    app = create_application([])
    project, python = _paths(tmp_path)
    editor = ModuleSettings((), Translator("en"))
    editor.add_button.click()
    monkeypatch.setattr(
        "PySide6.QtWidgets.QFileDialog.getExistingDirectory", lambda *_: str(project)
    )
    monkeypatch.setattr(
        "PySide6.QtWidgets.QFileDialog.getOpenFileName", lambda *_: (str(python), "")
    )

    editor.project_browse.click()
    editor.python_browse.click()
    app.processEvents()

    assert editor.project_edit.text() == str(project)
    assert editor.python_edit.text() == str(python)
    editor.close()


def test_check_connection_performs_descriptor_probe_only(tmp_path):
    from tts_builder.gui.module_settings import ModuleSettings

    app = create_application([])
    setting = _setting(tmp_path)
    framework = FrameworkDescriptor("v2ProPlus", "GPT-SoVITS v2ProPlus", ("train",), ())
    descriptor = ModuleDescriptor(1, "gpt-sovits-v2proplus", "1.0.0", (framework,))
    calls = []

    def probe(value):
        calls.append(value)
        return ProbeResult(True, descriptor=descriptor)

    editor = ModuleSettings((setting,), Translator("en"), probe=probe)
    editor.check_button.click()
    app.processEvents()

    assert calls == [setting]
    assert editor.connection_status.text() == "Connected · GPT-SoVITS v2ProPlus"
    editor.close()
