import os
import threading
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QScrollArea

from tts_builder.gui import app as gui_app
from tts_builder.gui.app import create_application
from tts_builder.gui.main_window import MainWindow
from tts_builder.gui.settings import AppSettings, TrainingModuleSetting
from tts_builder.training_modules.models import (
    FrameworkDescriptor,
    ModuleDescriptor,
    ProbeResult,
    TrainingDataDescriptor,
)


class RecordingController(QObject):
    event_received = Signal(object)
    task_completed = Signal(object)
    task_failed = Signal(str, str, str)
    task_cancelled = Signal()
    running_changed = Signal(bool)

    def __init__(self):
        super().__init__()
        self.running = False

    def start(self, *_):
        self.running = True

    def stop(self):
        pass


def _descriptor():
    framework = FrameworkDescriptor(
        id="v2ProPlus",
        display_name="GPT-SoVITS v2ProPlus",
        capabilities=("train",),
        training_data=TrainingDataDescriptor("file", (".list",)),
        fields=(),
    )
    return ModuleDescriptor(2, "gpt-sovits-v2proplus", "1.0.0", (framework,))


def _settings(tmp_path, modules=()):
    return AppSettings(
        first_run_completed=True,
        model_root=tmp_path / "models",
        output_root=tmp_path / "out",
        training_modules=modules,
    )


def test_empty_configuration_skips_probe_and_keeps_standalone_window(tmp_path):
    app = create_application([])
    settings = _settings(tmp_path)

    modules = gui_app.discover_available_modules(
        settings,
        probe=lambda *_: (_ for _ in ()).throw(AssertionError("probe must not run")),
    )
    window = gui_app.choose_window(settings, modules)

    assert modules == ()
    assert type(window) is MainWindow
    assert window.windowTitle() == "Voice Dataset Builder"
    app.processEvents()
    window.close()


def test_explicit_module_probe_runs_off_the_gui_thread(tmp_path):
    project = tmp_path / "voice-pipeline"
    python = tmp_path / "venv" / "python.exe"
    project.mkdir()
    python.parent.mkdir()
    python.write_bytes(b"")
    setting = TrainingModuleSetting("GPT-SoVITS", project, python, "voice_pipeline")
    caller_thread = threading.get_ident()
    probe_threads = []

    def probe(_setting):
        probe_threads.append(threading.get_ident())
        return ProbeResult(True, descriptor=_descriptor())

    modules = gui_app.discover_available_modules(_settings(tmp_path, (setting,)), probe=probe)

    assert modules == (_descriptor(),)
    assert probe_threads and probe_threads[0] != caller_thread


def test_available_module_binding_keeps_its_own_environment(tmp_path):
    first_project = tmp_path / "broken"
    second_project = tmp_path / "working"
    first_python = tmp_path / "venv1" / "python.exe"
    second_python = tmp_path / "venv2" / "python.exe"
    first_project.mkdir()
    second_project.mkdir()
    first_python.parent.mkdir()
    second_python.parent.mkdir()
    first_python.write_bytes(b"")
    second_python.write_bytes(b"")
    first = TrainingModuleSetting("Broken", first_project, first_python, "broken")
    second = TrainingModuleSetting("Working", second_project, second_python, "working")

    bindings = gui_app.discover_available_module_bindings(
        _settings(tmp_path, (first, second)),
        probe=lambda setting: (
            ProbeResult(False, error="offline")
            if setting is first
            else ProbeResult(True, descriptor=_descriptor())
        ),
    )

    assert bindings == ((second, _descriptor()),)


def test_available_module_uses_studio_with_two_localized_business_buttons(tmp_path):
    app = create_application([])
    window = gui_app.choose_window(_settings(tmp_path), (_descriptor(),))

    from tts_builder.gui.studio_window import StudioWindow

    assert type(window) is StudioWindow
    assert window.windowTitle() == "AudioClone Studio"
    assert window.navigation.workspace_label.text() == "Workspace"
    assert [button.text() for button in window.navigation.buttons] == [
        "Dataset Mining", "Training"
    ]
    assert window.pages.count() == 2
    window.locale_controller.set_locale("zh_CN")
    app.processEvents()
    assert window.navigation.workspace_label.text() == "工作区"
    assert [button.text() for button in window.navigation.buttons] == [
        "素材挖掘", "训练"
    ]
    window.close()


def test_selection_follows_clicks_and_ignores_dataset_completion(tmp_path):
    from tts_builder.gui.studio_window import StudioWindow

    app = create_application([])
    controller = RecordingController()
    window = StudioWindow(_settings(tmp_path), (_descriptor(),), controller)

    window.navigation.buttons[1].click()
    app.processEvents()
    assert window.pages.currentIndex() == 1
    assert [button.property("selected") for button in window.navigation.buttons] == [
        False, True
    ]

    controller.task_completed.emit(SimpleNamespace(accepted=3, rejected=0))
    app.processEvents()
    assert window.pages.currentIndex() == 1
    assert [button.property("selected") for button in window.navigation.buttons] == [
        False, True
    ]
    window.close()


def test_studio_fits_screen_scrolls_pages_and_collapses_narrow_navigation(tmp_path):
    from tts_builder.gui.studio_window import StudioWindow

    app = create_application([])
    window = StudioWindow(_settings(tmp_path), (_descriptor(),))
    available = app.primaryScreen().availableGeometry()

    assert window.width() <= available.width()
    assert window.height() <= available.height()
    assert all(area.widgetResizable() for area in window.findChildren(QScrollArea))

    window.show()
    app.processEvents()
    window.resize(700, min(650, available.height()))
    app.processEvents()
    assert window.navigation.compact is True

    window.resize(min(1100, available.width()), min(700, available.height()))
    app.processEvents()
    if available.width() >= 900:
        assert window.navigation.compact is False
    window.close()


def test_standalone_window_also_fits_small_screens_and_scrolls(tmp_path):
    app = create_application([])
    window = MainWindow(_settings(tmp_path))
    available = app.primaryScreen().availableGeometry()

    assert window.width() <= available.width()
    assert window.height() <= available.height()
    scroll_areas = window.findChildren(QScrollArea)
    assert len(scroll_areas) == 1
    assert scroll_areas[0].widgetResizable() is True
    window.close()
