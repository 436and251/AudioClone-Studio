import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest
pytest.importorskip('PySide6')

from dataclasses import replace
from types import SimpleNamespace

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QDialog

from tts_builder.gui.app import create_application
from tts_builder.gui import app as gui_app
from tts_builder.gui.main_window import MainWindow
from tts_builder.gui.settings import AppSettings


class RecordingController(QObject):
    event_received = Signal(object)
    task_completed = Signal(object)
    task_failed = Signal(str, str, str)
    task_cancelled = Signal()
    running_changed = Signal(bool)

    def __init__(self):
        super().__init__()
        self.running = False
        self.stop_calls = 0

    def stop(self):
        self.stop_calls += 1


def test_main_window_enables_start_only_with_source_and_speaker(tmp_path):
    app = create_application([])
    window = MainWindow(AppSettings(first_run_completed=True, model_root=tmp_path/'models', output_root=tmp_path/'out'))
    assert not window.start.isEnabled()
    window.source.set_value(str(tmp_path/'input.wav'))
    window.speaker.setText('suis')
    app.processEvents()
    assert window.start.isEnabled()
    window.close()


def test_main_window_preserves_original_structure_defaults_and_size(tmp_path):
    app = create_application([])
    window = MainWindow(AppSettings(
        first_run_completed=True,
        model_root=tmp_path / 'models',
        output_root=tmp_path / 'out',
    ))

    available = app.primaryScreen().availableGeometry()
    if available.width() >= 860 and available.height() >= 760:
        assert (window.width(), window.height()) == (860, 760)
    assert window.width() <= available.width()
    assert window.height() <= available.height()
    assert window.minimumWidth() <= available.width()
    assert window.minimumHeight() <= available.height()
    assert window.title_label.objectName() == 'Title'
    assert window.subtitle_label.objectName() == 'Subtitle'
    assert window.settings_btn.objectName() == 'IconButton'
    assert window.output_label.objectName() == 'FieldChip'
    assert window.stop.objectName() == 'Danger'
    assert window.start.objectName() == 'Primary'
    assert [window.language.itemData(i) for i in range(window.language.count())] == [
        'auto', 'ja', 'zh', 'en'
    ]
    assert window.model.currentText() == 'large-v3-turbo'
    app.processEvents()
    window.close()


def test_stop_action_and_controller_signals_keep_driving_the_page(tmp_path):
    app = create_application([])
    controller = RecordingController()
    window = MainWindow(
        AppSettings(model_root=tmp_path / 'models', output_root=tmp_path / 'out'),
        controller,
    )

    controller.running = True
    controller.running_changed.emit(True)
    app.processEvents()
    assert not window.source.isEnabled()
    assert window.stop.isEnabled()
    window.stop.click()
    assert controller.stop_calls == 1

    controller.task_completed.emit(SimpleNamespace(accepted=4, rejected=1))
    app.processEvents()
    assert window.status.text() == 'Ready · 4 accepted · 1 rejected'

    controller.task_cancelled.emit()
    app.processEvents()
    assert window.status.text() == 'Stopped · completed cache was preserved'
    window.close()


def test_output_actions_update_and_create_the_selected_directory(tmp_path, monkeypatch):
    app = create_application([])
    window = MainWindow(AppSettings(
        model_root=tmp_path / 'models', output_root=tmp_path / 'out'
    ))
    selected = tmp_path / 'selected'
    monkeypatch.setattr(
        'PySide6.QtWidgets.QFileDialog.getExistingDirectory',
        lambda *_: str(selected),
    )
    monkeypatch.setattr(
        'PySide6.QtGui.QDesktopServices.openUrl', lambda *_: True
    )

    window.output_browse.click()
    assert window.output.text() == str(selected)
    window.open_output.click()
    assert selected.is_dir()
    app.processEvents()
    window.close()


def test_settings_action_applies_accepted_values_to_the_live_window(tmp_path, monkeypatch):
    app = create_application([])
    settings = AppSettings(
        model_root=tmp_path / 'models', output_root=tmp_path / 'out'
    )
    changed = replace(
        settings,
        model_root=tmp_path / 'other-models',
        output_root=tmp_path / 'other-output',
        preferred_asr_model='small',
        locale='zh_CN',
    )
    monkeypatch.setattr('tts_builder.gui.settings_dialog.SettingsDialog.exec', lambda *_: QDialog.Accepted)
    monkeypatch.setattr('tts_builder.gui.settings_dialog.SettingsDialog.result_settings', lambda *_: changed)
    monkeypatch.setattr(AppSettings, 'save', lambda *_: None)
    window = MainWindow(settings)

    window.settings_btn.click()
    app.processEvents()

    assert window.output.text() == str(changed.output_root)
    assert window.model.currentText() == 'small'
    assert window.windowTitle() == '语音数据集构建器'
    window.close()


def test_source_gui_uses_its_window_icon_without_packaged_shell_identity():
    app = create_application([])

    assert not hasattr(gui_app, "set_windows_app_id")
    assert app.windowIcon().isNull() is False
