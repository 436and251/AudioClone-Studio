import os
import subprocess
import sys
from pathlib import Path
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest
pytest.importorskip('PySide6')

from dataclasses import replace
from types import SimpleNamespace

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QDialog

from tts_builder.gui.app import create_application
from tts_builder.gui.main_window import MainWindow
from tts_builder.gui.settings import AppSettings
from tts_builder.events import PipelineEvent


def test_settings_combos_ignore_mouse_wheel(tmp_path):
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QWheelEvent
    from PySide6.QtWidgets import QApplication
    from tts_builder.gui.settings_dialog import SettingsDialog

    create_application([])
    dialog = SettingsDialog(
        AppSettings(model_root=tmp_path / "models", output_root=tmp_path / "out")
    )
    widgets = (dialog.model, dialog.locale)
    before = [widget.currentIndex() for widget in widgets]
    for widget in widgets:
        QApplication.sendEvent(
            widget,
            QWheelEvent(
                QPointF(5, 5), QPointF(5, 5), QPoint(), QPoint(0, -120),
                Qt.NoButton, Qt.NoModifier, Qt.ScrollUpdate, False,
            ),
        )

    assert [widget.currentIndex() for widget in widgets] == before
    dialog.close()


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


def test_main_window_start_explains_each_missing_required_field(tmp_path):
    app = create_application([])
    window = MainWindow(AppSettings(first_run_completed=True, model_root=tmp_path/'models', output_root=tmp_path/'out'))
    assert window.start.isEnabled()
    window.start.click()
    assert window.status.text() == "Choose a media source first."

    window.source.set_value(str(tmp_path/'input.wav'))
    window.start.click()
    assert window.status.text() == "Enter the target speaker name."

    window.speaker.setText('suis')
    window.output.clear()
    app.processEvents()
    window.start.click()
    assert window.status.text() == "Choose an output directory."

    window.output.setText(str(tmp_path / "out"))
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
    assert window.settings_btn.objectName() == 'SettingsButton'
    assert window.settings_btn.width() == 32
    assert window.settings_btn.height() == 32
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


def test_dataset_long_messages_stay_in_activity_not_fixed_layout(tmp_path, monkeypatch):
    app = create_application([])
    controller = RecordingController()
    window = MainWindow(
        AppSettings(model_root=tmp_path / "models", output_root=tmp_path / "out"),
        controller,
    )
    dialogue = "Very long ASR dialogue " * 30
    failure = "Very long decoder failure " * 30

    controller.event_received.emit(
        PipelineEvent("stage_progress", "asr", dialogue, 1, 10)
    )
    app.processEvents()

    assert window.progress.rows["asr"][1].text() == "10%"
    assert dialogue not in window.status.text()
    assert dialogue in window.logs.view.toPlainText()

    controller.event_received.emit(PipelineEvent("stage_failed", "asr", failure))
    app.processEvents()
    assert window.progress.rows["asr"][1].text() == "Failed"
    assert failure not in window.status.text()
    assert failure in window.logs.view.toPlainText()

    monkeypatch.setattr("PySide6.QtWidgets.QMessageBox.exec", lambda *_: 0)
    controller.task_failed.emit("ASR failed", failure, "decoder traceback")
    app.processEvents()
    assert window.status.text() == "Failed · see Activity for details"
    assert "decoder traceback" in window.logs.view.toPlainText()
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


def test_open_output_explains_missing_directory(tmp_path):
    create_application([])
    window = MainWindow(AppSettings(
        model_root=tmp_path / "models", output_root=tmp_path / "out"
    ))
    window.output.clear()

    window.open_output.click()

    assert window.status.text() == "Choose an output directory."
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


@pytest.mark.skipif(sys.platform != "win32", reason="Windows taskbar identity and native icon")
@pytest.mark.parametrize("studio", [False, True])
def test_windows_taskbar_identity_and_native_window_icon(tmp_path, studio):
    probe = """
from ctypes import byref, cast, c_uint, c_wchar_p, c_ssize_t, c_void_p, windll
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QImage, QPainter
from tts_builder.gui.app import create_application
from tts_builder.gui.main_window import MainWindow
from tts_builder.gui.studio_window import StudioWindow
from tts_builder.gui.settings import AppSettings
from tts_builder.training_modules.models import FrameworkDescriptor, ModuleDescriptor, TrainingDataDescriptor
from pathlib import Path
import sys

app = create_application([])
app_id = c_wchar_p()
result = windll.shell32.GetCurrentProcessExplicitAppUserModelID(byref(app_id))
assert result == 0 and app_id.value == 'AudioMiner.Desktop', (result, app_id.value)
windll.ole32.CoTaskMemFree(c_void_p(cast(app_id, c_void_p).value))
root = Path(sys.argv[2])
settings = AppSettings(output_root=root/'voices', audio2score_output_root=root/'scores', model_root=root/'models')
framework = FrameworkDescriptor('v2ProPlus', 'GPT-SoVITS v2ProPlus', ('train',), TrainingDataDescriptor('file', ('.list',)), ())
module = ModuleDescriptor(2, 'gpt-sovits', '1.0', (framework,))
window = StudioWindow(settings, (module,)) if sys.argv[1] == 'True' else MainWindow(settings)
window.setAttribute(Qt.WA_DontShowOnScreen)
window.show()
app.processEvents()
send = windll.user32.SendMessageW
send.argtypes = [c_void_p, c_uint, c_void_p, c_void_p]
send.restype = c_ssize_t
for kind in (0, 1):
    handle = send(int(window.winId()), 0x7F, kind, 0)
    assert handle, 'Native WM_GETICON missing'
    actual = QImage.fromHICON(handle)
    expected = QIcon(sys.argv[3]).pixmap(actual.size(), 1.0).toImage()
    def on_black(source):
        target = QImage(source.size(), QImage.Format_RGB32)
        target.fill(Qt.black)
        painter = QPainter(target)
        painter.drawImage(0, 0, source)
        painter.end()
        return target
    actual, expected = on_black(actual), on_black(expected)
    assert actual.size() == expected.size()
    # HICON premultiplied-alpha conversion can round a channel by one level.
    assert all(abs(a-b) <= 1 for y in range(actual.height()) for x in range(actual.width())
               for a,b in zip(actual.pixelColor(x,y).getRgb(), expected.pixelColor(x,y).getRgb())), 'Native icon differs from application asset'
window.close()
print('taskbar identity and native icons verified')
"""

    env = {**os.environ, "QT_QPA_PLATFORM": "windows"}
    completed = subprocess.run(
        [sys.executable, "-c", probe, str(studio), str(tmp_path), str(Path(__file__).resolve().parents[2] / 'assets' / 'app.ico')],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert 'taskbar identity and native icons verified' in completed.stdout


def test_source_gui_uses_window_icon():
    app = create_application([])

    assert app.windowIcon().isNull() is False
