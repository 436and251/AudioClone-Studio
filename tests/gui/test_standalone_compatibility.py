import os

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest

pytest.importorskip('PySide6')

from PySide6.QtWidgets import QStackedWidget

from tts_builder.gui.app import create_application
from tts_builder.gui.main_window import MainWindow
from tts_builder.gui.settings import AppSettings


def test_empty_module_configuration_keeps_original_single_page_window(tmp_path):
    app = create_application([])
    window = MainWindow(
        AppSettings(
            first_run_completed=True,
            model_root=tmp_path / 'models',
            output_root=tmp_path / 'out',
            training_modules=(),
        )
    )

    assert window.windowTitle() == 'AudioMiner'
    assert window.findChild(QStackedWidget) is None
    assert window.start.isEnabled()
    window.start.click()
    assert window.status.text() == "Choose a media source first."
    window.source.set_value(str(tmp_path / 'input.wav'))
    window.speaker.setText('suis')
    app.processEvents()
    assert window.start.isEnabled()

    window.close()
