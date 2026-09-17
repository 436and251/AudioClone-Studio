import os

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest

pytest.importorskip('PySide6')

from tts_builder.gui.app import create_application
from tts_builder.gui.main_window import MainWindow
from tts_builder.gui.settings import AppSettings


def test_main_window_hosts_exactly_one_dataset_page(tmp_path):
    from tts_builder.gui.dataset_page import DatasetPage

    app = create_application([])
    window = MainWindow(AppSettings(
        first_run_completed=True,
        model_root=tmp_path / 'models',
        output_root=tmp_path / 'out',
    ))

    assert isinstance(window.centralWidget(), DatasetPage)
    assert window.findChildren(DatasetPage) == [window.centralWidget()]
    app.processEvents()
    window.close()
