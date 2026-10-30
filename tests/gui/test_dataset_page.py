import os
from types import SimpleNamespace

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest

pytest.importorskip('PySide6')

from PySide6.QtWidgets import QScrollArea, QTabWidget

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

    assert isinstance(window.centralWidget(), QTabWidget)
    assert isinstance(window.tabs.widget(0), QScrollArea)
    assert window.tabs.widget(0).widget() is window.dataset_page
    assert window.findChildren(DatasetPage) == [window.dataset_page]
    app.processEvents()
    window.close()


def test_changed_source_clears_completed_status_without_removing_dataset(tmp_path):
    from tts_builder.gui.dataset_page import DatasetPage
    from tts_builder.gui.controller import TaskController

    create_application([])
    dataset = tmp_path / "dataset.list"
    dataset.write_text("retained clips", encoding="utf-8")
    page = DatasetPage(AppSettings(output_root=tmp_path), TaskController(), training_available=True)
    page.source.set_value("first.wav")
    page._completed(SimpleNamespace(accepted=2, rejected=0))
    assert not page.continue_training.isHidden()
    page.source.set_value("second.wav")
    assert page.continue_training.isHidden()
    assert "2 accepted" not in page.status.text()
    assert dataset.read_text(encoding="utf-8") == "retained clips"
    page.close()
