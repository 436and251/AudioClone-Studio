import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from tts_builder.gui.app import create_application
from tts_builder.gui.i18n import LocaleController


class FakePlayer:
    def __init__(self):
        self.played = []

    def play(self, path):
        self.played.append(Path(path))

    def stop(self):
        pass


def test_inference_page_stays_editable_but_locked_without_promoted_model(tmp_path):
    from tts_builder.gui.inference_page import InferencePage

    create_application([])
    page = InferencePage(LocaleController("en"), player=FakePlayer())

    page.text.setPlainText("hello")

    assert page.text.isEnabled()
    assert page.txt_edit.isEnabled()
    assert [page.language.itemData(index) for index in range(page.language.count())] == [
        "zh", "ja", "en", "mixed"
    ]
    assert not page.start_button.isEnabled()
    assert page.model_hint.text() == "No promoted model has been selected yet."
    assert page.title.text() == "Select a model for inference"

    model = tmp_path / "model"
    model.mkdir()
    page.set_models((model,))
    assert page.start_button.isEnabled()
    assert page.title.text() == "Generate with the selected model"
    assert page.history.count() == 1
    assert page.history.item(0).text() == "model"

    text_file = tmp_path / "input.txt"
    text_file.write_text("file", encoding="utf-8")
    page.txt_edit.setText(str(text_file))
    assert page.start_button.isEnabled()
    page.close()


def test_inference_page_explains_conflicting_text_sources(tmp_path):
    from tts_builder.gui.inference_page import InferencePage

    create_application([])
    page = InferencePage(LocaleController("en"), player=FakePlayer())
    model = tmp_path / "model"
    model.mkdir()
    source = tmp_path / "input.txt"
    source.write_text("from file", encoding="utf-8")
    requested = []
    page.inference_requested.connect(requested.append)
    page.set_models((model,))
    page.text.setPlainText("inline")
    page.txt_edit.setText(str(source))

    assert page.start_button.isEnabled()
    page.start_button.click()

    assert requested == []
    assert page.error.text() == "Use either text or a TXT file, not both."
    assert page.error.isVisibleTo(page)
    page.close()


def test_inference_page_explains_missing_text_for_selected_model(tmp_path):
    from tts_builder.gui.inference_page import InferencePage

    create_application([])
    page = InferencePage(LocaleController("en"), player=FakePlayer())
    model = tmp_path / "model"
    model.mkdir()
    page.set_models((model,))

    assert page.start_button.isEnabled()
    page.start_button.click()

    assert page.error.text() == "Enter text or choose a TXT file."
    assert page.error.isVisibleTo(page)
    page.close()


def test_inference_history_selects_models_and_restores_each_result(tmp_path):
    from tts_builder.gui.inference_page import InferencePage

    create_application([])
    page = InferencePage(LocaleController("en"), player=FakePlayer())
    first = tmp_path / "Acane"
    second = tmp_path / "Lucy"
    first.mkdir()
    second.mkdir()
    first_audio = tmp_path / "first.wav"
    first_audio.write_bytes(b"wav")
    selected = []
    page.model_selected.connect(selected.append)

    page.set_models((second, first), results={first.resolve(): first_audio.resolve()})
    page.history.setCurrentRow(1)

    assert selected[-1] == first.resolve()
    assert page.model == first.resolve()
    assert page.result == first_audio.resolve()
    assert page.play_button.isEnabled()
    page.close()


def test_inference_button_immediately_shows_generating_state(tmp_path):
    from tts_builder.gui.inference_page import InferencePage

    app = create_application([])
    page = InferencePage(LocaleController("en"))
    model = tmp_path / "model"
    model.mkdir()
    page.set_model(model)
    page.text.setPlainText("Hello")
    page.inference_requested.connect(lambda _request: page.set_busy(True))

    page.start_button.click()
    app.processEvents()

    assert not page.start_button.isEnabled()
    assert page.start_button.text() == "Generating speech…"
    assert page.busy_hint.isVisibleTo(page)
    page.close()


def test_inference_page_emits_input_plays_result_opens_directory_and_hides_toast(
    tmp_path, monkeypatch
):
    from PySide6.QtGui import QDesktopServices
    from tts_builder.gui.inference_page import InferencePage

    app = create_application([])
    player = FakePlayer()
    page = InferencePage(LocaleController("zh_CN"), player=player)
    model = tmp_path / "model"
    model.mkdir()
    page.set_model(model)
    page.text.setPlainText("你好")
    page.language.setCurrentIndex(page.language.findData("zh"))
    requested = []
    page.inference_requested.connect(requested.append)

    page.start_button.click()

    assert len(requested) == 1
    assert requested[0].text == "你好"
    assert requested[0].text_file is None
    assert requested[0].language == "zh"

    output = tmp_path / "outputs" / "Acane" / "gui" / "result.wav"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"wav")
    opened = []
    monkeypatch.setattr(QDesktopServices, "openUrl", lambda url: opened.append(url) or True)
    page.set_result(output)
    app.processEvents()

    page.play_button.click()
    page.open_button.click()
    assert player.played == [output.resolve()]
    assert opened and Path(opened[0].toLocalFile()).resolve() == output.parent.resolve()
    assert page.toast.isVisibleTo(page)
    assert str(output.parent.resolve()) in page.toast.text()
    page.toast_timer.timeout.emit()
    assert page.toast.isHidden()
    page.close()
