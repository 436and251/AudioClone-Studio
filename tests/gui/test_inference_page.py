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

    model = tmp_path / "model"
    model.mkdir()
    page.set_model(model)
    assert page.start_button.isEnabled()

    text_file = tmp_path / "input.txt"
    text_file.write_text("file", encoding="utf-8")
    page.txt_edit.setText(str(text_file))
    assert not page.start_button.isEnabled()
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
