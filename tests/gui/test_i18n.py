import os
from string import Formatter

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from tts_builder.gui.app import create_application
from tts_builder.gui.i18n import CATALOGS, LocaleController, Translator
from tts_builder.gui.main_window import MainWindow
from tts_builder.gui.settings import AppSettings
from tts_builder.gui.settings_dialog import SettingsDialog


def _fields(value: str) -> set[str]:
    return {
        name
        for _, name, _, _ in Formatter().parse(value)
        if name is not None
    }


def test_catalogs_have_identical_keys_and_placeholders():
    assert set(CATALOGS) == {"en", "zh_CN", "ja"}
    english = CATALOGS["en"]
    for catalog in CATALOGS.values():
        assert set(catalog) == set(english)
        for key, value in catalog.items():
            assert _fields(value) == _fields(english[key])


def test_translator_rejects_unknown_keys_and_preserves_format_values():
    translator = Translator("zh_CN")
    assert "large-v3-turbo" in translator.text(
        "models.not_installed", model="large-v3-turbo"
    )
    with pytest.raises(KeyError):
        translator.text("missing.translation.key")


def test_locale_controller_accepts_exactly_three_locales():
    controller = LocaleController("en")
    changes = []
    controller.locale_changed.connect(changes.append)

    controller.set_locale("zh_CN")
    controller.set_locale("ja")

    assert changes == ["zh_CN", "ja"]
    with pytest.raises(ValueError):
        controller.set_locale("zh")


def test_settings_dialog_switches_its_labels_and_returns_locale(tmp_path):
    app = create_application([])
    dialog = SettingsDialog(AppSettings(
        model_root=tmp_path / "models", output_root=tmp_path / "out"
    ))

    dialog.locale.setCurrentIndex(dialog.locale.findData("zh_CN"))
    app.processEvents()

    assert dialog.windowTitle() == "设置"
    assert dialog.model_root_label.text() == "模型存储目录"
    assert dialog.result_settings().locale == "zh_CN"
    dialog.close()


def test_main_window_switches_language_without_replacing_runtime_state(tmp_path):
    app = create_application([])
    window = MainWindow(
        AppSettings(
            first_run_completed=True,
            model_root=tmp_path / "models",
            output_root=tmp_path / "out",
            locale="en",
        )
    )
    task_controller = window.controller
    progress_model = window.progress_model
    window.source.set_value("https://example.invalid/source")
    window.speaker.setText("acane")
    window._set_status("status.preparing")

    window.locale_controller.set_locale("zh_CN")
    app.processEvents()
    assert window.windowTitle() == "语音数据集构建器"
    assert window.start.text() == "▶  构建数据集"
    assert window.progress.names["prepare"].text() == "准备"
    assert window.settings_btn.toolTip() == "设置"
    assert window.status.text() == "正在准备…"

    window.locale_controller.set_locale("ja")
    app.processEvents()
    assert window.windowTitle() == "音声データセット作成ツール"
    assert window.start.text() == "▶  データセットを作成"
    assert window.source.edit.placeholderText().startswith("メディア URL")
    assert window.status.text() == "準備中…"
    assert window.controller is task_controller
    assert window.progress_model is progress_model
    assert window.source.value() == "https://example.invalid/source"
    assert window.speaker.text() == "acane"

    window.close()
