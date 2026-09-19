import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QUrl
from PySide6.QtMultimedia import QMediaPlayer

from tts_builder.gui.app import create_application
from tts_builder.gui.audio_player import AudioPlayer


def test_finished_audio_releases_its_source_file(tmp_path):
    create_application([])
    wav = tmp_path / "result.wav"
    wav.write_bytes(b"wav")
    player = AudioPlayer()
    player.player.setSource(QUrl.fromLocalFile(str(wav)))

    player._media_status_changed(QMediaPlayer.EndOfMedia)

    assert player.player.source().isEmpty()
