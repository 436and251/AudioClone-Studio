from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, QUrl
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer


class AudioPlayer(QObject):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.output = QAudioOutput(self)
        self.player = QMediaPlayer(self)
        self.player.setAudioOutput(self.output)
        self.player.mediaStatusChanged.connect(self._media_status_changed)

    def play(self, path: Path) -> None:
        self.player.stop()
        self.player.setSource(QUrl.fromLocalFile(str(Path(path).resolve())))
        self.player.play()

    def stop(self) -> None:
        self.player.stop()
        self.player.setSource(QUrl())

    def _media_status_changed(self, status) -> None:
        if status in {QMediaPlayer.EndOfMedia, QMediaPlayer.InvalidMedia}:
            self.stop()
