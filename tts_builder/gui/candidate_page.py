from __future__ import annotations

from dataclasses import dataclass
from functools import partial

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QButtonGroup, QFrame, QGridLayout, QHBoxLayout, QLabel, QMessageBox,
    QPushButton, QRadioButton, QVBoxLayout, QWidget,
)

from ..training_modules.artifacts import ListeningCandidate, ListeningSample
from .audio_player import AudioPlayer
from .i18n import LocaleController, Translator


@dataclass
class CandidateRow:
    language: str
    label: QLabel
    text: QLabel
    button: QPushButton
    sample: ListeningSample


@dataclass
class CandidateCard:
    widget: QFrame
    select: QRadioButton
    rows: list[CandidateRow]
    internal_id: str

    @property
    def play_buttons(self) -> list[QPushButton]:
        return [row.button for row in self.rows]


class CandidatePage(QWidget):
    promotion_requested = Signal(str)

    def __init__(
        self,
        locale_controller: LocaleController,
        parent=None,
        *,
        player=None,
    ) -> None:
        super().__init__(parent)
        self.locale_controller = locale_controller
        self.translator = Translator(locale_controller.locale)
        self.player = player or AudioPlayer(self)
        self.candidates: tuple[ListeningCandidate, ...] = ()
        self.cards: list[CandidateCard] = []
        self.placeholder_cards: list[QFrame] = []
        self._busy = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.title = QLabel()
        self.title.setObjectName("Subtitle")
        layout.addWidget(self.title)
        self.empty_hint = QLabel()
        self.empty_hint.setObjectName("Subtitle")
        self.empty_hint.setWordWrap(True)
        layout.addWidget(self.empty_hint)
        self.cards_layout = QHBoxLayout()
        layout.addLayout(self.cards_layout)
        actions = QHBoxLayout()
        actions.addStretch(1)
        self.promote_button = QPushButton()
        self.promote_button.setObjectName("Primary")
        self.promote_button.setEnabled(False)
        self.promote_button.clicked.connect(self._promote)
        actions.addWidget(self.promote_button)
        layout.addLayout(actions)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._group.buttonToggled.connect(self._selection_changed)
        locale_controller.locale_changed.connect(self._locale_changed)
        self.set_candidates(())
        self.retranslate_ui(self.translator)

    def set_candidates(self, candidates: tuple[ListeningCandidate, ...]) -> None:
        self.player.stop()
        self.candidates = tuple(candidates)
        while item := self.cards_layout.takeAt(0):
            if item.widget() is not None:
                item.widget().deleteLater()
        self._group.deleteLater()
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._group.buttonToggled.connect(self._selection_changed)
        self.cards = []
        self.placeholder_cards = []
        for candidate in self.candidates:
            card = self._card(candidate)
            self.cards.append(card)
            self._group.addButton(card.select)
            self.cards_layout.addWidget(card.widget, 1)
        if not self.candidates:
            for letter in "ABC":
                frame = QFrame()
                frame.setObjectName("Card")
                placeholder_layout = QVBoxLayout(frame)
                label = QLabel(letter)
                label.setObjectName("CandidatePlaceholderTitle")
                placeholder_layout.addWidget(label)
                placeholder_layout.addStretch(1)
                self.placeholder_cards.append(frame)
                self.cards_layout.addWidget(frame, 1)
        self.empty_hint.setVisible(not self.candidates)
        self.set_busy(self._busy)

    def selected_id(self) -> str | None:
        selected = self._group.checkedButton()
        if selected is None:
            return None
        return next(card.internal_id for card in self.cards if card.select is selected)

    def set_busy(self, busy: bool) -> None:
        self._busy = busy
        for card in self.cards:
            card.select.setEnabled(not busy)
            for button in card.play_buttons:
                button.setEnabled(not busy)
        self.promote_button.setEnabled(not busy and self.selected_id() is not None)

    def retranslate_ui(self, translator: Translator) -> None:
        self.translator = translator
        self.title.setText(translator.text("candidate.title"))
        self.empty_hint.setText(translator.text("candidate.empty"))
        self.promote_button.setText(translator.text("candidate.promote"))
        for card in self.cards:
            for row in card.rows:
                row.label.setText(translator.text(f"language.{row.language}"))
                row.button.setText(translator.text("candidate.play"))

    def _card(self, candidate: ListeningCandidate) -> CandidateCard:
        frame = QFrame()
        frame.setObjectName("Card")
        layout = QVBoxLayout(frame)
        select = QRadioButton(candidate.visible_name)
        select.setObjectName("CandidateChoice")
        layout.addWidget(select)
        grid = QGridLayout()
        rows = []
        for index, sample in enumerate(candidate.samples):
            language = QLabel(self.translator.text(f"language.{sample.language}"))
            text = QLabel(sample.text)
            text.setWordWrap(True)
            play = QPushButton(self.translator.text("candidate.play"))
            play.clicked.connect(partial(self.player.play, sample.path))
            grid.addWidget(language, index, 0)
            grid.addWidget(text, index, 1)
            grid.addWidget(play, index, 2)
            rows.append(CandidateRow(sample.language, language, text, play, sample))
        layout.addLayout(grid)
        return CandidateCard(frame, select, rows, candidate.internal_id)

    def _selection_changed(self, *_):
        self.promote_button.setEnabled(not self._busy and self.selected_id() is not None)

    def _promote(self) -> None:
        selection = self.selected_id()
        if selection is None:
            return
        answer = QMessageBox.question(
            self,
            self.translator.text("candidate.confirm_title"),
            self.translator.text("candidate.confirm", name=selection[-1]),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer == QMessageBox.Yes:
            self.promotion_requested.emit(selection)

    def _locale_changed(self, locale: str) -> None:
        self.retranslate_ui(Translator(locale))
