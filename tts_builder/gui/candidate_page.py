from __future__ import annotations

from dataclasses import dataclass
from functools import partial

from PySide6.QtCore import Qt, Signal
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
        self._promoting_id: str | None = None
        self.promoted_id: str | None = None
        self._has_promoted_model = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)
        hero = QFrame()
        hero.setObjectName("CandidateHero")
        hero_layout = QHBoxLayout(hero)
        self.icon = QLabel("♫")
        self.icon.setObjectName("CandidateIcon")
        hero_layout.addWidget(self.icon, 0, Qt.AlignTop)
        copy = QVBoxLayout()
        self.title = QLabel()
        self.title.setObjectName("CandidateTitle")
        copy.addWidget(self.title)
        self.empty_hint = QLabel()
        self.empty_hint.setObjectName("Subtitle")
        self.empty_hint.setWordWrap(True)
        copy.addWidget(self.empty_hint)
        self.operation_hint = QLabel()
        self.operation_hint.setObjectName("CandidateStatus")
        self.operation_hint.setWordWrap(True)
        self.operation_hint.hide()
        copy.addWidget(self.operation_hint)
        hero_layout.addLayout(copy, 1)
        layout.addWidget(hero)
        self.cards_layout = QHBoxLayout()
        self.cards_layout.setSpacing(12)
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
        self._promoting_id = None
        self.promoted_id = None
        self._has_promoted_model = False
        self.operation_hint.hide()
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
            self.cards_layout.addWidget(card.widget, 1, Qt.AlignTop)
        if not self.candidates:
            for letter in "ABC":
                frame = QFrame()
                frame.setObjectName("Card")
                placeholder_layout = QVBoxLayout(frame)
                label = QLabel(letter)
                label.setObjectName("CandidatePlaceholderTitle")
                placeholder_layout.addWidget(label)
                for language in ("zh", "ja", "en"):
                    row = QFrame()
                    row.setObjectName("CandidateSample")
                    row_layout = QHBoxLayout(row)
                    row_layout.addWidget(QLabel(self.translator.text(f"language.{language}")))
                    waiting = QLabel("—")
                    waiting.setObjectName("Subtitle")
                    row_layout.addWidget(waiting, 1)
                    placeholder_layout.addWidget(row)
                self.placeholder_cards.append(frame)
                self.cards_layout.addWidget(frame, 1, Qt.AlignTop)
        self.empty_hint.show()
        self.empty_hint.setText(self.translator.text(
            "candidate.ready" if self.candidates else "candidate.empty"
        ))
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
        self.promote_button.setEnabled(
            not busy and not self._has_promoted_model and self.selected_id() is not None
        )
        self._update_button_text()

    def set_promoting(self, selection: str) -> None:
        self._promoting_id = selection
        self.operation_hint.setText(self.translator.text(
            "candidate.promoting", name=selection[-1]
        ))
        self.operation_hint.setProperty("state", "running")
        self.operation_hint.style().unpolish(self.operation_hint)
        self.operation_hint.style().polish(self.operation_hint)
        self.operation_hint.show()
        self._update_button_text()

    def set_promoted(self, selection: str | None = None) -> None:
        self.promoted_id = selection or self._promoting_id or self.selected_id()
        self._has_promoted_model = True
        self._promoting_id = None
        if self.promoted_id:
            self.operation_hint.setText(self.translator.text(
                "candidate.promoted", name=self.promoted_id[-1]
            ))
        else:
            self.operation_hint.setText(self.translator.text("candidate.promoted_ready"))
        self.operation_hint.setProperty("state", "success")
        self.operation_hint.style().unpolish(self.operation_hint)
        self.operation_hint.style().polish(self.operation_hint)
        self.operation_hint.show()
        self.set_busy(False)

    def set_promotion_error(self, message: str) -> None:
        self._promoting_id = None
        self.operation_hint.setText(message)
        self.operation_hint.setProperty("state", "failed")
        self.operation_hint.style().unpolish(self.operation_hint)
        self.operation_hint.style().polish(self.operation_hint)
        self.operation_hint.show()
        self.set_busy(False)

    def clear_promotion(self) -> None:
        self._promoting_id = None
        self.promoted_id = None
        self._has_promoted_model = False
        self.operation_hint.hide()
        self.set_busy(self._busy)

    def retranslate_ui(self, translator: Translator) -> None:
        self.translator = translator
        self.title.setText(translator.text("candidate.title"))
        self.empty_hint.setText(translator.text(
            "candidate.ready" if self.candidates else "candidate.empty"
        ))
        self.promote_button.setText(translator.text("candidate.promote"))
        for card in self.cards:
            for row in card.rows:
                row.label.setText(translator.text(f"language.{row.language}"))
                row.button.setText(translator.text("candidate.play"))
        if self._promoting_id:
            self.operation_hint.setText(translator.text(
                "candidate.promoting", name=self._promoting_id[-1]
            ))
        elif self._has_promoted_model:
            self.operation_hint.setText(
                translator.text("candidate.promoted", name=self.promoted_id[-1])
                if self.promoted_id
                else translator.text("candidate.promoted_ready")
            )
        self._update_button_text()

    def _card(self, candidate: ListeningCandidate) -> CandidateCard:
        frame = QFrame()
        frame.setObjectName("Card")
        layout = QVBoxLayout(frame)
        layout.setSpacing(10)
        select = QRadioButton(candidate.visible_name)
        select.setObjectName("CandidateChoice")
        layout.addWidget(select)
        rows = []
        for sample in candidate.samples:
            sample_frame = QFrame()
            sample_frame.setObjectName("CandidateSample")
            grid = QGridLayout(sample_frame)
            language = QLabel(self.translator.text(f"language.{sample.language}"))
            language.setObjectName("LanguageChip")
            text = QLabel(sample.text)
            text.setWordWrap(True)
            play = QPushButton(self.translator.text("candidate.play"))
            play.clicked.connect(partial(self.player.play, sample.path))
            grid.addWidget(language, 0, 0)
            grid.addWidget(text, 0, 1)
            grid.addWidget(play, 0, 2)
            layout.addWidget(sample_frame)
            rows.append(CandidateRow(sample.language, language, text, play, sample))
        return CandidateCard(frame, select, rows, candidate.internal_id)

    def _selection_changed(self, *_):
        self.promote_button.setEnabled(
            not self._busy and not self._has_promoted_model and self.selected_id() is not None
        )

    def _update_button_text(self) -> None:
        if self._promoting_id:
            text = self.translator.text("candidate.promoting", name=self._promoting_id[-1])
        elif self._has_promoted_model:
            text = (
                self.translator.text("candidate.promoted", name=self.promoted_id[-1])
                if self.promoted_id
                else self.translator.text("candidate.promoted_ready")
            )
        else:
            text = self.translator.text("candidate.promote")
        self.promote_button.setText(text)

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
        if not self.candidates:
            self.set_candidates(())
