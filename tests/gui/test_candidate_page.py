import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QMessageBox

from tts_builder.gui.app import create_application
from tts_builder.gui.i18n import LocaleController
from tts_builder.training_modules.artifacts import load_listening_manifest


class FakePlayer:
    def __init__(self):
        self.played = []
        self.stops = 0

    def play(self, path):
        self.stops += 1
        self.played.append(path)

    def stop(self):
        self.stops += 1


def _candidates(tmp_path: Path):
    entries = []
    for letter in "ABC":
        samples = []
        for language in ("zh", "ja", "en"):
            path = tmp_path / f"candidate_{letter}" / f"{language}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"{letter}-{language}".encode())
            samples.append(
                {
                    "language": language,
                    "text": f"{language} sample",
                    "wav": f"candidate_{letter}/{language}.wav",
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        entries.append({"candidate": f"candidate_{letter}", "samples": samples})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"schema_version": 1, "candidates": entries}), encoding="utf-8"
    )
    return load_listening_manifest(manifest, tmp_path)


def test_candidate_cards_show_only_abc_three_languages_and_one_player(tmp_path):
    from tts_builder.gui.candidate_page import CandidatePage

    app = create_application([])
    player = FakePlayer()
    page = CandidatePage(LocaleController("en"), player=player)
    page.set_candidates(_candidates(tmp_path))
    app.processEvents()

    assert [card.select.text() for card in page.cards] == ["A", "B", "C"]
    assert all("candidate" not in card.select.text().lower() for card in page.cards)
    assert all(len(card.play_buttons) == 3 for card in page.cards)
    assert [row.language for row in page.cards[0].rows] == ["zh", "ja", "en"]
    initial_stops = player.stops
    page.cards[0].play_buttons[0].click()
    page.cards[1].play_buttons[1].click()
    assert len(player.played) == 2
    assert player.stops - initial_stops == 2
    page.close()


def test_candidate_selection_is_exclusive_and_confirmation_emits_internal_id(
    tmp_path, monkeypatch
):
    from tts_builder.gui.candidate_page import CandidatePage

    app = create_application([])
    page = CandidatePage(LocaleController("zh_CN"), player=FakePlayer())
    page.set_candidates(_candidates(tmp_path))
    selected = []
    page.promotion_requested.connect(selected.append)
    assert not page.promote_button.isEnabled()

    page.cards[0].select.click()
    page.cards[1].select.click()
    app.processEvents()
    assert [card.select.isChecked() for card in page.cards] == [False, True, False]
    assert page.promote_button.isEnabled()
    monkeypatch.setattr(QMessageBox, "question", lambda *_args, **_kwargs: QMessageBox.Yes)
    page.cards[0].select.click()
    page.promote_button.click()

    assert selected == ["candidate_A"]
    assert len(page.cards) == 3
    page.close()
