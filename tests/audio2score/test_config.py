import pytest

from tts_builder.audio2score.config import Audio2ScoreConfig


@pytest.mark.parametrize("title", ["", "..", "a/b", "a\\b", "CON", "CON.txt", "bad."])
def test_invalid_windows_project_titles_are_rejected(tmp_path, title):
    with pytest.raises(ValueError):
        Audio2ScoreConfig(tmp_path, title).project()
    assert list(tmp_path.iterdir()) == []


def test_unicode_title_builds_visible_project_layout(tmp_path):
    project = Audio2ScoreConfig(tmp_path, "春日影").project()
    assert project.root == tmp_path.resolve() / "春日影"
    assert project.work == project.root / ".work"
    assert project.stems == project.root / "stems"
    assert project.reports == project.root / "reports"
