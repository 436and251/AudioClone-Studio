from tts_builder.audio2score.config import Audio2ScoreConfig
from tts_builder.audio2score.pipeline import process_to_stems

from .test_pipeline import install_pipeline_stubs


def test_success_cleanup_keeps_formal_stems_and_removes_large_work(tmp_path, monkeypatch):
    install_pipeline_stubs(monkeypatch, tmp_path, create_large_work=True)
    summary = process_to_stems("source.wav", Audio2ScoreConfig(tmp_path, "Song"))
    assert (summary.project_dir / "stems" / "guitar.wav").is_file()
    assert (summary.project_dir / ".work" / "state.json").is_file()
    assert not (summary.project_dir / ".work" / "source").exists()
    assert not (summary.project_dir / ".work" / "mixture.wav").exists()
    assert not (summary.project_dir / ".work" / "separation").exists()


def test_keep_work_preserves_internal_analysis(tmp_path, monkeypatch):
    install_pipeline_stubs(monkeypatch, tmp_path, create_large_work=True)
    summary = process_to_stems(
        "source.wav", Audio2ScoreConfig(tmp_path, "Song", keep_work=True)
    )
    assert (summary.project_dir / ".work" / "mixture.wav").exists()
    assert (summary.project_dir / ".work" / "separation").exists()
