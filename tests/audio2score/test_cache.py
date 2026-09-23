import pytest

from tts_builder.audio2score.cache import ScoreCache, SourceMismatchError
from tts_builder.audio2score.config import Audio2ScoreConfig


def test_project_refuses_a_different_bound_source(tmp_path):
    project = Audio2ScoreConfig(tmp_path, "Song").project()
    cache = ScoreCache(project, "first.wav")
    cache.bind_source({"kind": "local", "path": "first.wav"})
    with pytest.raises(SourceMismatchError):
        ScoreCache(project, "second.wav").bind_source(
            {"kind": "local", "path": "second.wav"}
        )


def test_completed_stage_requires_matching_signature_and_artifact(tmp_path):
    project = Audio2ScoreConfig(tmp_path, "Song").project()
    cache = ScoreCache(project, "source.wav")
    artifact = project.work / "mix.wav"
    artifact.parent.mkdir(parents=True)
    artifact.touch()
    cache.complete("decode", "sig-a", path=cache.stored_path(artifact))
    assert cache.stage_hit("decode", "sig-a", artifact)
    assert not cache.stage_hit("decode", "sig-b", artifact)
