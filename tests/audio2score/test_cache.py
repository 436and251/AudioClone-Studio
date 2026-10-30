import pytest
import json

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


def test_explicit_source_replacement_invalidates_old_derivatives_but_keeps_base(tmp_path):
    project = Audio2ScoreConfig(tmp_path, "Song").project()
    ScoreCache(project, "first.wav").bind_source()
    old_files = [project.reports / "transcription.mid", project.root / "score" / "score.pdf",
                 project.stems / "harmony" / "backing_vocals.wav", project.root / "manifest.json"]
    base = project.stems / "base" / "guitar.wav"
    for path in [*old_files, base]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"old result")
    cache = ScoreCache(project, "second.wav", fresh=True)
    cache.bind_source()
    assert all(not path.exists() for path in old_files)
    assert base.read_bytes() == b"old result"
    assert cache.state["source"]["path"].endswith("second.wav")


def test_manifest_still_guards_source_when_work_cache_was_removed(tmp_path):
    project = Audio2ScoreConfig(tmp_path, "Song").project()
    project.root.mkdir(parents=True)
    (project.root / "manifest.json").write_text(json.dumps({
        "source": {"kind": "url", "url": "https://example.test/first"}
    }))
    with pytest.raises(SourceMismatchError):
        ScoreCache(project, "https://example.test/second").bind_source()


def test_fresh_rejects_its_own_input_before_deleting_anything(tmp_path):
    project = Audio2ScoreConfig(tmp_path, "Song").project()
    source = project.work / "source.wav"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"original")
    with pytest.raises(ValueError, match="source"):
        ScoreCache(project, str(source), fresh=True)
    assert source.read_bytes() == b"original"


@pytest.mark.parametrize("redirect", ["reports", "root"])
def test_fresh_rejects_redirected_output_before_deleting_work(tmp_path, redirect):
    import subprocess
    project = Audio2ScoreConfig(tmp_path, "Song").project()
    external = tmp_path / "external"
    external.mkdir()
    target = getattr(project, redirect)
    target.parent.mkdir(parents=True, exist_ok=True)
    if __import__("os").name == "nt":
        subprocess.run(["cmd", "/c", "mklink", "/J", str(target), str(external)],
                       check=True, capture_output=True)
    else:
        target.symlink_to(external, target_is_directory=True)
    project.work.mkdir(parents=True, exist_ok=True)
    marker = project.work / "keep.txt"
    marker.write_text("keep")
    try:
        with pytest.raises(ValueError, match="unsafe"):
            ScoreCache(project, "source.wav", fresh=True)
        assert marker.read_text() == "keep"
        assert external.is_dir()
    finally:
        if target.is_symlink():
            target.unlink()
        else:
            target.rmdir()
