import json

import numpy as np
import pytest
import soundfile as sf

from tts_builder.audio2score.config import Audio2ScoreConfig
from tts_builder.audio2score.models import SeparationResult
from tts_builder.audio2score.pipeline import process_to_stems
from tts_builder.audio2score.separation import EXPECTED_STEMS
from tts_builder.models import AcquiredSource


def install_pipeline_stubs(monkeypatch, tmp_path, fail_separate_once=False,
                           create_large_work=False):
    counters = {"acquire": 0, "decode": 0, "separate": 0}
    source_audio = tmp_path / "fixture.wav"
    sf.write(source_audio, np.zeros((100, 2), np.float32), 44100)

    def acquire(source, temp_dir, **kwargs):
        counters["acquire"] += 1
        if create_large_work:
            temp_dir.mkdir(parents=True, exist_ok=True)
            (temp_dir / "download.bin").write_bytes(b"source")
        return AcquiredSource(str(source_audio), "fixture.wav", "fixture", source)

    def decode(source, destination):
        counters["decode"] += 1
        destination.parent.mkdir(parents=True, exist_ok=True)
        sf.write(destination, np.zeros((100, 2), np.float32), 44100, subtype="FLOAT")
        return destination

    def separate(mix, project, device, event_sink=None, cancel_token=None):
        counters["separate"] += 1
        if fail_separate_once and counters["separate"] == 1:
            raise RuntimeError("interrupted")
        raw = project.work / "separation"
        evidence = raw / "htdemucs_ft"
        formal_model = raw / "htdemucs_6s"
        evidence.mkdir(parents=True, exist_ok=True)
        formal_model.mkdir(parents=True, exist_ok=True)
        sf.write(
            evidence / "bass.flac", np.zeros((100, 2), np.float32), 44100
        )
        project.stems.mkdir(parents=True, exist_ok=True)
        stems = {}
        for name in EXPECTED_STEMS:
            path = project.stems / f"{name}.wav"
            sf.write(path, np.zeros((100, 2), np.float32), 44100, subtype="FLOAT")
            stems[name] = path
        return SeparationResult(evidence, formal_model, stems)

    monkeypatch.setattr("tts_builder.audio2score.pipeline.acquire_source", acquire)
    monkeypatch.setattr("tts_builder.audio2score.pipeline.decode_analysis_mix", decode)
    monkeypatch.setattr("tts_builder.audio2score.pipeline.separate_analysis", separate)
    return counters


def test_process_to_stems_publishes_manifest_and_six_tracks(tmp_path, monkeypatch):
    install_pipeline_stubs(monkeypatch, tmp_path)
    summary = process_to_stems("source.wav", Audio2ScoreConfig(tmp_path, "Song"))
    assert summary.project_dir == tmp_path.resolve() / "Song"
    manifest = json.loads(
        (summary.project_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "stems_ready"
    assert manifest["formal_model"] == "htdemucs_6s"
    assert set(manifest["stems"]) == set(EXPECTED_STEMS)


def test_interrupted_run_resumes_at_first_incomplete_stage(tmp_path, monkeypatch):
    counters = install_pipeline_stubs(monkeypatch, tmp_path, fail_separate_once=True)
    config = Audio2ScoreConfig(tmp_path, "Song")
    with pytest.raises(RuntimeError, match="interrupted"):
        process_to_stems("source.wav", config)
    process_to_stems("source.wav", config)
    assert counters == {"acquire": 1, "decode": 1, "separate": 2}


def test_completed_formal_stems_skip_all_expensive_stages_on_rerun(tmp_path, monkeypatch):
    counters = install_pipeline_stubs(monkeypatch, tmp_path)
    config = Audio2ScoreConfig(tmp_path, "Song")
    process_to_stems("source.wav", config)
    process_to_stems("source.wav", config)
    assert counters == {"acquire": 1, "decode": 1, "separate": 1}


def test_changed_decode_signature_rebuilds_decode_and_children(tmp_path, monkeypatch):
    counters = install_pipeline_stubs(monkeypatch, tmp_path)
    process_to_stems("source.wav", Audio2ScoreConfig(tmp_path, "Song"))
    monkeypatch.setattr("tts_builder.audio2score.pipeline.DECODE_VERSION", 2)
    process_to_stems("source.wav", Audio2ScoreConfig(tmp_path, "Song"))
    assert counters["acquire"] == 1
    assert counters["decode"] == 2
    assert counters["separate"] == 2
