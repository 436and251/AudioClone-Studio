import json

import numpy as np
import pytest
import soundfile as sf

from tts_builder.audio2score.config import Audio2ScoreConfig
from tts_builder.audio2score.models import SeparationResult
from tts_builder.audio2score.pipeline import process_to_stems
from tts_builder.audio2score.separation import EXPECTED_STEMS
from tts_builder.models import AcquiredSource
from dataclasses import replace


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

    def separate(mix, project, device, event_sink=None, cancel_token=None, model_root=None):
        counters["separate"] += 1
        if fail_separate_once and counters["separate"] == 1:
            raise RuntimeError("interrupted")
        raw = project.work / "separation"
        evidence = raw / "six_stem"
        evidence.mkdir(parents=True, exist_ok=True)
        sf.write(
            evidence / "bass.wav", np.zeros((100, 2), np.float32), 44100
        )
        sf.write(project.work / "vocals_raw.wav", np.zeros((100, 2), np.float32), 44100, subtype="FLOAT")
        (project.stems / "base").mkdir(parents=True, exist_ok=True)
        stems = {}
        for name in EXPECTED_STEMS:
            path = project.stems / "base" / f"{name}.wav"
            sf.write(path, np.zeros((100, 2), np.float32), 44100, subtype="FLOAT")
            stems[name] = path
        return SeparationResult(stems)

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
    assert manifest["formal_model"] == "bs-roformer-sw+dereverb"
    assert set(manifest["stems"]) == set(EXPECTED_STEMS)
    assert all("sha256" not in item for item in manifest["stems"].values())
    assert "source_sha256" not in manifest
    assert (summary.project_dir / ".work" / "vocals_raw.wav").is_file()


def test_changed_source_rebuilds_and_failed_attempt_never_publishes_old_results(tmp_path, monkeypatch):
    from tts_builder.audio2score import pipeline
    counters = install_pipeline_stubs(monkeypatch, tmp_path)
    config = Audio2ScoreConfig(tmp_path, "Song")
    first = process_to_stems("first.wav", config)
    original = {name: path.read_bytes() for name, path in first.stems.items()}
    separate = pipeline.separate_analysis

    def interrupted(*args, **kwargs):
        raise RuntimeError("replacement interrupted")

    monkeypatch.setattr(pipeline, "separate_analysis", interrupted)
    with pytest.raises(RuntimeError, match="replacement interrupted"):
        process_to_stems("second.wav", replace(config, fresh=True))
    assert not first.manifest_path.exists()
    assert all(path.read_bytes() == original[name] for name, path in first.stems.items())
    monkeypatch.setattr(pipeline, "separate_analysis", separate)
    second = process_to_stems("second.wav", config)
    assert json.loads(second.manifest_path.read_text())["source"]["path"].endswith("second.wav")
    assert counters == {"acquire": 2, "decode": 2, "separate": 2}


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


def test_changed_separation_model_rebuilds_stems_without_redownloading_source(tmp_path, monkeypatch):
    from tts_builder.audio2score import pipeline

    counters = install_pipeline_stubs(monkeypatch, tmp_path)
    config = Audio2ScoreConfig(tmp_path, "Song")
    process_to_stems("source.wav", config)
    monkeypatch.setitem(
        pipeline.MODEL_FILES, "six_stem",
        ("replacement-six-stem", "new.ckpt", "new.yaml"),
    )
    process_to_stems("source.wav", config)
    assert counters == {"acquire": 1, "decode": 2, "separate": 2}


def test_replaced_checkpoint_invalidates_formal_stem_cache(tmp_path, monkeypatch):
    from tts_builder.audio2score.separation import MODEL_FILES

    model_root = tmp_path / "models"
    for folder, checkpoint, config_file in MODEL_FILES.values():
        assets = model_root / "bs-roformer" / folder
        assets.mkdir(parents=True)
        (assets / checkpoint).write_bytes(b"old")
        (assets / config_file).write_text("config", encoding="utf-8")
    counters = install_pipeline_stubs(monkeypatch, tmp_path)
    config = Audio2ScoreConfig(tmp_path, "Song", model_root=model_root)
    process_to_stems("source.wav", config)
    folder, checkpoint, _ = MODEL_FILES["six_stem"]
    (model_root / "bs-roformer" / folder / checkpoint).write_bytes(b"new weights")
    process_to_stems("source.wav", config)
    assert counters == {"acquire": 1, "decode": 2, "separate": 2}


def test_karaoke_checkpoint_does_not_invalidate_base_stems(tmp_path, monkeypatch):
    from tts_builder.audio2score.separation import MODEL_FILES

    model_root = tmp_path / "models"
    folder, checkpoint, config_file = MODEL_FILES["karaoke"]
    assets = model_root / "bs-roformer" / folder
    assets.mkdir(parents=True)
    (assets / checkpoint).write_bytes(b"old")
    (assets / config_file).write_text("config", encoding="utf-8")
    counters = install_pipeline_stubs(monkeypatch, tmp_path)
    config = Audio2ScoreConfig(tmp_path, "Song", model_root=model_root)
    process_to_stems("source.wav", config)
    (assets / checkpoint).write_bytes(b"new weights")
    process_to_stems("source.wav", config)
    assert counters == {"acquire": 1, "decode": 1, "separate": 1}


def test_optional_harmony_is_additive_and_cached(tmp_path, monkeypatch):
    from tts_builder.audio2score import pipeline

    counters = install_pipeline_stubs(monkeypatch, tmp_path)
    config = Audio2ScoreConfig(tmp_path, "Song")
    base = process_to_stems("source.wav", config)
    originals = {name: path.read_bytes() for name, path in base.stems.items()}
    calls = []

    def harmony(project, device, event_sink=None, cancel_token=None, model_root=None):
        calls.append(project.work / "vocals_raw.wav")
        result = {}
        (project.stems / "harmony").mkdir(exist_ok=True)
        for name in ("lead_vocals_dry", "backing_vocals"):
            path = project.stems / "harmony" / f"{name}.wav"
            sf.write(path, np.zeros((100, 2), np.float32), 44100, subtype="FLOAT")
            result[name] = path
        return result

    monkeypatch.setattr(pipeline, "separate_vocal_harmony", harmony, raising=False)
    first = pipeline.process_to_vocal_harmony("source.wav", config)
    second = pipeline.process_to_vocal_harmony("source.wav", config)
    restored = pipeline.process_to_stems("source.wav", config)
    assert len(calls) == 1
    assert counters == {"acquire": 1, "decode": 1, "separate": 1}
    assert len(first.stems) == len(second.stems) == len(restored.stems) == 8
    assert all(path.read_bytes() == originals[name] for name, path in base.stems.items())


def test_optional_harmony_missing_raw_vocals_does_not_rerun_base(tmp_path, monkeypatch):
    from tts_builder.audio2score import pipeline

    counters = install_pipeline_stubs(monkeypatch, tmp_path)
    config = Audio2ScoreConfig(tmp_path, "Song")
    base = process_to_stems("source.wav", config)
    (base.project_dir / ".work" / "vocals_raw.wav").unlink()
    with pytest.raises(FileNotFoundError, match="vocals_raw"):
        pipeline.process_to_vocal_harmony("source.wav", config)
    assert counters == {"acquire": 1, "decode": 1, "separate": 1}


def test_optional_harmony_manifest_failure_remains_retryable(tmp_path, monkeypatch):
    from tts_builder.audio2score import pipeline

    install_pipeline_stubs(monkeypatch, tmp_path)
    config = Audio2ScoreConfig(tmp_path, "Song")
    process_to_stems("source.wav", config)
    calls = []

    def harmony(project, device, event_sink=None, cancel_token=None, model_root=None):
        calls.append(1)
        destination = project.stems / "harmony"
        destination.mkdir(exist_ok=True)
        result = {}
        for name in ("lead_vocals_dry", "backing_vocals"):
            path = destination / f"{name}.wav"
            sf.write(path, np.zeros((100, 2), np.float32), 44100, subtype="FLOAT")
            result[name] = path
        return result

    original = pipeline._atomic_json
    failed = False

    def write_manifest(path, data):
        nonlocal failed
        if "optional_vocal_stems" in data and not failed:
            failed = True
            raise OSError("manifest write failed")
        original(path, data)

    monkeypatch.setattr(pipeline, "separate_vocal_harmony", harmony)
    monkeypatch.setattr(pipeline, "_atomic_json", write_manifest)
    with pytest.raises(OSError, match="manifest write failed"):
        pipeline.process_to_vocal_harmony("source.wav", config)
    pipeline.process_to_vocal_harmony("source.wav", config)
    assert len(calls) == 2
    manifest = json.loads((tmp_path / "Song" / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["optional_vocal_stems"]) == {"lead_vocals_dry", "backing_vocals"}


def test_optional_harmony_works_after_original_media_is_removed(tmp_path, monkeypatch):
    from tts_builder.audio2score import pipeline

    install_pipeline_stubs(monkeypatch, tmp_path)
    source = tmp_path / "original.wav"
    sf.write(source, np.zeros((100, 2), np.float32), 44100)
    config = Audio2ScoreConfig(tmp_path, "Song")
    process_to_stems(str(source), config)
    source.unlink()

    def harmony(project, device, event_sink=None, cancel_token=None, model_root=None):
        destination = project.stems / "harmony"
        destination.mkdir(exist_ok=True)
        result = {}
        for name in ("lead_vocals_dry", "backing_vocals"):
            path = destination / f"{name}.wav"
            sf.write(path, np.zeros((100, 2), np.float32), 44100, subtype="FLOAT")
            result[name] = path
        return result

    monkeypatch.setattr(pipeline, "separate_vocal_harmony", harmony)
    summary = pipeline.process_to_vocal_harmony(str(source), config)
    assert len(summary.stems) == 8


def test_corrupt_formal_wav_does_not_hit_stem_cache(tmp_path, monkeypatch):
    counters = install_pipeline_stubs(monkeypatch, tmp_path)
    config = Audio2ScoreConfig(tmp_path, "Song")
    summary = process_to_stems("source.wav", config)
    summary.stems["piano"].write_bytes(b"broken wav")
    process_to_stems("source.wav", config)
    assert counters["separate"] == 2
