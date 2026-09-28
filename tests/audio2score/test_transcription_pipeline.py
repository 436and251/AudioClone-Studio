import json
from copy import deepcopy

import pytest

from tts_builder.audio2score import pipeline
from tts_builder.audio2score.config import Audio2ScoreConfig
from tts_builder.audio2score.pipeline import process_to_transcription

from .test_pipeline import install_pipeline_stubs


def test_missing_transcription_runtime_fails_before_source_acquisition(
    tmp_path, monkeypatch
):
    checked = []

    def require_runtime():
        checked.append(True)
        raise RuntimeError("Audio2Score dependencies are missing")

    monkeypatch.setattr(
        pipeline, "require_audio2score_runtime", require_runtime, raising=False
    )
    with pytest.raises(RuntimeError, match="dependencies are missing"):
        process_to_transcription("missing.wav", Audio2ScoreConfig(tmp_path, "Song"))
    assert checked == [True]


def test_transcription_uses_shared_preparation_not_stems_pipeline(
    tmp_path, monkeypatch
):
    counters = install_pipeline_stubs(monkeypatch, tmp_path)
    config = Audio2ScoreConfig(tmp_path, "Song")
    pipeline.process_to_stems("source.wav", config)
    assert not (config.project().work / "separation").exists()

    def nested_pipeline(*args, **kwargs):
        raise AssertionError("transcription must not nest process_to_stems")

    monkeypatch.setattr(pipeline, "process_to_stems", nested_pipeline)
    monkeypatch.setattr(
        pipeline,
        "analyze_timeline",
        lambda mixture, drums: {"primary_tempo": 120.0},
    )

    class FakeBasicPitch:
        parameters = {"model": "fake-onnx"}

        def transcribe(self, audio_path, source, playable_range=None):
            return {"source": source, "candidates": []}

    monkeypatch.setattr(pipeline, "BasicPitchONNX", FakeBasicPitch)
    monkeypatch.setattr(
        pipeline, "transcribe_drums", lambda audio_path: {"attacks": []}
    )

    summary = process_to_transcription(
        "source.wav", config
    )

    assert summary.transcription_path.is_file()
    assert counters["separate"] == 2


def test_failed_guitar_route_resumes_without_repeating_completed_stages(
    tmp_path, monkeypatch
):
    counters = install_pipeline_stubs(monkeypatch, tmp_path)
    counters.update(timeline=0, drums=0, routes={})
    fail_guitar = {"once": True}

    def timeline(mixture, drums):
        counters["timeline"] += 1
        return {
            "status": "candidate",
            "duration_seconds": 1.0,
            "tempo_candidates": [{"relation": "base", "bpm": 120.0}],
            "primary_tempo": 120.0,
            "beat_times": [0.0, 0.5, 1.0],
            "anchors": [],
            "meter_candidates": [],
            "musical_accuracy_verified": False,
        }

    class FakeBasicPitch:
        parameters = {"model": "fake-onnx", "onset_threshold": 0.5}

        def transcribe(self, audio_path, source, playable_range=None):
            counters["routes"][source] = counters["routes"].get(source, 0) + 1
            if source == "guitar" and fail_guitar["once"]:
                fail_guitar["once"] = False
                raise RuntimeError("guitar interrupted")
            return {
                "version": 1,
                "source": source,
                "raw_count": 1,
                "candidates": [{"start_seconds": 0.0, "end_seconds": 0.5,
                                "pitch_midi": 60, "confidence": 0.8}],
                "duplicates": [],
                "review": [],
                "quantized": False,
                "musical_accuracy_verified": False,
                "model": "fake-onnx",
            }

    def drums(audio_path):
        counters["drums"] += 1
        return {"version": 1, "source": "drums", "attacks": [],
                "quantized": False, "musical_accuracy_verified": False}

    monkeypatch.setattr("tts_builder.audio2score.pipeline.analyze_timeline", timeline)
    monkeypatch.setattr("tts_builder.audio2score.pipeline.BasicPitchONNX", FakeBasicPitch)
    monkeypatch.setattr("tts_builder.audio2score.pipeline.transcribe_drums", drums)
    config = Audio2ScoreConfig(tmp_path, "Song")

    with pytest.raises(RuntimeError, match="guitar interrupted"):
        process_to_transcription("source.wav", config)

    events = []
    summary = process_to_transcription("source.wav", config, events.append)

    assert counters["acquire"] == 1
    assert counters["decode"] == 1
    assert counters["separate"] == 1
    assert counters["timeline"] == 1
    assert counters["routes"] == {
        "bass_ft": 1,
        "bass_6s": 1,
        "guitar": 2,
        "piano": 1,
    }
    assert counters["drums"] == 1
    progress = [event for event in events if event.kind == "stage_progress"
                and event.stage == "transcribe"]
    assert [(event.current, event.total) for event in progress] == [
        (1, 5), (2, 5), (3, 5), (4, 5), (5, 5)
    ]
    timeline_report = json.loads(summary.timeline_path.read_text(encoding="utf-8"))
    transcription = json.loads(summary.transcription_path.read_text(encoding="utf-8"))
    assert timeline_report["primary_tempo"] == 120.0
    assert set(transcription["routes"]) == {"bass_ft", "bass_6s", "guitar", "piano"}
    assert transcription["quantized"] is False
    assert transcription["musical_accuracy_verified"] is False
    assert not (summary.project_dir / ".work" / "mixture.wav").exists()
    assert not (summary.project_dir / ".work" / "separation").exists()
    assert not (summary.project_dir / ".work" / "transcription").exists()

    completed_counts = deepcopy(counters)
    process_to_transcription("source.wav", config)
    assert counters == completed_counts
