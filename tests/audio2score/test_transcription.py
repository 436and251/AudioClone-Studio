from pathlib import Path

import numpy as np

from tts_builder.audio2score.transcription import BasicPitchONNX, retain_note_candidates


def test_retention_only_removes_near_identical_same_pitch_intervals():
    raw = [
        [0, 0.10, 60, 0.8, None],
        [0.005, 0.105, 60, 0.7, None],
        [0.006, 0.106, 64, 0.6, None],
    ]
    result = retain_note_candidates(raw, source="guitar")
    assert [note["pitch_midi"] for note in result["candidates"]] == [60, 64]
    assert result["raw_count"] == 3
    assert result["quantized"] is False


def test_out_of_range_note_is_retained_for_review():
    result = retain_note_candidates(
        [[0, 1, 20, 0.9, None]], source="bass", playable_range=(28, 67)
    )
    assert result["candidates"][0]["pitch_midi"] == 20
    assert result["review"][0]["reason"] == "outside_requested_range_retained"


def test_numpy_pitch_bend_is_json_safe():
    result = retain_note_candidates(
        [[np.float32(0), np.float64(1), np.int64(69), np.float32(.8), np.array([1, 2])]],
        source="lead",
    )
    assert result["candidates"][0]["pitch_bend"] == [1, 2]


def test_onnx_adapter_reuses_one_model_and_returns_real_candidate_evidence():
    loaded = []
    predicted = []

    def model_factory(path):
        loaded.append(Path(path))
        return "onnx-model"

    def predictor(path, model, **options):
        predicted.append((Path(path), model, options))
        return {}, object(), [(0.1, 0.4, 69, 0.75, [1, 2])]

    adapter = BasicPitchONNX(
        predictor=predictor,
        model_factory=model_factory,
        model_path=Path("nmp.onnx"),
    )
    first = adapter.transcribe(Path("guitar.wav"), "guitar")
    second = adapter.transcribe(Path("piano.wav"), "piano")

    assert first["candidates"] == [{
        "raw_index": 0,
        "start_seconds": 0.1,
        "end_seconds": 0.4,
        "pitch_midi": 69,
        "confidence": 0.75,
        "pitch_bend": [1, 2],
    }]
    assert first["model"] == "basic-pitch-0.4.0-onnx"
    assert second["source"] == "piano"
    assert loaded == [Path("nmp.onnx")]
    assert predicted == [
        (Path("guitar.wav"), "onnx-model", {
            "onset_threshold": 0.5,
            "frame_threshold": 0.3,
            "minimum_note_length": 127.7,
        }),
        (Path("piano.wav"), "onnx-model", {
            "onset_threshold": 0.5,
            "frame_threshold": 0.3,
            "minimum_note_length": 127.7,
        }),
    ]
