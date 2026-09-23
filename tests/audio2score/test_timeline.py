import sys
from types import SimpleNamespace

import numpy as np

from tts_builder.audio2score.timeline import analyze_timeline, build_timeline


def test_timeline_keeps_half_base_double_and_three_anchors():
    result = build_timeline([0, 0.5, 1, 1.5, 2], [1, 0.4, 0.9, 0.5, 1], 2.0)
    assert [row["bpm"] for row in result["tempo_candidates"]] == [60.0, 120.0, 240.0]
    assert [row["position"] for row in result["anchors"]] == ["start", "middle", "end"]
    assert result["musical_accuracy_verified"] is False


def test_beatless_timeline_is_insufficient():
    result = build_timeline([], [], 4.0)
    assert result["status"] == "insufficient"
    assert result["primary_tempo"] is None
    assert result["meter_candidates"] == []


def test_analyze_timeline_uses_the_full_aligned_drum_timeline(monkeypatch):
    loaded = []

    def load(path, *, sr, mono):
        loaded.append((str(path), sr, mono))
        return np.zeros(8), 4

    fake = SimpleNamespace(
        load=load,
        onset=SimpleNamespace(onset_strength=lambda **kwargs: np.array([1, .4, .9, .5, 1.])),
        beat=SimpleNamespace(beat_track=lambda **kwargs: (120.0, np.arange(5))),
        frames_to_time=lambda frames, **kwargs: np.array([0, .5, 1, 1.5, 2.]),
    )
    monkeypatch.setitem(sys.modules, "librosa", fake)

    result = analyze_timeline("mix.wav", "drums.wav")

    assert loaded == [("mix.wav", None, True), ("drums.wav", 4, True)]
    assert result["duration_seconds"] == 2.0
    assert result["beat_times"] == [0.0, 0.5, 1.0, 1.5, 2.0]
