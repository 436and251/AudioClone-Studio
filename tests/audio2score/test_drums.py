import sys
from types import SimpleNamespace

import numpy as np

from tts_builder.audio2score.drums import classify_drum_candidates, transcribe_drums


def test_one_onset_can_keep_kick_snare_and_hat_candidates():
    result = classify_drum_candidates(
        [1.0], {"low": [1.0], "mid": [0.9], "high": [0.8]}
    )
    assert result["attacks"][0]["pitches"] == [36, 38, 42]
    assert result["musical_accuracy_verified"] is False


def test_silent_bands_are_insufficient_not_a_rest():
    result = classify_drum_candidates(
        [1.0], {"low": [0], "mid": [0], "high": [0]}
    )
    assert result["attacks"][0]["status"] == "insufficient"
    assert "rest" not in result["attacks"][0]


def test_transcribe_drums_extracts_multiband_novelty_without_basic_pitch(monkeypatch):
    spectrum = np.zeros((6, 3), dtype=float)
    spectrum[1, 1] = 1.0
    spectrum[3, 1] = 0.95
    spectrum[5, 1] = 0.9
    fake = SimpleNamespace(
        load=lambda path, sr, mono: (np.zeros(12000), 12000),
        onset=SimpleNamespace(
            onset_strength=lambda **kwargs: np.array([0.0, 1.0, 0.0]),
            onset_detect=lambda **kwargs: np.array([1]),
        ),
        stft=lambda audio: spectrum,
        fft_frequencies=lambda sr: np.array([0, 100, 500, 1000, 3000, 5000]),
        frames_to_time=lambda frames, sr: np.array([0.5]),
    )
    monkeypatch.setitem(sys.modules, "librosa", fake)

    result = transcribe_drums("drums.wav")

    assert result["attacks"][0]["pitches"] == [36, 38, 42]
    assert "basic_pitch" not in sys.modules
