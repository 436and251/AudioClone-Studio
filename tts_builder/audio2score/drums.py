from __future__ import annotations

from pathlib import Path

import numpy as np


BAND_PITCHES = {"low": 36, "mid": 38, "high": 42}


def classify_drum_candidates(onset_times, band_scores) -> dict:
    times = np.asarray(onset_times, dtype=float)
    if times.ndim != 1 or not np.isfinite(times).all():
        raise ValueError("onset times must be a finite one-dimensional array")
    if set(band_scores) != set(BAND_PITCHES):
        raise ValueError("band scores must contain low, mid, and high")
    bands = {name: np.asarray(values, dtype=float) for name, values in band_scores.items()}
    if any(values.shape != times.shape for values in bands.values()):
        raise ValueError("band scores must align with onset times")
    if any(not np.isfinite(values).all() or np.any(values < 0) for values in bands.values()):
        raise ValueError("band scores must be finite and nonnegative")

    attacks = []
    for index, onset in enumerate(times):
        scores = {name: float(values[index]) for name, values in bands.items()}
        peak = max(scores.values(), default=0.0)
        if peak <= 0:
            attacks.append({
                "onset_seconds": float(onset),
                "status": "insufficient",
                "pitches": [],
                "band_scores": scores,
            })
            continue
        active = [name for name in BAND_PITCHES if scores[name] / peak >= 0.6]
        attacks.append({
            "onset_seconds": float(onset),
            "status": "candidates",
            "pitches": [BAND_PITCHES[name] for name in active],
            "band_scores": scores,
        })
    return {
        "version": 1,
        "source": "drums",
        "attacks": attacks,
        "quantized": False,
        "musical_accuracy_verified": False,
        "limitations": [
            "band_energy_is_not_verified_drum_identity",
            "tom_and_cymbal_subtypes_require_contextual_review",
        ],
    }


def transcribe_drums(audio_path: Path) -> dict:
    import librosa

    audio, sample_rate = librosa.load(str(audio_path), sr=None, mono=True)
    onset_envelope = librosa.onset.onset_strength(y=audio, sr=sample_rate)
    onset_frames = np.asarray(
        librosa.onset.onset_detect(
            onset_envelope=onset_envelope, sr=sample_rate, units="frames"
        ),
        dtype=int,
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sample_rate)
    power = np.abs(librosa.stft(audio)) ** 2
    frequencies = np.asarray(librosa.fft_frequencies(sr=sample_rate), dtype=float)
    if power.shape[0] != len(frequencies):
        raise ValueError("STFT bins do not match frequency bins")

    masks = {
        "low": (frequencies >= 20) & (frequencies < 180),
        "mid": (frequencies >= 180) & (frequencies < 4000),
        "high": frequencies >= 4000,
    }
    scores = {name: [] for name in BAND_PITCHES}
    for frame in onset_frames:
        if frame < 0 or frame >= power.shape[1]:
            raise ValueError("onset frame falls outside the STFT")
        previous = max(0, frame - 1)
        novelty = np.maximum(power[:, frame] - power[:, previous], 0)
        for name, mask in masks.items():
            scores[name].append(float(novelty[mask].sum()))
    return classify_drum_candidates(onset_times, scores)
