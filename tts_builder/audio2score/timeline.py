from __future__ import annotations

from pathlib import Path

import numpy as np


def _meter_candidates(strengths: np.ndarray) -> list[dict]:
    candidates = []
    for beats_per_group, label in ((3, "3/4"), (4, "4/4"), (6, "6/8")):
        if strengths.size < beats_per_group * 2:
            continue
        phase_means = [
            float(np.mean(strengths[offset::beats_per_group]))
            for offset in range(beats_per_group)
        ]
        strongest = max(phase_means)
        baseline = float(np.mean(phase_means))
        score = 0.0 if strongest <= 0 else max(0.0, (strongest - baseline) / strongest)
        candidates.append({"meter": label, "score": round(score, 6)})
    return sorted(candidates, key=lambda item: item["score"], reverse=True)


def build_timeline(beat_times, beat_strengths, duration_seconds: float) -> dict:
    times = np.asarray(beat_times, dtype=float)
    strengths = np.asarray(beat_strengths, dtype=float)
    if duration_seconds <= 0 or not np.isfinite(duration_seconds):
        raise ValueError("duration_seconds must be positive and finite")
    if times.ndim != 1 or strengths.ndim != 1 or len(times) != len(strengths):
        raise ValueError("beat times and strengths must be aligned one-dimensional arrays")
    if not np.isfinite(times).all() or not np.isfinite(strengths).all():
        raise ValueError("beat evidence must be finite")

    intervals = np.diff(times)
    positive = intervals[intervals > 0]
    if len(times) < 2 or positive.size == 0:
        return {
            "status": "insufficient",
            "duration_seconds": float(duration_seconds),
            "tempo_candidates": [],
            "primary_tempo": None,
            "beat_times": times.tolist(),
            "anchors": [],
            "meter_candidates": [],
            "musical_accuracy_verified": False,
        }

    base_bpm = round(60.0 / float(np.median(positive)), 6)
    candidates = [
        {"relation": "half", "bpm": round(base_bpm / 2, 6)},
        {"relation": "base", "bpm": base_bpm},
        {"relation": "double", "bpm": round(base_bpm * 2, 6)},
    ]
    anchors = []
    for position, target in (
        ("start", 0.0),
        ("middle", duration_seconds / 2),
        ("end", duration_seconds),
    ):
        index = int(np.argmin(np.abs(times - target)))
        anchors.append({
            "position": position,
            "target_seconds": float(target),
            "beat_index": index,
            "beat_seconds": float(times[index]),
        })
    return {
        "status": "candidate",
        "duration_seconds": float(duration_seconds),
        "tempo_candidates": candidates,
        "primary_tempo": base_bpm,
        "beat_times": times.tolist(),
        "anchors": anchors,
        "meter_candidates": _meter_candidates(strengths),
        "musical_accuracy_verified": False,
    }


def analyze_timeline(mixture: Path, drums: Path) -> dict:
    import librosa

    mix, sample_rate = librosa.load(str(mixture), sr=None, mono=True)
    drum_audio, drum_rate = librosa.load(str(drums), sr=sample_rate, mono=True)
    if drum_rate != sample_rate or len(drum_audio) != len(mix):
        raise ValueError("mixture and drums must share one complete timeline")
    envelope = np.asarray(
        librosa.onset.onset_strength(y=drum_audio, sr=sample_rate), dtype=float
    )
    _, beat_frames = librosa.beat.beat_track(
        onset_envelope=envelope, sr=sample_rate, sparse=True
    )
    beat_frames = np.asarray(beat_frames, dtype=int)
    if np.any(beat_frames < 0) or np.any(beat_frames >= len(envelope)):
        raise ValueError("librosa returned beat frames outside the onset envelope")
    beat_times = librosa.frames_to_time(beat_frames, sr=sample_rate)
    duration = len(mix) / sample_rate
    return build_timeline(beat_times, envelope[beat_frames], duration)
