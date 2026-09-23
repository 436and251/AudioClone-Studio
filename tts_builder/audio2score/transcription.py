from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


class BasicPitchONNX:
    def __init__(self, *, predictor=None, model_factory=None,
                 model_path: Path | None = None,
                 onset_threshold: float = 0.5,
                 frame_threshold: float = 0.3,
                 minimum_note_length: float = 127.7):
        self._predictor = predictor
        self._model_factory = model_factory
        self._model_path = model_path
        self._model = None
        self.onset_threshold = onset_threshold
        self.frame_threshold = frame_threshold
        self.minimum_note_length = minimum_note_length

    @property
    def parameters(self) -> dict:
        return {
            "model": "basic-pitch-0.4.0-onnx",
            "onset_threshold": self.onset_threshold,
            "frame_threshold": self.frame_threshold,
            "minimum_note_length": self.minimum_note_length,
        }

    def _load_model(self):
        if self._model is not None:
            return self._model
        if self._predictor is None or self._model_factory is None or self._model_path is None:
            import basic_pitch
            from basic_pitch.inference import Model, predict

            self._predictor = self._predictor or predict
            self._model_factory = self._model_factory or Model
            self._model_path = self._model_path or Path(basic_pitch.ICASSP_2022_MODEL_PATH)
        if Path(self._model_path).suffix.lower() != ".onnx":
            raise RuntimeError("Basic Pitch must use the packaged ONNX model")
        self._model = self._model_factory(self._model_path)
        return self._model

    def transcribe(self, audio_path: Path, source: str,
                   playable_range: tuple[int, int] | None = None) -> dict:
        model = self._load_model()
        _, _, note_events = self._predictor(
            audio_path,
            model,
            onset_threshold=self.onset_threshold,
            frame_threshold=self.frame_threshold,
            minimum_note_length=self.minimum_note_length,
        )
        evidence = retain_note_candidates(
            note_events, source=source, playable_range=playable_range
        )
        evidence["model"] = "basic-pitch-0.4.0-onnx"
        evidence["parameters"] = self.parameters
        return evidence


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def retain_note_candidates(raw, *, source: str,
                           playable_range: tuple[int, int] | None = None) -> dict:
    if playable_range is not None:
        low, high = playable_range
        if not (0 <= low <= high <= 127):
            raise ValueError("playable_range must be within MIDI 0..127")

    rows = []
    for raw_index, raw_note in enumerate(raw):
        if len(raw_note) != 5:
            raise ValueError("expected [start, end, pitch, confidence, pitch_bend]")
        start, end, pitch, confidence, pitch_bend = map(_json_value, raw_note)
        values = np.asarray([start, end, confidence], dtype=float)
        if not np.isfinite(values).all() or float(start) < 0 or float(end) <= float(start):
            raise ValueError("invalid note interval or confidence")
        if isinstance(pitch, bool) or int(pitch) != pitch or not 0 <= int(pitch) <= 127:
            raise ValueError("pitch must be an integer MIDI value")
        rows.append((raw_index, float(start), float(end), int(pitch), float(confidence), pitch_bend))

    candidates = []
    duplicates = []
    review = []
    for raw_index, start, end, pitch, confidence, pitch_bend in sorted(
        rows, key=lambda row: row[1]
    ):
        duplicate = next(
            (
                note for note in reversed(candidates)
                if note["pitch_midi"] == pitch
                and abs(note["start_seconds"] - start) <= 0.012
                and abs(note["end_seconds"] - end) <= 0.012
            ),
            None,
        )
        if duplicate is not None:
            duplicates.append({
                "raw_index": raw_index,
                "kept_raw_index": duplicate["raw_index"],
                "reason": "same_pitch_near_identical_interval",
            })
            continue
        note = {
            "raw_index": raw_index,
            "start_seconds": start,
            "end_seconds": end,
            "pitch_midi": pitch,
            "confidence": confidence,
        }
        if pitch_bend is not None:
            note["pitch_bend"] = _json_value(pitch_bend)
        candidates.append(note)
        if playable_range is not None and not playable_range[0] <= pitch <= playable_range[1]:
            review.append({
                "raw_index": raw_index,
                "reason": "outside_requested_range_retained",
            })

    serialized = json.dumps([list(map(_json_value, note)) for note in raw], separators=(",", ":"))
    return {
        "version": 1,
        "source": source,
        "raw_count": len(rows),
        "raw_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
        "candidates": candidates,
        "duplicates": duplicates,
        "review": review,
        "quantized": False,
        "musical_accuracy_verified": False,
    }
