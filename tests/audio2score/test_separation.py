from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from tts_builder.audio2score.separation import (
    EXPECTED_STEMS,
    StemValidationError,
    decode_analysis_mix,
    publish_six_stems,
    separate_analysis,
)
from tts_builder.audio2score.config import Audio2ScoreConfig


def make_synthetic_stems(root, names=EXPECTED_STEMS):
    root.mkdir(parents=True)
    audio = np.zeros((44100, 2), dtype=np.float32)
    for name in names:
        sf.write(root / f"{name}.flac", audio, 44100)
    return root


def test_decode_uses_shared_stereo_float_timeline(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "tts_builder.audio2score.separation.require_ffmpeg", lambda: "ffmpeg"
    )
    monkeypatch.setattr(
        "tts_builder.audio2score.separation.subprocess.run",
        lambda args, **kwargs: calls.append(args),
    )
    decode_analysis_mix(Path("song.mp3"), tmp_path / "mix.wav")
    command = calls[0]
    assert command[command.index("-ar") + 1] == "44100"
    assert command[command.index("-ac") + 1] == "2"
    assert command[command.index("-c:a") + 1] == "pcm_f32le"


def test_publish_six_stems_is_atomic_and_complete(tmp_path):
    source = make_synthetic_stems(tmp_path / "model")
    destination = tmp_path / "formal"
    result = publish_six_stems(source, destination)
    assert set(result) == set(EXPECTED_STEMS)
    assert all(sf.info(path).subtype == "FLOAT" for path in result.values())
    assert not destination.with_name(destination.name + ".tmp").exists()


def test_missing_stem_never_publishes_partial_directory(tmp_path):
    source = make_synthetic_stems(
        tmp_path / "model", names=("vocals", "drums", "bass", "other", "guitar")
    )
    destination = tmp_path / "formal"
    with pytest.raises(StemValidationError, match="piano"):
        publish_six_stems(source, destination)
    assert not destination.exists()


def test_misaligned_stem_never_publishes_partial_directory(tmp_path):
    source = make_synthetic_stems(tmp_path / "model")
    sf.write(source / "piano.flac", np.zeros((22050, 2), np.float32), 44100)
    with pytest.raises(StemValidationError, match="frames"):
        publish_six_stems(source, tmp_path / "formal")


def test_non_stereo_timeline_is_rejected_before_publication(tmp_path):
    source = tmp_path / "model"
    source.mkdir()
    for name in EXPECTED_STEMS:
        sf.write(source / f"{name}.flac", np.zeros(44100, np.float32), 44100)
    destination = tmp_path / "formal"
    with pytest.raises(StemValidationError, match="channels"):
        publish_six_stems(source, destination)
    assert not destination.exists()


def test_score_separation_uses_distinct_full_stem_models_on_one_mix(
    tmp_path, monkeypatch
):
    calls = []
    project = Audio2ScoreConfig(tmp_path, "Song").project()
    mix = tmp_path / "mix.wav"

    def run(mixture, output, model, device, segment, sink, token, *, two_stems):
        calls.append((mixture, output, model, two_stems))

    monkeypatch.setattr("tts_builder.audio2score.separation.run_demucs", run)
    monkeypatch.setattr(
        "tts_builder.audio2score.separation._model_output",
        lambda root, model: root / model / "song",
    )
    monkeypatch.setattr(
        "tts_builder.audio2score.separation.publish_six_stems",
        lambda source, destination: {"bass": destination / "bass.wav"},
    )

    result = separate_analysis(mix, project, "cpu")

    assert calls == [
        (mix, project.work / "demucs", "htdemucs_ft", None),
        (mix, project.work / "demucs", "htdemucs_6s", None),
    ]
    assert result.evidence_dir == project.work / "demucs" / "htdemucs_ft" / "song"
    assert result.formal_model_dir == project.work / "demucs" / "htdemucs_6s" / "song"
