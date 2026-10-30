from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from tts_builder.audio2score.separation import (
    EXPECTED_STEMS,
    StemValidationError,
    decode_analysis_mix,
    publish_stems,
    run_model,
    separate_analysis,
)
from tts_builder.audio2score.config import Audio2ScoreConfig


def make_synthetic_stems(root, names=EXPECTED_STEMS):
    root.mkdir(parents=True)
    audio = np.zeros((44100, 2), dtype=np.float32)
    for name in names:
        sf.write(root / f"{name}.wav", audio, 44100, subtype="FLOAT")
    return {name: root / f"{name}.wav" for name in names}


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
    result = publish_stems(source, destination)
    assert set(result) == set(EXPECTED_STEMS)
    assert set(result) == {
        "drums", "bass", "other", "guitar", "piano",
        "vocals_dry",
    }
    assert all(sf.info(path).subtype == "FLOAT" for path in result.values())
    assert not destination.with_name(destination.name + ".tmp").exists()


def test_missing_stem_never_publishes_partial_directory(tmp_path):
    source = make_synthetic_stems(
        tmp_path / "model", names=tuple(name for name in EXPECTED_STEMS if name != "piano")
    )
    destination = tmp_path / "formal"
    with pytest.raises(StemValidationError, match="piano"):
        publish_stems(source, destination)
    assert not destination.exists()


def test_misaligned_stem_never_publishes_partial_directory(tmp_path):
    source = make_synthetic_stems(tmp_path / "model")
    sf.write(source["piano"], np.zeros((22050, 2), np.float32), 44100, subtype="FLOAT")
    with pytest.raises(StemValidationError, match="frames"):
        publish_stems(source, tmp_path / "formal")


def test_non_stereo_timeline_is_rejected_before_publication(tmp_path):
    source = tmp_path / "model"
    source.mkdir()
    for name in EXPECTED_STEMS:
        sf.write(source / f"{name}.wav", np.zeros(44100, np.float32), 44100)
    destination = tmp_path / "formal"
    with pytest.raises(StemValidationError, match="stereo"):
        publish_stems({name: source / f"{name}.wav" for name in EXPECTED_STEMS}, destination)
    assert not destination.exists()


def test_non_float_source_is_rejected_before_publication(tmp_path):
    sources = make_synthetic_stems(tmp_path / "model")
    sf.write(sources["guitar"], np.zeros((44100, 2), np.float32), 44100, subtype="PCM_16")
    with pytest.raises(StemValidationError, match="FLOAT"):
        publish_stems(sources, tmp_path / "formal")


def test_score_separation_skips_karaoke_and_preserves_raw_vocals(
    tmp_path, monkeypatch
):
    from tts_builder.audio2score.separation import MODEL_FILES

    calls = []
    project = Audio2ScoreConfig(tmp_path, "Song").project()
    mix = tmp_path / "mix.wav"
    sf.write(mix, np.zeros((44100, 2), np.float32), 44100)
    model_root = tmp_path / "models"
    for role in ("six_stem", "dereverb"):
        folder, checkpoint, config = MODEL_FILES[role]
        assets = model_root / "bs-roformer" / folder
        assets.mkdir(parents=True)
        (assets / checkpoint).write_bytes(b"checkpoint")
        (assets / config).write_text("config", encoding="utf-8")

    def run(input_path, output, model, device, model_root, token, *, progress=None):
        calls.append((input_path, model))
        names = {
            "six_stem": ("vocals", "drums", "bass", "other", "guitar", "piano"),
            "karaoke": ("Vocals", "instrumental"),
            "dereverb": ("noreverb", "instrumental"),
        }[model]
        output.mkdir(parents=True, exist_ok=True)
        for name in names:
            sf.write(output / f"{name}.wav", np.zeros((44100, 2), np.float32), 44100, subtype="FLOAT")
        return {name: output / f"{name}.wav" for name in names}

    monkeypatch.setattr("tts_builder.audio2score.separation.run_model", run, raising=False)

    harmony = project.stems / "harmony" / "backing_vocals.wav"
    harmony.parent.mkdir(parents=True)
    harmony.write_bytes(b"existing harmony")

    result = separate_analysis(mix, project, "cpu", model_root=model_root)

    assert [(path.name, model) for path, model in calls] == [
        ("mix.wav", "six_stem"),
        ("vocals_raw.wav", "dereverb"),
    ]
    assert set(result.stems) == {
        "drums", "bass", "other", "guitar", "piano",
        "vocals_dry",
    }
    assert all(path.is_file() for path in result.stems.values())
    assert all(path.parent == project.stems / "base" for path in result.stems.values())
    assert harmony.read_bytes() == b"existing harmony"
    assert (project.work / "vocals_raw.wav").is_file()


def test_optional_harmony_uses_only_raw_vocals_and_preserves_base_stems(tmp_path, monkeypatch):
    from tts_builder.audio2score import separation

    project = Audio2ScoreConfig(tmp_path, "Song").project()
    raw = project.work / "vocals_raw.wav"
    raw.parent.mkdir(parents=True)
    sf.write(raw, np.zeros((100, 2), np.float32), 44100, subtype="FLOAT")
    base = make_synthetic_stems(project.stems)
    original = {name: path.read_bytes() for name, path in base.items()}
    calls = []

    def run(input_path, output, model, device, model_root, token, *, progress=None):
        calls.append((input_path, model))
        output.mkdir(parents=True, exist_ok=True)
        names = ("Vocals", "instrumental") if model == "karaoke" else ("noreverb", "instrumental")
        result = {}
        for name in names:
            path = output / f"{name}.wav"
            sf.write(path, np.zeros((100, 2), np.float32), 44100, subtype="FLOAT")
            result[name] = path
        return result

    monkeypatch.setattr(separation, "run_model", run)
    monkeypatch.setattr(separation, "_model_assets", lambda *args: None)
    result = separation.separate_vocal_harmony(project, "cpu", model_root=tmp_path)
    assert [model for _, model in calls] == ["karaoke", "dereverb"]
    assert calls[0][0] == raw
    assert set(result) == {"lead_vocals_dry", "backing_vocals"}
    assert all(path.is_file() for path in result.values())
    assert all(path.read_bytes() == original[name] for name, path in base.items())


@pytest.mark.parametrize("role,names,config_text", [
    ("karaoke", ("Vocals", "instrumental"), "audio:\n  chunk_size: 882000\n"),
    ("acoustic_guitar", ("acoustic-guitar", "instrumental"),
     "audio:\n  chunk_size: 441000\ninference:\n  chunk_size: 882000\n"),
])
def test_runner_uses_configured_weights_and_aligned_chunks(tmp_path, monkeypatch, role, names, config_text):
    from tts_builder.audio2score.separation import MODEL_FILES

    folder, checkpoint, config = MODEL_FILES[role]
    assets = tmp_path / "models" / "bs-roformer" / folder
    assets.mkdir(parents=True)
    (assets / checkpoint).write_bytes(b"checkpoint")
    (assets / config).write_text(config_text, encoding="utf-8")
    vocal = tmp_path / "vocal.wav"
    sf.write(vocal, np.zeros((512, 2), np.float32), 44100, subtype="FLOAT")
    commands = []

    class CompletedProcess:
        returncode = 0

        def poll(self):
            return 0

    def launch(command, **kwargs):
        commands.append(command)
        config_path = Path(command[command.index("--config_path") + 1])
        assert "chunk_size: 881664" in config_path.read_text(encoding="utf-8")
        output = Path(command[command.index("--store_dir") + 1])
        output.mkdir(parents=True, exist_ok=True)
        for name in names:
            sf.write(output / f"track_{name}.wav", np.zeros((512, 2), np.float32), 44100, subtype="FLOAT")
        return CompletedProcess()

    monkeypatch.setattr("tts_builder.audio2score.separation.subprocess.Popen", launch)
    outputs = run_model(vocal, tmp_path / "work", role, "cuda:0", tmp_path / "models", None)
    repeated = run_model(vocal, tmp_path / "work", role, "cuda:0", tmp_path / "models", None)
    assert set(outputs) == set(names)
    assert repeated == outputs
    assert len(commands) == 1
    assert commands[0][commands[0].index("--model_path") + 1] == str(assets / checkpoint)
    assert commands[0][-2:] == ["--device", "cuda:0"]
    assert not (tmp_path / "work" / "input" / "track.wav").exists()


def test_failed_model_run_can_retry_without_stale_input_link(tmp_path, monkeypatch):
    from tts_builder.audio2score.separation import MODEL_FILES

    folder, checkpoint, config = MODEL_FILES["six_stem"]
    assets = tmp_path / "models" / "bs-roformer" / folder
    assets.mkdir(parents=True)
    (assets / checkpoint).write_bytes(b"checkpoint")
    (assets / config).write_text("model: test\n", encoding="utf-8")
    vocal = tmp_path / "vocal.wav"
    sf.write(vocal, np.zeros((512, 2), np.float32), 44100, subtype="FLOAT")
    stale = tmp_path / "work" / "input" / "track.wav"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"incomplete prior attempt")

    class CompletedProcess:
        returncode = 0

        def poll(self):
            return 0

    def launch(command, **kwargs):
        output = Path(command[command.index("--store_dir") + 1])
        output.mkdir(parents=True, exist_ok=True)
        for name in ("bass", "drums", "other", "vocals", "guitar", "piano"):
            sf.write(output / f"track_{name}.wav", np.zeros((512, 2), np.float32), 44100, subtype="FLOAT")
        return CompletedProcess()

    monkeypatch.setattr("tts_builder.audio2score.separation.subprocess.Popen", launch)
    result = run_model(vocal, tmp_path / "work", "six_stem", "cpu", tmp_path / "models", None)
    assert result["bass"].is_file()
    assert not stale.exists()


@pytest.mark.parametrize("device_args", [[], ["--device", "cuda:0"]])
def test_model_child_rejects_cpu_only_runtime_before_model_load(monkeypatch, device_args):
    from tts_builder.audio2score import model_runner
    import sys
    monkeypatch.setattr(sys, "argv", ["model_runner", *device_args])
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr(
        "bs_roformer.inference.main",
        lambda: pytest.fail("CPU fallback launched"),
    )

    with pytest.raises(RuntimeError, match="CUDA"):
        model_runner.main()


def test_model_failure_reports_error_without_streaming_progress_noise(tmp_path, monkeypatch):
    from tts_builder.audio2score.separation import MODEL_FILES

    folder, checkpoint, config = MODEL_FILES["six_stem"]
    assets = tmp_path / "models" / "bs-roformer" / folder
    assets.mkdir(parents=True)
    (assets / checkpoint).write_bytes(b"checkpoint")
    (assets / config).write_text("model: test\n", encoding="utf-8")
    source = tmp_path / "source.wav"
    sf.write(source, np.zeros((512, 2), np.float32), 44100, subtype="FLOAT")

    class FailedProcess:
        returncode = 1

        def poll(self):
            return 1

    def launch(command, **kwargs):
        kwargs["stdout"].write(b"progress noise\nmodel load failed\n")
        return FailedProcess()

    monkeypatch.setattr("tts_builder.audio2score.separation.subprocess.Popen", launch)
    with pytest.raises(RuntimeError, match="model load failed"):
        run_model(source, tmp_path / "work", "six_stem", "cpu", tmp_path / "models", None)


def test_missing_dereverb_model_is_reported_before_any_inference(tmp_path, monkeypatch):
    from tts_builder.audio2score.separation import MODEL_FILES

    model_root = tmp_path / "models"
    for role, (folder, checkpoint, config) in MODEL_FILES.items():
        assets = model_root / "bs-roformer" / folder
        assets.mkdir(parents=True)
        if role != "dereverb":
            (assets / checkpoint).write_bytes(b"checkpoint")
        (assets / config).write_text("config", encoding="utf-8")
    mix = tmp_path / "mix.wav"
    sf.write(mix, np.zeros((512, 2), np.float32), 44100, subtype="FLOAT")
    monkeypatch.setattr(
        "tts_builder.audio2score.separation.run_model",
        lambda *args: pytest.fail("inference started before asset preflight"),
    )

    with pytest.raises(FileNotFoundError, match="de-reverb"):
        separate_analysis(mix, Audio2ScoreConfig(tmp_path, "Song").project(), "cpu",
                          model_root=model_root)
