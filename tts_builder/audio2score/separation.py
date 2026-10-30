from __future__ import annotations

import json
import re
import os
import shutil
import subprocess
import sys
from pathlib import Path

import soundfile as sf

from ..events import CancellationToken, EventSink, PipelineEvent, emit
from ..gui.settings import AppSettings
from ..media import require_ffmpeg
from .models import Audio2ScoreProject, SeparationResult


EXPECTED_STEMS = (
    "drums", "bass", "other", "guitar", "piano",
    "vocals_dry",
)
OPTIONAL_STEMS = ("lead_vocals_dry", "backing_vocals")
MODEL_FILES = {
    "acoustic_guitar": (
        "bs-roformer-mega-acoustic-guitar",
        "bs_roformer_acoustic_guitar.ckpt", "bs_roformer_acoustic_guitar.yaml",
    ),
    "six_stem": (
        "roformer-model-bs-roformer-sw-by-jarredou",
        "BS-Rofo-SW-Fixed.ckpt", "BS-Rofo-SW-Fixed.yaml",
    ),
    "karaoke": (
        "bs-roformer-karaoke-frazer-becruily",
        "bs_roformer_karaoke_frazer_becruily.ckpt",
        "config_karaoke_frazer_becruily.yaml",
    ),
    "dereverb": (
        "roformer-model-bs-roformer-de-reverb",
        "deverb_bs_roformer_8_384dim_10depth.ckpt",
        "deverb_bs_roformer_8_384dim_10depth_config.yaml",
    ),
}
MODEL_OUTPUTS = {
    "acoustic_guitar": ("acoustic-guitar", "instrumental"),
    "six_stem": ("bass", "drums", "other", "vocals", "guitar", "piano"),
    "karaoke": ("Vocals", "instrumental"),
    "dereverb": ("noreverb", "instrumental"),
}


class StemValidationError(ValueError):
    pass


def valid_stems(stems: dict[str, Path]) -> bool:
    try:
        infos = [sf.info(path) for path in stems.values()]
    except (OSError, RuntimeError, ValueError):
        return False
    return bool(infos) and all(
        (info.samplerate, info.channels, info.frames, info.format, info.subtype) ==
        (44100, 2, infos[0].frames, "WAV", "FLOAT") for info in infos
    )


def _model_assets(model_root: Path, model: str) -> tuple[Path, Path]:
    folder, checkpoint, config = MODEL_FILES[model]
    assets = model_root / "bs-roformer" / folder
    paths = (assets / checkpoint, assets / config)
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Audio2Score model asset missing: {path}")
    return paths


def decode_analysis_mix(input_path: Path, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        require_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(input_path), "-vn", "-ar", "44100", "-ac", "2",
        "-c:a", "pcm_f32le", str(output_path),
    ]
    subprocess.run(command, check=True)
    return output_path


def publish_stems(sources: dict[str, Path], destination: Path,
                  names: tuple[str, ...] = EXPECTED_STEMS) -> dict[str, Path]:
    missing = [name for name in names if name not in sources or not sources[name].is_file()]
    if missing:
        raise StemValidationError(f"missing stems: {', '.join(missing)}")
    infos = {name: sf.info(sources[name]) for name in names}
    reference = infos[names[0]]
    if reference.samplerate != 44100 or reference.channels != 2:
        raise StemValidationError("stems must be 44.1 kHz stereo")
    for name, info in infos.items():
        if info.format != "WAV" or info.subtype != "FLOAT":
            raise StemValidationError(f"{name} must be a FLOAT WAV")
        for field in ("samplerate", "channels", "frames"):
            if getattr(info, field) != getattr(reference, field):
                raise StemValidationError(f"{name} has mismatched {field}")

    temporary = destination.with_name(destination.name + ".tmp")
    backup = destination.with_name(destination.name + ".old")
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True)
    try:
        for name in names:
            shutil.copyfile(sources[name], temporary / f"{name}.wav")
        if destination.exists():
            if backup.exists():
                shutil.rmtree(backup)
            destination.replace(backup)
        temporary.replace(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        if backup.exists() and not destination.exists():
            backup.replace(destination)
        raise
    if backup.exists():
        shutil.rmtree(backup)
    return {name: destination / f"{name}.wav" for name in names}


def run_model(input_path: Path, output: Path, model: str, device: str,
              model_root: Path, cancel_token: CancellationToken | None, *, progress=None) -> dict[str, Path]:
    checkpoint_path, config_path = _model_assets(model_root, model)
    def stamp(path: Path) -> list[str | int]:
        stat = path.stat()
        return [str(path.resolve()), stat.st_size, stat.st_mtime_ns]

    identity = {
        "model": model, "device": device,
        "input": stamp(input_path),
        "assets": [stamp(checkpoint_path), stamp(config_path)],
    }
    marker = output / "completed.json"
    result_dir = output / "result"
    outputs = {name: result_dir / f"track_{name}.wav" for name in MODEL_OUTPUTS[model]}
    if marker.is_file():
        try:
            input_info = sf.info(input_path)
            if json.loads(marker.read_text(encoding="utf-8")) == identity and all(
                path.is_file()
                and (info := sf.info(path)).frames == input_info.frames
                and info.samplerate == 44100 and info.channels == 2
                and info.format == "WAV" and info.subtype == "FLOAT"
                for path in outputs.values()
            ):
                if progress:
                    progress(1, 1)
                return outputs
        except (OSError, ValueError, RuntimeError):
            pass
    input_dir = output / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    linked_input = input_dir / "track.wav"
    linked_input.unlink(missing_ok=True)
    linked_input.hardlink_to(input_path)
    shutil.rmtree(result_dir, ignore_errors=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    if model in {"karaoke", "acoustic_guitar"}:
        corrected = re.sub(
            r"(?m)^(\s*chunk_size:\s*)882000(\s*)$", r"\g<1>881664\2",
            config_path.read_text(encoding="utf-8"), count=1,
        )
        config_path = output / f"{model}-config.yaml"
        config_path.write_text(corrected, encoding="utf-8")
    command = [
        sys.executable, "-m", "tts_builder.audio2score.model_runner",
        "--input_folder", str(input_dir.resolve()), "--store_dir", str(result_dir.resolve()),
        "--model_path", str(checkpoint_path.resolve()), "--config_path", str(config_path.resolve()),
    ]
    if device != "auto":
        command.extend(("--device", device))
    log_path = output / "model.log"
    pending = b""

    def read_progress(reader):
        nonlocal pending
        pending += reader.read()
        lines = pending.split(b"\n")
        pending = lines.pop()[-16384:]
        for line in lines:
            marker = line.find(b"A2S_PROGRESS ")
            if marker < 0 or progress is None:
                continue
            try:
                row = json.loads(line[marker + len(b"A2S_PROGRESS "):])
                current, total = row["current"], row["total"]
                if type(current) is int and type(total) is int and 0 <= current <= total and total > 0:
                    progress(current, total)
            except (ValueError, KeyError, TypeError):
                continue

    try:
        with log_path.open("wb") as log, log_path.open("rb") as progress_reader:
            process = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT,
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"},
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            try:
                while process.poll() is None:
                    log.flush()
                    read_progress(progress_reader)
                    if cancel_token:
                        cancel_token.raise_if_cancelled()
                    try:
                        process.wait(timeout=0.2)
                    except subprocess.TimeoutExpired:
                        pass
                log.flush()
                read_progress(progress_reader)
            except BaseException:
                if process.poll() is None:
                    process.terminate()
                    process.wait()
                raise
        if process.returncode:
            tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-8:]
            raise RuntimeError(f"{model} inference failed: {' | '.join(tail)}")
    finally:
        linked_input.unlink(missing_ok=True)
    input_info = sf.info(input_path)
    for name, path in outputs.items():
        if not path.is_file():
            raise StemValidationError(f"{model} missing output: {name}")
        info = sf.info(path)
        if (info.samplerate, info.channels, info.frames, info.format, info.subtype) != (
            44100, 2, input_info.frames, "WAV", "FLOAT"
        ):
            raise StemValidationError(f"{model} invalid output: {name}")
    marker.write_text(json.dumps(identity), encoding="utf-8")
    return outputs


def _run_model_stage(input_path, output, model, device, model_root, token, stage, sink):
    metadata = {"step": model}
    emit(sink, PipelineEvent("stage_started", stage, metadata=metadata))
    result = run_model(input_path, output, model, device, model_root, token,
                       progress=lambda current, total: emit(sink, PipelineEvent(
                           "stage_progress", stage, current=current, total=total, metadata=metadata)))
    emit(sink, PipelineEvent("stage_completed", stage, metadata=metadata))
    return result


def separate_analysis(mix_path: Path, project: Audio2ScoreProject, device: str,
                      event_sink: EventSink | None = None,
                      cancel_token: CancellationToken | None = None,
                      model_root: Path | None = None) -> SeparationResult:
    model_root = model_root or AppSettings.load().model_root
    for role in ("six_stem", "dereverb"):
        _model_assets(model_root, role)
    work = project.work / "separation"
    raw = _run_model_stage(mix_path, work / "six_stem", "six_stem", device,
                           model_root, cancel_token, "separate", event_sink)
    raw_vocals = project.work / "vocals_raw.wav"
    raw_vocals.parent.mkdir(parents=True, exist_ok=True)
    raw_temp = raw_vocals.with_suffix(".wav.tmp")
    shutil.copyfile(raw["vocals"], raw_temp)
    raw_temp.replace(raw_vocals)
    dry = _run_model_stage(
        raw_vocals, work / "dereverb", "dereverb",
        device, model_root, cancel_token, "separate", event_sink,
    )
    sources = {name: raw[name] for name in ("drums", "bass", "other", "guitar", "piano")}
    sources["vocals_dry"] = dry["noreverb"]
    emit(event_sink, PipelineEvent("stage_started", "separate", metadata={"step": "publish"}))
    stems = publish_stems(sources, project.stems / "base")
    return SeparationResult(stems)


def separate_vocal_harmony(project: Audio2ScoreProject, device: str,
                           event_sink: EventSink | None = None,
                           cancel_token: CancellationToken | None = None,
                           model_root: Path | None = None) -> dict[str, Path]:
    model_root = model_root or AppSettings.load().model_root
    raw_vocals = project.work / "vocals_raw.wav"
    if not raw_vocals.is_file():
        raise FileNotFoundError(f"raw vocals missing: {raw_vocals}")
    raw_info = sf.info(raw_vocals)
    if (raw_info.samplerate, raw_info.channels, raw_info.format, raw_info.subtype) != (
        44100, 2, "WAV", "FLOAT"
    ):
        raise StemValidationError(f"invalid raw vocals: {raw_vocals}")
    for role in ("karaoke", "dereverb"):
        _model_assets(model_root, role)
    work = project.work / "harmony"
    split = _run_model_stage(raw_vocals, work / "karaoke", "karaoke", device,
                             model_root, cancel_token, "vocal_harmony", event_sink)
    dry = _run_model_stage(
        split["Vocals"], work / "dereverb", "dereverb", device, model_root,
        cancel_token, "vocal_harmony", event_sink,
    )
    sources = {"lead_vocals_dry": dry["noreverb"], "backing_vocals": split["instrumental"]}
    if sf.info(sources["lead_vocals_dry"]).frames != raw_info.frames:
        raise StemValidationError("lead vocal has mismatched frames")
    emit(event_sink, PipelineEvent("stage_started", "vocal_harmony", metadata={"step": "publish"}))
    return publish_stems(sources, project.stems / "harmony", OPTIONAL_STEMS)
