from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import soundfile as sf

from ..demucs_runner import run_demucs
from ..events import CancellationToken, EventSink, PipelineEvent, emit
from ..media import require_ffmpeg
from .models import Audio2ScoreProject, SeparationResult


EXPECTED_STEMS = ("vocals", "drums", "bass", "other", "guitar", "piano")


class StemValidationError(ValueError):
    pass


def decode_analysis_mix(input_path: Path, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        require_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(input_path), "-vn", "-ar", "44100", "-ac", "2",
        "-c:a", "pcm_f32le", str(output_path),
    ]
    subprocess.run(command, check=True)
    return output_path


def publish_six_stems(model_output: Path, destination: Path) -> dict[str, Path]:
    sources = {name: model_output / f"{name}.flac" for name in EXPECTED_STEMS}
    missing = [name for name, path in sources.items() if not path.is_file()]
    if missing:
        raise StemValidationError(f"missing stems: {', '.join(missing)}")

    infos = {name: sf.info(path) for name, path in sources.items()}
    reference = infos[EXPECTED_STEMS[0]]
    if reference.samplerate != 44100:
        raise StemValidationError(
            f"stems have unexpected samplerate: {reference.samplerate}"
        )
    if reference.channels != 2:
        raise StemValidationError(
            f"stems have unexpected channels: {reference.channels}"
        )
    for name, info in infos.items():
        for field in ("samplerate", "channels", "frames"):
            if getattr(info, field) != getattr(reference, field):
                raise StemValidationError(f"{name} has mismatched {field}")

    temporary = destination.with_name(destination.name + ".tmp")
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True)
    try:
        for name, source in sources.items():
            audio, sample_rate = sf.read(source, dtype="float32", always_2d=True)
            sf.write(temporary / f"{name}.wav", audio, sample_rate, subtype="FLOAT")
        if destination.exists():
            shutil.rmtree(destination)
        temporary.replace(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {name: destination / f"{name}.wav" for name in EXPECTED_STEMS}


def _model_output(work_root: Path, model: str) -> Path:
    matches = [path.parent for path in (work_root / model).glob("**/vocals.flac")]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {model} output directory, found {len(matches)}")
    return matches[0]


def separate_analysis(mix_path: Path, project: Audio2ScoreProject, device: str,
                      event_sink: EventSink | None = None,
                      cancel_token: CancellationToken | None = None) -> SeparationResult:
    work_root = project.work / "demucs"
    outputs = {}
    for current, model in enumerate(("htdemucs_ft", "htdemucs_6s"), start=1):
        run_demucs(
            mix_path, work_root, model, device, 7, event_sink, cancel_token,
            two_stems=None,
        )
        outputs[model] = _model_output(work_root, model)
        emit(
            event_sink,
            PipelineEvent("stage_progress", "separate", model, current, 2),
        )
    stems = publish_six_stems(outputs["htdemucs_6s"], project.stems)
    return SeparationResult(outputs["htdemucs_ft"], outputs["htdemucs_6s"], stems)
