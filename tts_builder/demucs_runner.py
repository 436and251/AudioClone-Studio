from __future__ import annotations

from pathlib import Path

from .events import CancellationToken, EventSink, PipelineEvent, emit


def build_demucs_args(input_path: Path, output_root: Path, model: str, device: str,
                      segment_seconds: float, *,
                      two_stems: str | None = "vocals") -> list[str]:
    args = ["-n", model]
    if two_stems is not None:
        args.extend([f"--two-stems={two_stems}", "--other-method", "none"])
    args.extend([
        "--flac", "--segment", str(int(segment_seconds)), "-o", str(output_root),
    ])
    if device != "auto":
        args.extend(["-d", device])
    args.append(str(input_path))
    return args


def run_demucs(input_path: Path, output_root: Path, model: str, device: str,
               segment_seconds: float, event_sink: EventSink | None = None,
               cancel_token: CancellationToken | None = None, *,
               two_stems: str | None = "vocals") -> None:
    if cancel_token:
        cancel_token.raise_if_cancelled()
    try:
        from demucs.separate import main as demucs_main
    except ImportError as exc:
        raise RuntimeError("Demucs is required: pip install demucs") from exc
    args = build_demucs_args(
        input_path, output_root, model, device, segment_seconds, two_stems=two_stems
    )
    emit(event_sink, PipelineEvent("log", "separate", f"Demucs model: {model}"))
    demucs_main(args)
    if cancel_token:
        cancel_token.raise_if_cancelled()
