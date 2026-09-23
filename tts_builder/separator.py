from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from .demucs_runner import build_demucs_args, run_demucs
from .events import CancellationToken, EventSink


def build_demucs_command(input_path: Path, output_root: Path, model: str = "htdemucs",
                         device: str = "auto", segment_seconds: float = 7.0) -> list[str]:
    return [sys.executable, "-m", "demucs", *build_demucs_args(
        input_path, output_root, model, device, segment_seconds
    )]


def separate_vocals(input_path: Path, output_root: Path, model: str = "htdemucs",
                     device: str = "auto", segment_seconds: float = 7.0, *,
                     event_sink: EventSink | None = None,
                     cancel_token: CancellationToken | None = None) -> Path:
    """Run Demucs through its Python entry point in the active source environment."""
    if importlib.util.find_spec("demucs") is None:
        raise RuntimeError("Demucs is required: pip install demucs")
    run_demucs(input_path, output_root, model, device, segment_seconds, event_sink, cancel_token)
    vocals = list(output_root.glob(f"{model}/**/vocals.flac"))
    if len(vocals) != 1:
        raise RuntimeError(f"expected one Demucs vocals.flac, found {len(vocals)}")
    return vocals[0]
