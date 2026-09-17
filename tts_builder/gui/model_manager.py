from __future__ import annotations

import os
import threading
import time
from pathlib import Path

from ..events import CancellationToken, EventSink, PipelineEvent, emit

_REPOS = {
    "small": "Systran/faster-whisper-small",
    "large-v3": "Systran/faster-whisper-large-v3",
    "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
}
_ESTIMATED = {
    "small": 490_000_000,
    "large-v3": 3_100_000_000,
    "large-v3-turbo": 1_620_000_000,
}


def asr_repo_id(model: str) -> str:
    return model if "/" in model else _REPOS.get(model, model)


def estimated_asr_bytes(model: str) -> int | None:
    return _ESTIMATED.get(model)


def _repo_cache_dir(repo_id: str, hf_home: Path | None = None) -> Path:
    root = hf_home or Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface"))
    return Path(root) / "hub" / f"models--{repo_id.replace('/', '--')}"


def _tree_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            pass
    return total


def cached_asr_snapshot(model: str, hf_home: Path | None = None) -> Path | None:
    snapshots = _repo_cache_dir(asr_repo_id(model), hf_home) / "snapshots"
    if not snapshots.exists():
        return None
    for snapshot in snapshots.iterdir():
        if (snapshot / "model.bin").exists() and (snapshot / "config.json").exists():
            return snapshot
    return None


def is_asr_model_ready(model: str, hf_home: Path | None = None) -> bool:
    return cached_asr_snapshot(model, hf_home) is not None


def prepare_asr_model(model: str, event_sink: EventSink | None = None,
                      cancel_token: CancellationToken | None = None) -> Path:
    if cancel_token:
        cancel_token.raise_if_cancelled()
    repo = asr_repo_id(model)
    cached = cached_asr_snapshot(model)
    if cached is not None:
        emit(event_sink, PipelineEvent("log", "prepare", f"Whisper {model}: cached"))
        return cached
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required to prepare Whisper models") from exc

    stop = threading.Event()
    watcher = threading.Thread(
        target=_watch_download, args=(repo, model, stop, event_sink), daemon=True
    )
    watcher.start()
    try:
        cache_dir = Path(os.environ["HF_HOME"]) / "hub"
        path = Path(snapshot_download(repo_id=repo, cache_dir=str(cache_dir)))
    finally:
        stop.set()
        watcher.join(timeout=1.0)
    if cancel_token:
        cancel_token.raise_if_cancelled()
    return path


def prepare_demucs_model(model: str, event_sink: EventSink | None = None,
                         cancel_token: CancellationToken | None = None) -> None:
    if cancel_token:
        cancel_token.raise_if_cancelled()
    emit(event_sink, PipelineEvent("log", "prepare", f"Checking Demucs {model}"))
    try:
        from demucs.pretrained import get_model
    except ImportError as exc:
        raise RuntimeError("Demucs is required: pip install demucs") from exc
    get_model(model)
    if cancel_token:
        cancel_token.raise_if_cancelled()


def _watch_download(repo: str, model: str, stop: threading.Event, sink: EventSink | None) -> None:
    total = estimated_asr_bytes(model)
    cache = _repo_cache_dir(repo)
    while not stop.wait(0.5):
        current = _tree_size(cache)
        if total:
            current = min(current, total)
        emit(sink, PipelineEvent("stage_progress", "prepare", f"Downloading {model}", current, total))


def friendly_download_error(exc: Exception) -> tuple[str, str]:
    text = str(exc)
    low = text.lower()
    if any(key in low for key in ("timeout", "timed out", "connection", "incompleteread", "network")):
        return (
            "Network connection interrupted",
            "The model download could not finish. Downloaded cache data is preserved; Retry will resume when possible.",
        )
    if "ffmpeg" in low and "not found" in low:
        return "FFmpeg not found", "FFmpeg is required for audio preparation. Install FFmpeg and add it to PATH, then retry."
    if "no space" in low or ("disk" in low and "space" in low):
        return "Not enough storage", "Free some disk space or choose another model storage location, then retry."
    return "Model preparation failed", text or exc.__class__.__name__


def friendly_task_error(exc: Exception) -> tuple[str, str]:
    text = str(exc) or exc.__class__.__name__
    low = text.lower()
    if any(key in low for key in ("timeout", "timed out", "connection", "incompleteread", "http error")):
        return "Network connection interrupted", "The task could not continue. Completed and partial cache data was preserved; retrying can resume reusable stages."
    if "out of memory" in low or "cuda" in low and "memory" in low:
        return "GPU memory is insufficient", "Close other GPU applications or use CPU mode / a smaller ASR model, then retry."
    if "no space" in low:
        return "Not enough storage", "Free some disk space or choose another output/model location, then retry."
    return "Task failed", text
