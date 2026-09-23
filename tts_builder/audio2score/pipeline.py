from __future__ import annotations

import hashlib
import importlib.metadata
import json
import shutil
from pathlib import Path

from ..cache import source_identity
from ..events import CancellationToken, EventSink, PipelineCancelled, PipelineEvent, emit
from ..media import acquire_source
from ..models import AcquiredSource
from .cache import ScoreCache
from .config import Audio2ScoreConfig
from .models import Audio2ScoreSummary
from .separation import EXPECTED_STEMS, decode_analysis_mix, separate_analysis


DECODE_VERSION = 1
DECODE_PARAMETERS = {"sample_rate": 44100, "channels": 2, "codec": "pcm_f32le"}
SEPARATION_PARAMETERS = {
    "models": ["htdemucs_ft", "htdemucs_6s"],
    "segment_seconds": 7,
    "formal_model": "htdemucs_6s",
}


def _event(sink: EventSink | None, kind: str, stage: str | None = None,
           message: str = "", current=None, total=None, **metadata) -> None:
    emit(sink, PipelineEvent(kind, stage, message, current, total, metadata))


def _check(token: CancellationToken | None) -> None:
    if token:
        token.raise_if_cancelled()


def _run_stage(cache: ScoreCache, name: str, signature: str, action, *,
               sink: EventSink | None, token: CancellationToken | None):
    _check(token)
    cache.begin(name, signature)
    _event(sink, "stage_started", name)
    try:
        result = action()
        _check(token)
    except PipelineCancelled:
        _event(sink, "pipeline_cancelled", name, "Cancelled")
        raise
    except Exception as exc:
        cache.fail(name, signature, exc)
        _event(sink, "stage_failed", name, str(exc))
        raise
    _event(sink, "stage_completed", name)
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _demucs_version() -> str:
    try:
        return importlib.metadata.version("demucs")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _formal_stems(project_root: Path) -> dict[str, Path]:
    return {name: project_root / "stems" / f"{name}.wav" for name in EXPECTED_STEMS}


def _formal_hit(cache: ScoreCache, signature: str, manifest: Path,
                stems: dict[str, Path]) -> bool:
    return (
        cache.stage_hit("separate", signature)
        and manifest.is_file()
        and all(path.is_file() for path in stems.values())
    )


def _load_acquired(cache: ScoreCache, signature: str) -> AcquiredSource | None:
    stage = cache.stage_data("source")
    if stage.get("status") != "completed" or stage.get("signature") != signature:
        return None
    path = cache.resolved_path(stage.get("path", ""))
    if not path.is_file():
        return None
    return AcquiredSource(
        str(path), stage["label"], stage["source_id"], stage["original"]
    )


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _cleanup(project, sink: EventSink | None) -> None:
    removed = []
    for path in (
        project.work / "source",
        project.work / "mixture.wav",
        project.work / "demucs",
        project.work / "separation",
    ):
        if not path.exists():
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        removed.append(path.name)
    _event(sink, "cleanup_completed", "cleanup", ", ".join(removed))


def process_to_stems(source: str, config: Audio2ScoreConfig,
                     event_sink: EventSink | None = None,
                     cancel_token: CancellationToken | None = None) -> Audio2ScoreSummary:
    _event(event_sink, "pipeline_started", message=source)
    project = config.project()
    cache = ScoreCache(project, source, fresh=config.fresh)
    identity = source_identity(source)
    cache.bind_source(identity)

    source_sig = cache.signature({"source": identity})
    decode_sig = cache.signature({
        "parent": source_sig, "version": DECODE_VERSION, **DECODE_PARAMETERS,
    })
    separate_sig = cache.signature({
        "parent": decode_sig, "device": config.device, **SEPARATION_PARAMETERS,
    })
    manifest_path = project.root / "manifest.json"
    stems = _formal_stems(project.root)

    if _formal_hit(cache, separate_sig, manifest_path, stems):
        _event(event_sink, "stage_cache_hit", "separate", "Formal stems")
        _event(event_sink, "pipeline_completed", message="Stems ready")
        return Audio2ScoreSummary(project.root, manifest_path, stems)

    try:
        acquired = _load_acquired(cache, source_sig)
        if acquired is None:
            acquired = _run_stage(
                cache, "source", source_sig,
                lambda: acquire_source(
                    source, project.work / "source",
                    event_sink=event_sink, cancel_token=cancel_token,
                ),
                sink=event_sink, token=cancel_token,
            )
            cache.complete(
                "source", source_sig, path=cache.stored_path(Path(acquired.path)),
                label=acquired.label, source_id=acquired.source_id,
                original=acquired.original,
            )
        else:
            _event(event_sink, "stage_cache_hit", "source", "Cached source")

        mixture = project.work / "mixture.wav"
        if cache.stage_hit("decode", decode_sig, mixture):
            _event(event_sink, "stage_cache_hit", "decode", "Cached mixture")
        else:
            _run_stage(
                cache, "decode", decode_sig,
                lambda: decode_analysis_mix(Path(acquired.path), mixture),
                sink=event_sink, token=cancel_token,
            )
            cache.complete("decode", decode_sig, path=cache.stored_path(mixture))

        result = _run_stage(
            cache, "separate", separate_sig,
            lambda: separate_analysis(
                mixture, project, config.device, event_sink, cancel_token
            ),
            sink=event_sink, token=cancel_token,
        )
        stems = result.stems
        cache.complete(
            "separate", separate_sig,
            stems={name: cache.stored_path(path) for name, path in stems.items()},
        )

        manifest = {
            "status": "stems_ready",
            "source": identity,
            "source_sha256": _sha256(Path(acquired.path)),
            "decode": {"version": DECODE_VERSION, **DECODE_PARAMETERS},
            "demucs_version": _demucs_version(),
            "models": SEPARATION_PARAMETERS["models"],
            "formal_model": "htdemucs_6s",
            "device": config.device,
            "stems": {
                name: {
                    "path": path.relative_to(project.root).as_posix(),
                    "sha256": _sha256(path),
                }
                for name, path in stems.items()
            },
        }
        _atomic_json(manifest_path, manifest)
        if not config.keep_work:
            _cleanup(project, event_sink)
        _event(event_sink, "pipeline_completed", message="Stems ready")
        return Audio2ScoreSummary(project.root, manifest_path, stems)
    except PipelineCancelled:
        _event(event_sink, "pipeline_cancelled", message="Completed stages preserved")
        raise
    except Exception as exc:
        _event(event_sink, "pipeline_failed", message=str(exc))
        raise
