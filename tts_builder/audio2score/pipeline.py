from __future__ import annotations

import hashlib
import importlib.metadata
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from ..cache import source_identity
from ..events import CancellationToken, EventSink, PipelineCancelled, PipelineEvent, emit
from ..media import acquire_source
from ..models import AcquiredSource
from ..gui.settings import AppSettings
from .cache import ScoreCache, SourceMismatchError
from .config import Audio2ScoreConfig
from .drums import transcribe_drums
from .models import Audio2ScoreProject, Audio2ScoreSummary
from .runtime import require_audio2score_runtime
from .separation import (MODEL_FILES, EXPECTED_STEMS, OPTIONAL_STEMS,
                         decode_analysis_mix, separate_analysis, separate_vocal_harmony,
                         valid_stems)
from .timeline import analyze_timeline
from .transcription import BasicPitchONNX


DECODE_VERSION = 1
DECODE_PARAMETERS = {"sample_rate": 44100, "channels": 2, "codec": "pcm_f32le"}
SEPARATION_PARAMETERS = {
    "formal_model": "bs-roformer-sw+dereverb",
    "version": 3,
}
TIMELINE_PARAMETERS = {"version": 1, "source": "formal_drums"}
DRUM_PARAMETERS = {"version": 1, "bands": ["low", "mid", "high"]}
MELODIC_ROUTES = {
    "bass": ("bass", (28, 67)),
    "guitar": ("guitar", (40, 88)),
    "piano": ("piano", (21, 108)),
}


@dataclass(frozen=True)
class _AudioContext:
    project: Audio2ScoreProject
    cache: ScoreCache
    identity: dict
    source_sig: str
    decode_sig: str
    separate_sig: str
    manifest_path: Path
    stems: dict[str, Path]
    mixture: Path
    model_root: Path


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


def _formal_stems(project_root: Path) -> dict[str, Path]:
    return {name: project_root / "stems" / "base" / f"{name}.wav" for name in EXPECTED_STEMS}


def _available_stems(context: _AudioContext) -> dict[str, Path]:
    stems = dict(context.stems)
    manifest = _load_json(context.manifest_path)
    optional = manifest.get("optional_vocal_stems", {})
    if all(name in optional for name in OPTIONAL_STEMS):
        paths = {
            name: context.project.root / optional[name]["path"]
            for name in OPTIONAL_STEMS
        }
        if valid_stems({**stems, **paths}):
            stems.update(paths)
    return stems


def _model_assets_identity(model_root: Path, roles=("six_stem", "dereverb")) -> dict:
    result = {}
    for role in roles:
        folder, checkpoint, config = MODEL_FILES[role]
        files = []
        for name in (checkpoint, config):
            path = model_root / "bs-roformer" / folder / name
            try:
                stat = path.stat()
                files.append((str(path), stat.st_size, stat.st_mtime_ns))
            except FileNotFoundError:
                files.append((str(path), "missing"))
        result[role] = files
    return result


def _signatures(cache: ScoreCache, identity: dict, device: str,
                model_root: Path) -> tuple[str, str, str]:
    source_sig = cache.signature({"source": identity})
    decode_sig = cache.signature({
        "parent": source_sig, "version": DECODE_VERSION, **DECODE_PARAMETERS,
    })
    separate_sig = cache.signature({
        "parent": decode_sig, "device": device,
        "model_assets": _model_assets_identity(model_root), **SEPARATION_PARAMETERS,
    })
    return source_sig, decode_sig, separate_sig


def _formal_hit(cache: ScoreCache, signature: str, manifest: Path,
                stems: dict[str, Path]) -> bool:
    return (
        cache.stage_hit("separate", signature)
        and manifest.is_file()
        and valid_stems(stems)
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
        project.work / "harmony",
        project.work / "transcription",
    ):
        if not path.exists():
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        removed.append(path.name)
    _event(sink, "cleanup_completed", "cleanup", ", ".join(removed))


def _context(source: str, config: Audio2ScoreConfig,
             use_bound_identity: bool = False) -> _AudioContext:
    project = config.project()
    cache = ScoreCache(project, source, fresh=config.fresh)
    provided = source_identity(source)
    if use_bound_identity:
        identity = cache.state.get("source")
        if not identity:
            raise FileNotFoundError(f"six-stem project state missing: {cache.state_path}")
        if any(provided.get(key) != identity.get(key)
               for key in ("kind", "path", "url", "bvid")):
            raise SourceMismatchError("project is already bound to a different source")
    else:
        identity = provided
        cache.bind_source(identity)
    model_root = (config.model_root or AppSettings.load().model_root).expanduser().resolve()
    source_sig, decode_sig, separate_sig = _signatures(cache, identity, config.device, model_root)
    return _AudioContext(
        project=project,
        cache=cache,
        identity=identity,
        source_sig=source_sig,
        decode_sig=decode_sig,
        separate_sig=separate_sig,
        manifest_path=project.root / "manifest.json",
        stems=_formal_stems(project.root),
        mixture=project.work / "mixture.wav",
        model_root=model_root,
    )


def _prepare_audio(source: str, config: Audio2ScoreConfig, context: _AudioContext,
                   require_mix: bool, event_sink: EventSink | None,
                   cancel_token: CancellationToken | None) -> Path | None:
    cache = context.cache
    formal_ready = _formal_hit(
        cache, context.separate_sig, context.manifest_path, context.stems
    )
    if formal_ready and (not require_mix or context.mixture.is_file()):
        _event(event_sink, "stage_cache_hit", "separate", "Formal stems")
        return None

    acquired = _load_acquired(cache, context.source_sig)
    if acquired is None:
        acquired = _run_stage(
            cache, "source", context.source_sig,
            lambda: acquire_source(
                source, context.project.work / "source",
                event_sink=event_sink, cancel_token=cancel_token,
            ),
            sink=event_sink, token=cancel_token,
        )
        cache.complete(
            "source", context.source_sig,
            path=cache.stored_path(Path(acquired.path)),
            label=acquired.label, source_id=acquired.source_id,
            original=acquired.original,
        )
    else:
        _event(event_sink, "stage_cache_hit", "source", "Cached source")

    if cache.stage_hit("decode", context.decode_sig, context.mixture):
        _event(event_sink, "stage_cache_hit", "decode", "Cached mixture")
    else:
        _run_stage(
            cache, "decode", context.decode_sig,
            lambda: decode_analysis_mix(Path(acquired.path), context.mixture),
            sink=event_sink, token=cancel_token,
        )
        cache.complete(
            "decode", context.decode_sig,
            path=cache.stored_path(context.mixture),
        )

    if formal_ready:
        _event(event_sink, "stage_cache_hit", "separate", "Formal stems")
        return None

    result = _run_stage(
        cache, "separate", context.separate_sig,
        lambda: separate_analysis(
            context.mixture, context.project, config.device,
            event_sink, cancel_token, model_root=context.model_root,
        ),
        sink=event_sink, token=cancel_token,
    )
    cache.complete(
        "separate", context.separate_sig,
        stems={
            name: cache.stored_path(path) for name, path in result.stems.items()
        },
    )
    manifest = {
        "status": "stems_ready",
        "source": context.identity,
        "decode": {"version": DECODE_VERSION, **DECODE_PARAMETERS},
        "models": {role: MODEL_FILES[role] for role in ("six_stem", "dereverb")},
        "formal_model": SEPARATION_PARAMETERS["formal_model"],
        "device": config.device,
        "stems": {
            name: {
                "path": path.relative_to(context.project.root).as_posix(),
            }
            for name, path in result.stems.items()
        },
        "raw_vocals": ".work/vocals_raw.wav",
    }
    _atomic_json(context.manifest_path, manifest)
    return None


def process_to_stems(source: str, config: Audio2ScoreConfig,
                     event_sink: EventSink | None = None,
                     cancel_token: CancellationToken | None = None) -> Audio2ScoreSummary:
    _event(event_sink, "pipeline_started", message=source)
    try:
        context = _context(source, config)
        _prepare_audio(
            source, config, context, False, event_sink, cancel_token
        )
        if not config.keep_work:
            _cleanup(context.project, event_sink)
        _event(event_sink, "pipeline_completed", message="Stems ready")
        return Audio2ScoreSummary(
            context.project.root, context.manifest_path, _available_stems(context)
        )
    except PipelineCancelled:
        _event(event_sink, "pipeline_cancelled", message="Completed stages preserved")
        raise
    except Exception as exc:
        _event(event_sink, "pipeline_failed", message=str(exc))
        raise


def process_to_vocal_harmony(source: str, config: Audio2ScoreConfig,
                             event_sink: EventSink | None = None,
                             cancel_token: CancellationToken | None = None) -> Audio2ScoreSummary:
    if config.fresh:
        raise ValueError("vocal harmony requires an existing six-stem project")
    context = _context(source, config, use_bound_identity=True)
    raw_vocals = context.project.work / "vocals_raw.wav"
    if not _formal_hit(context.cache, context.separate_sig,
                       context.manifest_path, context.stems):
        raise FileNotFoundError(f"six formal stems missing: {context.project.stems}")
    if not raw_vocals.is_file():
        raise FileNotFoundError(f"raw vocals missing: {raw_vocals}")
    optional = {
        name: context.project.stems / "harmony" / f"{name}.wav"
        for name in OPTIONAL_STEMS
    }
    signature = context.cache.signature({
        "parent": context.separate_sig,
        "raw_vocals_sha256": _sha256(raw_vocals),
        "device": config.device,
        "model_assets": _model_assets_identity(context.model_root, ("karaoke", "dereverb")),
        "version": 1,
    })
    if not (context.cache.stage_hit("vocal_harmony", signature)
            and valid_stems({**context.stems, **optional})):
        result = _run_stage(
            context.cache, "vocal_harmony", signature,
            lambda: separate_vocal_harmony(
                context.project, config.device, event_sink, cancel_token,
                model_root=context.model_root,
            ),
            sink=event_sink, token=cancel_token,
        )
        manifest = _load_json(context.manifest_path)
        manifest["optional_vocal_stems"] = {
            name: {"path": path.relative_to(context.project.root).as_posix()}
            for name, path in result.items()
        }
        _atomic_json(context.manifest_path, manifest)
        context.cache.complete(
            "vocal_harmony", signature,
            stems={name: context.cache.stored_path(path) for name, path in result.items()},
        )
    if not config.keep_work:
        _cleanup(context.project, event_sink)
    return Audio2ScoreSummary(
        context.project.root, context.manifest_path, {**context.stems, **optional}
    )


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _transcribe(source: str, config: Audio2ScoreConfig,
                             event_sink: EventSink | None = None,
                             cancel_token: CancellationToken | None = None,
                             finish: bool = True
                             ) -> Audio2ScoreSummary:
    require_audio2score_runtime()
    _event(event_sink, "pipeline_started", message=source)
    try:
        context = _context(source, config, use_bound_identity=(
            not config.fresh and not Path(source).is_file()
            and (config.project().work / "state.json").is_file()
        ))
        cache = context.cache
        adapter = BasicPitchONNX()
        librosa_version = importlib.metadata.version("librosa")
        transcription_sig = cache.signature({
            "parent": context.separate_sig,
            "librosa": librosa_version,
            "timeline": TIMELINE_PARAMETERS,
            "basic_pitch": adapter.parameters,
            "drums": DRUM_PARAMETERS,
        })
        timeline_path = context.project.reports / "timeline.json"
        transcription_path = context.project.reports / "transcription.json"
        if (
            cache.stage_hit("transcription", transcription_sig, transcription_path)
            and timeline_path.is_file()
            and context.manifest_path.is_file()
            and all(path.is_file() for path in context.stems.values())
        ):
            _event(
                event_sink, "stage_cache_hit", "transcribe",
                "Transcription report",
            )
            if finish:
                _event(event_sink, "pipeline_completed", message="Transcription ready")
            return Audio2ScoreSummary(
                context.project.root, context.manifest_path, _available_stems(context),
                timeline_path, transcription_path,
            )

        _prepare_audio(
            source, config, context, False, event_sink, cancel_token
        )

        timeline_sig = cache.signature({
            "parent": context.separate_sig,
            "librosa": librosa_version,
            **TIMELINE_PARAMETERS,
        })
        if cache.stage_hit("timeline", timeline_sig, timeline_path):
            timeline = _load_json(timeline_path)
            _event(event_sink, "stage_cache_hit", "timeline", "Cached timeline")
        else:
            timeline = _run_stage(
                cache,
                "timeline",
                timeline_sig,
                lambda: analyze_timeline(
                    context.mixture if context.mixture.is_file() else context.stems["drums"],
                    context.stems["drums"],
                ),
                sink=event_sink,
                token=cancel_token,
            )
            _atomic_json(timeline_path, timeline)
            cache.complete("timeline", timeline_sig, path=cache.stored_path(timeline_path))

        route_dir = context.project.work / "transcription"
        route_sources = {
            "bass": context.stems["bass"],
            "guitar": context.stems["guitar"],
            "piano": context.stems["piano"],
        }
        routes = {}
        source_hashes = {}
        for current, (route, (_, playable_range)) in enumerate(
            MELODIC_ROUTES.items(), start=1
        ):
            audio_path = route_sources[route]
            if not audio_path.is_file():
                raise FileNotFoundError(f"transcription source missing: {audio_path}")
            source_hash = _sha256(audio_path)
            source_hashes[route] = source_hash
            route_path = route_dir / f"{route}.json"
            route_sig = cache.signature({
                "parent": context.separate_sig,
                "source_sha256": source_hash,
                "parameters": adapter.parameters,
                "playable_range": playable_range,
            })
            stage = f"transcribe_{route}"
            if cache.stage_hit(stage, route_sig, route_path):
                evidence = _load_json(route_path)
                _event(event_sink, "stage_cache_hit", "transcribe", route)
            else:
                evidence = _run_stage(
                    cache,
                    stage,
                    route_sig,
                    lambda p=audio_path, r=route, limits=playable_range:
                        adapter.transcribe(p, r, limits),
                    sink=event_sink,
                    token=cancel_token,
                )
                _atomic_json(route_path, evidence)
                cache.complete(stage, route_sig, path=cache.stored_path(route_path))
            routes[route] = evidence
            _event(event_sink, "stage_progress", "transcribe", route, current, 4)

        drum_source = context.stems["drums"]
        drum_hash = _sha256(drum_source)
        source_hashes["drums"] = drum_hash
        drum_path = route_dir / "drums.json"
        drum_sig = cache.signature({
            "parent": context.separate_sig,
            "source_sha256": drum_hash,
            **DRUM_PARAMETERS,
        })
        if cache.stage_hit("transcribe_drums", drum_sig, drum_path):
            drums = _load_json(drum_path)
            _event(event_sink, "stage_cache_hit", "transcribe", "drums")
        else:
            drums = _run_stage(
                cache,
                "transcribe_drums",
                drum_sig,
                lambda: transcribe_drums(drum_source),
                sink=event_sink,
                token=cancel_token,
            )
            _atomic_json(drum_path, drums)
            cache.complete(
                "transcribe_drums", drum_sig, path=cache.stored_path(drum_path)
            )
        _event(event_sink, "stage_progress", "transcribe", "drums", 4, 4)

        report = {
            "version": 1,
            "status": "candidates_ready",
            "source_hashes": source_hashes,
            "routes": routes,
            "drums": drums,
            "quantized": False,
            "musical_accuracy_verified": False,
        }
        _atomic_json(transcription_path, report)
        cache.complete(
            "transcription",
            transcription_sig,
            path=cache.stored_path(transcription_path),
        )
        if finish and not config.keep_work:
            _cleanup(context.project, event_sink)
        if finish:
            _event(event_sink, "pipeline_completed", message="Transcription ready")
        return Audio2ScoreSummary(
            context.project.root, context.manifest_path, _available_stems(context),
            timeline_path, transcription_path,
        )
    except PipelineCancelled:
        _event(event_sink, "pipeline_cancelled", message="Completed stages preserved")
        raise
    except Exception as exc:
        _event(event_sink, "pipeline_failed", message=str(exc))
        raise


def process_to_transcription(source: str, config: Audio2ScoreConfig,
                             event_sink: EventSink | None = None,
                             cancel_token: CancellationToken | None = None) -> Audio2ScoreSummary:
    return _transcribe(source, config, event_sink, cancel_token)


def process_to_score(source: str, config: Audio2ScoreConfig,
                     event_sink: EventSink | None = None,
                     cancel_token: CancellationToken | None = None) -> Audio2ScoreSummary:
    from .score_pipeline import build_score_outputs
    import importlib.util
    missing = [name for name in ("guitarpro", "mido") if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError("Install requirements-audio2score.txt; missing " + ", ".join(missing))
    from .exporters import find_musescore
    if find_musescore(config.musescore) is None:
        raise RuntimeError("MuseScore Studio not found. Set its executable path in Settings, --musescore or MUSESCORE_BIN.")

    summary = _transcribe(source, config, event_sink, cancel_token, finish=False)
    result = build_score_outputs(summary, config, event_sink, cancel_token)
    if result.status == "completed" and not config.keep_work:
        _cleanup(config.project(), event_sink)
    _event(event_sink, "pipeline_completed" if result.status == "completed" else "pipeline_partial",
           message=result.message or "Score draft ready")
    return result
