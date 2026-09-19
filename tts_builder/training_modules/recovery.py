from __future__ import annotations

import heapq
import json
from dataclasses import dataclass
from pathlib import Path

from .event_store import EventStore
from .models import ModuleEvent


MAX_JOBS = 30
MAX_JOB_BYTES = 1024 * 1024
MAX_JOURNAL_BYTES = 16 * 1024 * 1024
_PIPELINE_STAGES = ("preprocess", "s2", "s1", "evaluate")


@dataclass(frozen=True)
class FailedJob:
    job_path: Path
    stage: str


@dataclass(frozen=True)
class ModelHistoryItem:
    model: Path
    latest_audio: Path | None


_BUNDLE_FILES = (
    "model.yaml",
    "metadata.json",
    "weights/s1.ckpt",
    "weights/s2.pth",
    "reference/default.json",
    "reference/default.wav",
)


def local_model_history(
    project_root: Path, module_id: str, framework: str
) -> tuple[ModelHistoryItem, ...]:
    raw_root = Path(project_root)
    if not raw_root.is_absolute() or not raw_root.is_dir():
        return ()
    root = raw_root.resolve()
    models_root = (root / "models").resolve()
    if models_root.parent != root or not models_root.is_dir():
        return ()
    try:
        models = [
            path.resolve()
            for path in models_root.iterdir()
            if path.is_dir() and _complete_local_bundle(path, models_root)
        ]
        models.sort(key=lambda path: path.stat().st_mtime_ns, reverse=True)
    except OSError:
        return ()
    return tuple(
        ModelHistoryItem(model, _latest_audio(root, model.name)) for model in models
    )


def latest_promoted_model(
    project_root: Path,
    module_id: str,
    framework: str,
    project_name: str,
) -> Path | None:
    raw_root = Path(project_root)
    if (
        not raw_root.is_absolute()
        or not raw_root.is_dir()
        or not all(isinstance(value, str) and value for value in (
            module_id, framework, project_name
        ))
    ):
        return None
    root = raw_root.resolve()
    for _, directory, job, journal in _candidate_jobs(root):
        job_id = _matching_job(
            job, directory, root, module_id, framework, project_name
        )
        if job_id is None:
            continue
        model = _completed_model(journal, root, job_id)
        if model is not None:
            return model
    relative = Path(project_name)
    if relative.name != project_name or project_name in {".", ".."}:
        return None
    models_root = (root / "models").resolve()
    model = (models_root / project_name).resolve()
    return model if _complete_local_bundle(model, models_root) else None


def latest_failed_job(
    project_root: Path,
    module_id: str,
    framework: str,
    project_name: str,
) -> FailedJob | None:
    raw_root = Path(project_root)
    if (
        not raw_root.is_absolute()
        or not raw_root.is_dir()
        or not all(
            isinstance(value, str) and value
            for value in (module_id, framework, project_name)
        )
    ):
        return None
    root = raw_root.resolve()
    for _, directory, job, _ in _candidate_jobs(root):
        if _matching_job(
            job, directory, root, module_id, framework, project_name
        ) is None:
            continue
        failed = _failed_job(job, directory, root, project_name)
        if failed is not None:
            return failed
    return None


def _candidate_jobs(root: Path) -> list[tuple[int, Path, Path, Path]]:
    jobs = root / "jobs"
    if not jobs.is_dir() or jobs.resolve().parent != root:
        return []
    candidates = []
    try:
        entries = jobs.iterdir()
        for entry in entries:
            try:
                directory = entry.resolve()
                if not entry.is_dir() or directory.parent != jobs.resolve():
                    continue
                job = directory / "job.json"
                journal = directory / "events.jsonl"
                if (
                    not job.is_file()
                    or not journal.is_file()
                    or job.resolve().parent != directory
                    or journal.resolve().parent != directory
                ):
                    continue
                modified = max(job.stat().st_mtime_ns, journal.stat().st_mtime_ns)
                candidates.append((modified, directory, job.resolve(), journal.resolve()))
            except OSError:
                continue
    except OSError:
        return []
    return heapq.nlargest(MAX_JOBS, candidates, key=lambda item: item[0])


def _matching_job(
    job: Path,
    directory: Path,
    root: Path,
    module_id: str,
    framework: str,
    project_name: str,
) -> str | None:
    try:
        if job.stat().st_size > MAX_JOB_BYTES:
            return None
        payload = json.loads(job.read_text(encoding="utf-8", errors="strict"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    job_id = payload.get("job_id")
    expected = {
        "protocol_version": 2,
        "module_id": module_id,
        "framework": framework,
        "project_name": project_name,
    }
    if (
        not isinstance(job_id, str)
        or not job_id
        or job_id != directory.name
        or any(payload.get(key) != value for key, value in expected.items())
        or not _same_path(payload.get("project_root"), root)
        or not _same_path(payload.get("job_dir"), directory)
    ):
        return None
    return job_id


def _job_project_name(
    job: Path,
    directory: Path,
    root: Path,
    module_id: str,
    framework: str,
) -> str | None:
    try:
        if job.stat().st_size > MAX_JOB_BYTES:
            return None
        payload = json.loads(job.read_text(encoding="utf-8", errors="strict"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    project_name = payload.get("project_name")
    if not isinstance(project_name, str) or not project_name:
        return None
    return project_name if _matching_job(
        job, directory, root, module_id, framework, project_name
    ) is not None else None


def _complete_local_bundle(model: Path, models_root: Path) -> bool:
    try:
        model = model.resolve()
        return (
            model.parent == models_root
            and all((model / relative).is_file() for relative in _BUNDLE_FILES)
        )
    except OSError:
        return False


def _latest_audio(root: Path, project_name: str) -> Path | None:
    directory = (root / "outputs" / project_name / "gui").resolve()
    if not directory.is_dir() or not directory.is_relative_to(root):
        return None
    try:
        candidates = (
            path.resolve() for path in directory.glob("*.wav") if path.is_file()
        )
        return max(candidates, key=lambda path: path.stat().st_mtime_ns, default=None)
    except OSError:
        return None


def _same_path(value: object, expected: Path) -> bool:
    if not isinstance(value, str) or not Path(value).is_absolute():
        return False
    try:
        return Path(value).resolve() == expected
    except OSError:
        return False


def _failed_job(
    job: Path, directory: Path, root: Path, project_name: str
) -> FailedJob | None:
    try:
        payload = json.loads(job.read_text(encoding="utf-8", errors="strict"))
        output_root = Path(payload["output_root"])
        stages = payload["stages"]
        if (
            not output_root.is_absolute()
            or not isinstance(stages, list)
            or not stages
            or any(stage not in _PIPELINE_STAGES for stage in stages)
        ):
            return None
        output_root = output_root.resolve()
        if output_root != root and not output_root.is_relative_to(root):
            return None
        run_dir = (output_root / project_name).resolve()
        if not run_dir.is_relative_to(root):
            return None
        state_path = run_dir / "pipeline-state.json"
        if state_path.stat().st_size > MAX_JOB_BYTES:
            return None
        state = json.loads(state_path.read_text(encoding="utf-8", errors="strict"))
    except (KeyError, OSError, UnicodeError, json.JSONDecodeError, TypeError):
        return None
    if not isinstance(state, dict) or state.get("schema_version") != 1:
        return None
    identity = state.get("identity")
    statuses = state.get("stages")
    failure = state.get("failure")
    if (
        not isinstance(identity, dict)
        or identity.get("stages") != stages
        or not _same_path(identity.get("pipeline"), directory / "module" / "pipeline.yaml")
        or not _same_path(identity.get("config"), directory / "module" / "train.yaml")
        or not isinstance(statuses, dict)
        or list(statuses) != stages
        or not isinstance(failure, dict)
    ):
        return None
    stage = failure.get("stage")
    if not isinstance(stage, str) or statuses.get(stage) != "failed":
        return None
    return FailedJob(job, stage)


def _completed_model(journal: Path, root: Path, job_id: str) -> Path | None:
    try:
        size = journal.stat().st_size
    except OSError:
        return None
    if size > MAX_JOURNAL_BYTES:
        return None
    store = EventStore(journal, root, expected_job_id=job_id)
    pending: Path | None = None
    completed: list[Path] = []
    while store.offset < size:
        before = store.offset
        batch = store.read_new()
        for event in batch.events:
            if event.type == "artifact":
                pending = _promoted_artifact(event)
            elif event.type == "promotion_completed" and pending is not None:
                completed.append(pending)
                pending = None
            elif event.type == "promotion_failed":
                pending = None
        if store.offset == before:
            break
    return next((model for model in reversed(completed) if model.is_dir()), None)


def _promoted_artifact(event: ModuleEvent) -> Path | None:
    models = [
        Path(str(artifact["path"]))
        for artifact in event.artifacts
        if artifact.get("type") == "promoted_model"
    ]
    return models[-1] if models else None
