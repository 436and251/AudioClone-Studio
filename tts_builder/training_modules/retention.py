from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import shutil


FAILURE_RETENTION = timedelta(hours=48)
MAX_JOB_BYTES = 1024 * 1024
MAX_JOURNAL_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class CleanupSummary:
    removed: int = 0
    retained_failures: int = 0
    uncertain: int = 0


@dataclass(frozen=True, slots=True)
class _Job:
    directory: Path
    project_name: str
    state: str
    terminal_at: datetime | None = None


def cleanup_jobs(
    project_root: Path,
    *,
    now: datetime | None = None,
    active_job: Path | None = None,
) -> CleanupSummary:
    root = Path(project_root)
    if not root.is_absolute() or not root.is_dir():
        return CleanupSummary(uncertain=1)
    root = root.resolve()
    jobs_root = root / "jobs"
    if not jobs_root.is_dir() or jobs_root.resolve().parent != root:
        return CleanupSummary()
    active = _active_directory(active_job)
    records: list[_Job] = []
    uncertain = 0
    for entry in jobs_root.iterdir():
        if not entry.is_dir():
            continue
        if active is not None and entry.resolve() == active:
            continue
        record = _load_job(entry, root)
        if record is None:
            uncertain += 1
        else:
            records.append(record)

    clock = now or datetime.now(timezone.utc)
    if clock.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    retained_failures: set[Path] = set()
    failures: dict[str, list[_Job]] = {}
    for record in records:
        if record.state == "failed":
            failures.setdefault(record.project_name, []).append(record)
    cutoff = clock - FAILURE_RETENTION
    for group in failures.values():
        newest = max(group, key=lambda item: _failure_time(item))
        if _failure_time(newest) >= cutoff:
            retained_failures.add(newest.directory)

    removed = 0
    for record in records:
        if record.state in {"incomplete", "pending"}:
            continue
        if record.state == "failed" and record.directory in retained_failures:
            continue
        if _remove_job(record.directory, jobs_root):
            removed += 1
        else:
            uncertain += 1
    return CleanupSummary(removed, len(retained_failures), uncertain)


def _load_job(directory: Path, root: Path) -> _Job | None:
    try:
        if _link_like(directory):
            return None
        directory = directory.resolve()
        if directory.parent != (root / "jobs").resolve():
            return None
        job_path = directory / "job.json"
        if not job_path.is_file() or job_path.stat().st_size > MAX_JOB_BYTES:
            return None
        payload = json.loads(job_path.read_text(encoding="utf-8", errors="strict"))
        if not isinstance(payload, dict):
            return None
        project_name = payload.get("project_name")
        stages = payload.get("stages")
        if (
            payload.get("protocol_version") != 2
            or payload.get("job_id") != directory.name
            or not isinstance(project_name, str)
            or not project_name
            or not _same_path(payload.get("project_root"), root)
            or not _same_path(payload.get("job_dir"), directory)
            or not isinstance(stages, list)
            or any(not isinstance(stage, str) for stage in stages)
        ):
            return None
        events = _read_events(directory / "events.jsonl", directory.name)
        if events is None:
            return None
        state, terminal_at = _classify(directory.name, stages, events)
        return _Job(directory, project_name, state, terminal_at)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        return None


def _read_events(path: Path, job_id: str) -> list[tuple[str, datetime | None]] | None:
    if not path.exists():
        return []
    if not path.is_file() or path.stat().st_size > MAX_JOURNAL_BYTES:
        return None
    events: list[tuple[str, datetime | None]] = []
    for line in path.read_text(encoding="utf-8", errors="strict").splitlines():
        payload = json.loads(line)
        if (
            not isinstance(payload, dict)
            or payload.get("protocol_version") != 1
            or payload.get("job_id") != job_id
            or not isinstance(payload.get("type"), str)
        ):
            return None
        events.append((payload["type"], _timestamp(payload.get("timestamp"))))
    return events


def _classify(
    job_id: str,
    stages: list[str],
    events: list[tuple[str, datetime | None]],
) -> tuple[str, datetime | None]:
    if job_id.startswith("inference-"):
        terminal = _latest(events, {"inference_started", "inference_failed", "inference_completed"})
        if terminal is None or terminal[0] == "inference_started":
            return "incomplete", None
        return ("failed" if terminal[0] == "inference_failed" else "delete", terminal[1])

    job_terminal = _latest(events, {"job_failed", "job_cancelled", "job_completed"})
    promotion = _latest(events, {"promotion_completed"})
    if promotion is not None and (
        job_terminal is None or promotion[2] > job_terminal[2]
    ):
        return "delete", promotion[1]
    if job_terminal is None:
        return "incomplete", None
    kind, timestamp, _ = job_terminal
    if kind == "job_failed":
        return "failed", timestamp
    if kind == "job_cancelled":
        return "delete", timestamp
    return ("pending" if "evaluate" in stages else "delete", timestamp)


def _latest(
    events: list[tuple[str, datetime | None]], kinds: set[str]
) -> tuple[str, datetime | None, int] | None:
    found = [
        (kind, timestamp, index)
        for index, (kind, timestamp) in enumerate(events)
        if kind in kinds
    ]
    return found[-1] if found else None


def _failure_time(job: _Job) -> datetime:
    if job.terminal_at is not None:
        return job.terminal_at
    return datetime.fromtimestamp(job.directory.stat().st_mtime, timezone.utc)


def _timestamp(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def _active_directory(value: Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.name == "job.json":
        path = path.parent
    try:
        return path.resolve()
    except OSError:
        return None


def _same_path(value: object, expected: Path) -> bool:
    if not isinstance(value, str) or not Path(value).is_absolute():
        return False
    return Path(value).resolve() == expected.resolve()


def _link_like(path: Path) -> bool:
    return path.is_symlink() or path.is_junction()


def _remove_job(directory: Path, jobs_root: Path) -> bool:
    try:
        if _link_like(directory):
            return False
        resolved = directory.resolve()
        if resolved.parent != jobs_root.resolve():
            return False
        shutil.rmtree(resolved)
        return True
    except OSError:
        return False


__all__ = ["CleanupSummary", "cleanup_jobs"]
