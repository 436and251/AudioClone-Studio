from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4


LANGUAGES = {"zh", "ja", "en", "mixed"}


def ensure_inference_job(
    project_root: Path,
    module_id: str,
    framework: str,
    promoted_model: Path,
) -> Path:
    project = _absolute_directory(project_root, "project_root")
    model = _validated_model(promoted_model, project)
    job_id = f"inference-{model.name}"
    job_dir = project / "jobs" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    context = job_dir / "context.list"
    context.touch(exist_ok=True)
    job = job_dir / "job.json"
    payload = {
        "protocol_version": 2,
        "job_id": job_id,
        "module_id": module_id,
        "project_name": model.name,
        "project_root": str(project),
        "output_root": str((project / "outputs").resolve()),
        "training_data": {"path": str(context.resolve()), "kind": "file"},
        "framework": framework,
        "stages": ["evaluate"],
        "device": "cuda:0",
        "precision": "fp16",
        "parameters": {},
        "reference": None,
        "job_dir": str(job_dir.resolve()),
    }
    now = datetime.now(timezone.utc).isoformat()
    events = (
        {
            "protocol_version": 1,
            "job_id": job_id,
            "type": "artifact",
            "timestamp": now,
            "artifacts": [{"type": "promoted_model", "path": str(model)}],
        },
        {
            "protocol_version": 1,
            "job_id": job_id,
            "type": "promotion_completed",
            "timestamp": now,
        },
    )
    _write_json(job, payload)
    _write_text(
        job_dir / "events.jsonl",
        "".join(json.dumps(event, ensure_ascii=False) + "\n" for event in events),
    )
    return job.resolve()


def build_inference_request(
    job_path: Path,
    promoted_model: Path,
    text: str | None,
    text_file: Path | None,
    language: str,
    device: str,
    now: datetime,
) -> Path:
    job = Path(job_path).resolve()
    try:
        payload = json.loads(job.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid job JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("job JSON must be an object")
    if payload.get("protocol_version") != 2:
        raise ValueError("unsupported job protocol_version")

    project = _absolute_directory(payload.get("project_root"), "project_root")
    job_dir = _absolute_directory(payload.get("job_dir"), "job_dir")
    if job != job_dir / "job.json" or not job_dir.is_relative_to(project):
        raise ValueError("job directory must remain inside project_root")
    project_name = payload.get("project_name")
    if (
        not isinstance(project_name, str)
        or not project_name.strip()
        or Path(project_name).name != project_name
        or project_name in {".", ".."}
    ):
        raise ValueError("project_name is invalid")

    model = _validated_model(promoted_model, project)
    inline = text.strip() if isinstance(text, str) else ""
    source = Path(text_file).resolve() if text_file is not None else None
    if bool(inline) == bool(source):
        raise ValueError("exactly one text source is required")
    source_text = None
    if source is not None:
        if not source.is_file() or source.suffix.lower() != ".txt":
            raise ValueError("text_file must be an existing TXT file")
        try:
            source_text = source.read_text(encoding="utf-8", errors="strict")
        except UnicodeError as error:
            raise ValueError("text_file must be valid UTF-8") from error
        if not source_text.strip():
            raise ValueError("text_file must not be empty")
    if language not in LANGUAGES:
        raise ValueError("unsupported inference language")
    if not isinstance(device, str) or not device.strip():
        raise ValueError("device must not be empty")

    return _write_request(
        job_dir,
        project,
        project_name,
        model,
        inline,
        source_text,
        language,
        device.strip(),
        now,
    )


def _validated_model(promoted_model: Path, project: Path) -> Path:
    model = Path(promoted_model).resolve()
    if not model.is_dir() or not model.is_relative_to(project):
        raise ValueError("promoted model must be a directory inside project_root")
    return model


def _write_request(
    job_dir: Path,
    project: Path,
    project_name: str,
    model: Path,
    inline: str,
    source_text: str | None,
    language: str,
    device: str,
    now: datetime,
) -> Path:
    request_id = uuid4().hex
    request_dir = job_dir / "inference" / request_id
    request = request_dir / "request.json"
    snapshot = request_dir / "input.txt" if source_text is not None else None
    payload = {
        "protocol_version": 2,
        "request_id": request_id,
        "project_root": str(project),
        "project_name": project_name,
        "model": str(model),
        "text": inline or None,
        "text_file": str(snapshot.resolve()) if snapshot is not None else None,
        "language": language,
        "device": device,
        "output": str(
            _available_output(
                project / "outputs" / project_name / "gui", job_dir, now
            ).resolve()
        ),
    }
    try:
        request_dir.mkdir(parents=True)
        if snapshot is not None:
            snapshot.write_text(source_text, encoding="utf-8")
        temporary = request_dir / ".request.json.tmp"
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(request)
    except Exception:
        for path in (request_dir / ".request.json.tmp", request, snapshot):
            if path is not None and path.exists():
                path.unlink()
        if request_dir.exists():
            request_dir.rmdir()
        raise
    return request


def _absolute_directory(value: object, field: str) -> Path:
    if not isinstance(value, (str, Path)) or not Path(value).is_absolute():
        raise ValueError(f"{field} must be absolute")
    path = Path(value).resolve()
    if not path.is_dir():
        raise ValueError(f"{field} is unavailable")
    return path


def _available_output(directory: Path, job_dir: Path, now: datetime) -> Path:
    stem = now.strftime("%Y%m%d-%H%M%S")
    reserved = set()
    for request in (job_dir / "inference").glob("*/request.json"):
        try:
            payload = json.loads(request.read_text(encoding="utf-8"))
            value = payload.get("output") if isinstance(payload, dict) else None
            if isinstance(value, str) and Path(value).is_absolute():
                reserved.add(Path(value).resolve())
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
    candidate = directory / f"{stem}.wav"
    suffix = 2
    while candidate.exists() or candidate.resolve() in reserved:
        candidate = directory / f"{stem}-{suffix}.wav"
        suffix += 1
    return candidate


def _write_json(path: Path, payload: dict) -> None:
    _write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _write_text(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
