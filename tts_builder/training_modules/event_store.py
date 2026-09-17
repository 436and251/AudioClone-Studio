from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

from .models import ModuleEvent


MAX_LINE_BYTES = 1024 * 1024
_READ_BYTES = 64 * 1024
_FIELDS = {
    "protocol_version", "job_id", "type", "timestamp", "stage", "message_key",
    "message_args", "current", "total", "artifacts",
}


@dataclass(frozen=True)
class EventBatch:
    events: tuple[ModuleEvent, ...]
    errors: tuple[str, ...]


class EventStore:
    def __init__(
        self,
        journal: Path,
        allowed_root: Path,
        *,
        expected_job_id: str | None = None,
    ) -> None:
        self.journal = Path(journal)
        self.allowed_root = Path(allowed_root).resolve()
        self.expected_job_id = expected_job_id
        self.offset = 0
        self._buffer = bytearray()
        self._discarding_oversized_line = False

    def read_new(self) -> EventBatch:
        if not self.journal.exists():
            return EventBatch((), ())
        if self.journal.stat().st_size < self.offset:
            self.offset = 0
            self._buffer.clear()
            self._discarding_oversized_line = False

        events: list[ModuleEvent] = []
        errors: list[str] = []
        with self.journal.open("rb") as stream:
            stream.seek(self.offset)
            while chunk := stream.read(_READ_BYTES):
                self.offset += len(chunk)
                self._consume(chunk, events, errors)
        return EventBatch(tuple(events), tuple(errors))

    def _consume(
        self,
        chunk: bytes,
        events: list[ModuleEvent],
        errors: list[str],
    ) -> None:
        if self._discarding_oversized_line:
            newline = chunk.find(b"\n")
            if newline < 0:
                return
            self._discarding_oversized_line = False
            chunk = chunk[newline + 1 :]
        self._buffer.extend(chunk)
        while (newline := self._buffer.find(b"\n")) >= 0:
            line = bytes(self._buffer[:newline])
            del self._buffer[: newline + 1]
            if len(line) + 1 > MAX_LINE_BYTES:
                errors.append(f"event line exceeds maximum {MAX_LINE_BYTES} bytes")
            elif line:
                self._parse_line(line, events, errors)
        if len(self._buffer) > MAX_LINE_BYTES:
            self._buffer.clear()
            self._discarding_oversized_line = True
            errors.append(f"event line exceeds maximum {MAX_LINE_BYTES} bytes")

    def _parse_line(
        self,
        line: bytes,
        events: list[ModuleEvent],
        errors: list[str],
    ) -> None:
        try:
            payload = json.loads(line.decode("utf-8"))
            events.append(_parse_event(payload, self.allowed_root, self.expected_job_id))
        except (UnicodeError, json.JSONDecodeError, TypeError, ValueError) as error:
            errors.append(f"invalid event JSON: {error}")


def _parse_event(payload: object, root: Path, expected_job_id: str | None) -> ModuleEvent:
    if not isinstance(payload, dict) or set(payload) - _FIELDS:
        raise ValueError("event must be an object with supported fields")
    if payload.get("protocol_version") != 1 or type(payload.get("protocol_version")) is not int:
        raise ValueError("protocol_version must be integer 1")
    job_id = _text(payload.get("job_id"), "job_id")
    if expected_job_id is not None and job_id != expected_job_id:
        raise ValueError("job_id does not match the active job")
    message_args = payload.get("message_args")
    if message_args is not None and (
        not isinstance(message_args, Mapping)
        or any(not isinstance(key, str) for key in message_args)
    ):
        raise ValueError("message_args must be an object with string keys")
    return ModuleEvent(
        1,
        job_id,
        _text(payload.get("type"), "type"),
        _text(payload.get("timestamp"), "timestamp"),
        _optional_text(payload.get("stage"), "stage"),
        _optional_text(payload.get("message_key"), "message_key"),
        dict(message_args) if message_args is not None else None,
        _progress(payload.get("current"), "current"),
        _progress(payload.get("total"), "total"),
        _artifacts(payload.get("artifacts", []), root),
    )


def _artifacts(value: object, root: Path) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, list):
        raise ValueError("artifacts must be an array")
    normalized = []
    for artifact in value:
        if not isinstance(artifact, dict) or any(not isinstance(key, str) for key in artifact):
            raise ValueError("artifact must be an object")
        artifact_type = _text(artifact.get("type"), "artifact type")
        raw_path = artifact.get("path")
        if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
            raise ValueError("artifact path must be absolute")
        path = Path(raw_path).resolve()
        if path != root and not path.is_relative_to(root):
            raise ValueError("artifact path is outside the project")
        normalized.append({**artifact, "type": artifact_type, "path": str(path)})
    return tuple(normalized)


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _optional_text(value: object, name: str) -> str | None:
    return None if value is None else _text(value, name)


def _progress(value: object, name: str) -> int | float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError(f"{name} must be a non-negative number")
    return value
