from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class FieldDescriptor:
    key: str
    kind: str
    default: object
    constraints: Mapping[str, object]
    labels: Mapping[str, str]


@dataclass(frozen=True)
class TrainingDataDescriptor:
    kind: str
    extensions: tuple[str, ...]


@dataclass(frozen=True)
class FrameworkDescriptor:
    id: str
    display_name: str
    capabilities: tuple[str, ...]
    training_data: TrainingDataDescriptor
    fields: tuple[FieldDescriptor, ...]


@dataclass(frozen=True)
class ModuleDescriptor:
    protocol_version: int
    module_id: str
    module_version: str
    frameworks: tuple[FrameworkDescriptor, ...]


@dataclass(frozen=True)
class ModuleEvent:
    protocol_version: int
    job_id: str
    type: str
    timestamp: str
    stage: str | None = None
    message_key: str | None = None
    message_args: Mapping[str, object] | None = None
    current: int | float | None = None
    total: int | float | None = None
    artifacts: tuple[Mapping[str, object], ...] = ()


@dataclass(frozen=True)
class ProbeResult:
    available: bool
    descriptor: ModuleDescriptor | None = None
    error: str | None = None
