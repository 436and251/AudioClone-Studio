from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Audio2ScoreProject:
    root: Path
    work: Path
    stems: Path
    reports: Path


@dataclass(frozen=True)
class SeparationResult:
    stems: dict[str, Path]


@dataclass(frozen=True)
class Audio2ScoreSummary:
    project_dir: Path
    manifest_path: Path
    stems: dict[str, Path]
    timeline_path: Path | None = None
    transcription_path: Path | None = None
    score_paths: dict[str, Path] = field(default_factory=dict)
    status: str = "completed"
    message: str = ""
