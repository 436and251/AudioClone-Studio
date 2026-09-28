from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Audio2ScoreProject:
    root: Path
    work: Path
    stems: Path
    reports: Path


@dataclass(frozen=True)
class SeparationResult:
    evidence_dir: Path
    formal_model_dir: Path
    stems: dict[str, Path]


@dataclass(frozen=True)
class Audio2ScoreSummary:
    project_dir: Path
    manifest_path: Path
    stems: dict[str, Path]
    timeline_path: Path | None = None
    transcription_path: Path | None = None
