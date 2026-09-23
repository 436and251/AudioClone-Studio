from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .models import Audio2ScoreProject


WINDOWS_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def validate_title(value: str) -> str:
    title = value.strip()
    invalid = (
        not title
        or title in {".", ".."}
        or title[-1] in {".", " "}
        or any(ord(char) < 32 or char in '<>:"/\\|?*' for char in title)
        or title.upper().split(".", 1)[0] in WINDOWS_RESERVED
    )
    if invalid:
        raise ValueError("title is not a safe project directory name")
    return title


@dataclass(frozen=True)
class Audio2ScoreConfig:
    output: Path
    title: str
    device: str = "auto"
    keep_work: bool = False
    fresh: bool = False

    def project(self) -> Audio2ScoreProject:
        title = validate_title(self.title)
        root = self.output.expanduser().resolve() / title
        project = Audio2ScoreProject(
            root=root,
            work=root / ".work",
            stems=root / "stems",
            reports=root / "reports",
        )
        return project
