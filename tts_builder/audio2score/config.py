from __future__ import annotations

from dataclasses import dataclass
import math
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
    model_root: Path | None = None
    tempo: float | None = None
    meter: tuple[int, int] | None = None
    musescore: Path | None = None

    def project(self) -> Audio2ScoreProject:
        title = validate_title(self.title)
        if self.tempo is not None and (not math.isfinite(self.tempo) or not 20 <= self.tempo <= 400):
            raise ValueError("tempo must be between 20 and 400 BPM")
        if self.meter is not None and (len(self.meter) != 2 or type(self.meter[0]) is not int
                                       or not 1 <= self.meter[0] <= 32 or self.meter[1] not in (2, 4, 8, 16)):
            raise ValueError("invalid meter")
        root = self.output.expanduser().resolve() / title
        project = Audio2ScoreProject(
            root=root,
            work=root / ".work",
            stems=root / "stems",
            reports=root / "reports",
        )
        return project
