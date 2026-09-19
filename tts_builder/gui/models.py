from __future__ import annotations

from dataclasses import dataclass, field

from ..events import PipelineEvent

STAGES = ("prepare", "source", "separate", "normalize", "asr", "segment", "export")


@dataclass
class StageState:
    status: str = "pending"
    percent: int = 0


@dataclass
class ProgressModel:
    stages: dict[str, StageState] = field(
        default_factory=lambda: {name: StageState() for name in STAGES}
    )

    def consume(self, event: PipelineEvent) -> None:
        if event.stage not in self.stages:
            return
        state = self.stages[event.stage]
        if event.kind == "stage_started":
            state.status = "running"
        elif event.kind == "stage_cache_hit":
            state.status = "cached"
            state.percent = 100
        elif event.kind == "stage_completed":
            state.status = "completed"
            state.percent = 100
        elif event.kind == "stage_progress":
            state.status = "running"
            if event.total and event.total > 0 and event.current is not None:
                state.percent = max(0, min(99, int(event.current / event.total * 100)))
        elif event.kind in {"stage_failed", "pipeline_failed"}:
            state.status = "failed"

    @property
    def overall_percent(self) -> int:
        total = 0.0
        for state in self.stages.values():
            if state.status in {"completed", "cached"}:
                total += 1.0
            elif state.status == "running":
                total += state.percent / 100.0
        return int(round(total / len(self.stages) * 100))
