from __future__ import annotations

import json
import shutil
from pathlib import Path

from ..cache import _digest, source_identity
from .models import Audio2ScoreProject


class SourceMismatchError(ValueError):
    pass


class ScoreCache:
    def __init__(self, project: Audio2ScoreProject, source: str, fresh: bool = False):
        self.project = project
        self.source = source
        if fresh:
            shutil.rmtree(project.work, ignore_errors=True)
        self.state_path = project.work / "state.json"
        self.state = self._load_state()

    def signature(self, payload) -> str:
        return _digest(payload)

    def bind_source(self, identity: dict | None = None) -> None:
        identity = identity or source_identity(self.source)
        digest = _digest(identity)
        bound = self.state.get("source_digest")
        if bound is not None and bound != digest:
            raise SourceMismatchError("project is already bound to a different source")
        if bound is None:
            self.state["source"] = identity
            self.state["source_digest"] = digest
            self._save()

    def begin(self, name: str, signature: str) -> None:
        self._set_stage(name, {"status": "running", "signature": signature})

    def complete(self, name: str, signature: str, **data) -> None:
        self._set_stage(name, {"status": "completed", "signature": signature, **data})

    def fail(self, name: str, signature: str, error: Exception) -> None:
        self._set_stage(
            name, {"status": "failed", "signature": signature, "error": str(error)}
        )

    def stage_hit(self, name: str, signature: str, artifact: Path | None = None) -> bool:
        stage = self.stage_data(name)
        if stage.get("status") != "completed" or stage.get("signature") != signature:
            return False
        return artifact is None or artifact.exists()

    def stage_data(self, name: str) -> dict:
        return self.state.get("stages", {}).get(name, {})

    def stored_path(self, path: Path) -> str:
        resolved = path.resolve()
        try:
            return resolved.relative_to(self.project.root.resolve()).as_posix()
        except ValueError:
            return str(resolved)

    def resolved_path(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.project.root / path

    def _load_state(self) -> dict:
        if self.state_path.exists():
            try:
                return json.loads(self.state_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        return {"version": 1, "stages": {}}

    def _set_stage(self, name: str, data: dict) -> None:
        self.state.setdefault("stages", {})[name] = data
        self._save()

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.state_path.with_suffix(".json.tmp")
        temp.write_text(
            json.dumps(self.state, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temp.replace(self.state_path)
