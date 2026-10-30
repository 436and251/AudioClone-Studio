from __future__ import annotations

import json
import shutil
from pathlib import Path

from ..cache import _digest, source_identity
from .models import Audio2ScoreProject


class SourceMismatchError(ValueError):
    pass


def project_source(project: Audio2ScoreProject) -> dict | None:
    for path in (project.work / "state.json", project.root / "manifest.json"):
        try:
            source = json.loads(path.read_text(encoding="utf-8")).get("source")
            if isinstance(source, dict) and source.get("kind"):
                return source
        except (OSError, ValueError, AttributeError):
            continue
    return None


def same_source(bound: dict, provided: dict) -> bool:
    # Existing stems can still be scored when the original local file is offline.
    if bound.get("kind") == provided.get("kind") == "local" and provided.get("size") is None:
        return bound.get("path") == provided.get("path")
    return bound == provided


class ScoreCache:
    def __init__(self, project: Audio2ScoreProject, source: str, fresh: bool = False):
        self.project = project
        self.source = source
        self.state_path = project.work / "state.json"
        if fresh:
            self._reset()
        self.state = self._load_state()
        bound = project_source(project)
        if bound:
            self.state.setdefault("source", bound)
            self.state.setdefault("source_digest", _digest(bound))

    def _reset(self) -> None:
        if self.project.root.is_symlink() or self.project.root.is_junction():
            raise ValueError(f"unsafe project root: {self.project.root}")
        root = self.project.root.resolve()
        targets = (self.project.work, self.project.reports, self.project.root / "score",
                   self.project.stems / "harmony", self.project.root / "score-events.json",
                   self.project.root / "manifest.json")
        identity = source_identity(self.source)
        if identity["kind"] == "local":
            source = Path(identity["path"])
            if any(source.is_relative_to(path.resolve()) for path in (*targets, self.project.stems)):
                raise ValueError("source must be outside generated project directories before a fresh run")
        # Preflight every target before any deletion; never delete the project root.
        for path in targets:
            if (path.is_symlink() or path.is_junction()
                    or not path.resolve().is_relative_to(root) or path.resolve() == root):
                raise ValueError(f"unsafe generated output path: {path}")
        for path in targets:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)

    def signature(self, payload) -> str:
        return _digest(payload)

    def bind_source(self, identity: dict | None = None) -> None:
        identity = identity or source_identity(self.source)
        digest = _digest(identity)
        bound = self.state.get("source_digest")
        if bound is not None and not same_source(self.state.get("source", {}), identity):
            raise SourceMismatchError("source changed; rerun the full task with --fresh to replace previous results")
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
