import json
import os
from pathlib import Path

import pytest


MODULE = "gpt-sovits"
FRAMEWORK = "v2ProPlus"
PROJECT_NAME = "Acane"


def _job(project: Path, job_id: str, *, overrides=None) -> tuple[Path, Path]:
    directory = project / "jobs" / job_id
    directory.mkdir(parents=True)
    payload = {
        "protocol_version": 2,
        "job_id": job_id,
        "module_id": MODULE,
        "framework": FRAMEWORK,
        "project_name": PROJECT_NAME,
        "project_root": str(project.resolve()),
        "job_dir": str(directory.resolve()),
    }
    payload.update(overrides or {})
    job = directory / "job.json"
    job.write_text(json.dumps(payload), encoding="utf-8")
    return job, directory / "events.jsonl"


def _promotion(job_id: str, model: Path) -> str:
    events = [
        {
            "protocol_version": 1,
            "job_id": job_id,
            "type": "artifact",
            "timestamp": "2026-09-18T00:00:00+00:00",
            "artifacts": [{"type": "promoted_model", "path": str(model.resolve())}],
        },
        {
            "protocol_version": 1,
            "job_id": job_id,
            "type": "promotion_completed",
            "timestamp": "2026-09-18T00:00:01+00:00",
        },
    ]
    return "".join(json.dumps(event) + "\n" for event in events)


def _recover(project: Path):
    from tts_builder.training_modules.recovery import latest_promoted_model

    return latest_promoted_model(project, MODULE, FRAMEWORK, PROJECT_NAME)


def test_returns_newest_valid_completed_promotion(tmp_path):
    project = tmp_path / PROJECT_NAME
    old = project / "models" / "old"
    new = project / "models" / "new"
    old.mkdir(parents=True)
    new.mkdir(parents=True)
    _, old_events = _job(project, "old")
    _, new_events = _job(project, "new")
    old_events.write_text(_promotion("old", old), encoding="utf-8")
    new_events.write_text(_promotion("new", new), encoding="utf-8")
    os.utime(old_events, (1, 1))
    os.utime(new_events, (2, 2))

    assert _recover(project) == new.resolve()


def test_accepts_equivalent_windows_path_separators(tmp_path):
    project = tmp_path / PROJECT_NAME
    model = project / "models" / "model"
    model.mkdir(parents=True)
    directory = project / "jobs" / "job"
    _, events = _job(
        project,
        "job",
        overrides={
            "project_root": project.resolve().as_posix(),
            "job_dir": directory.resolve().as_posix(),
        },
    )
    events.write_text(_promotion("job", model), encoding="utf-8")

    assert _recover(project) == model.resolve()


def test_skips_missing_malformed_and_incomplete_files(tmp_path):
    project = tmp_path / PROJECT_NAME
    project.mkdir()
    _job(project, "missing-events")
    _, malformed = _job(project, "malformed")
    malformed.write_text("not-json\n", encoding="utf-8")
    model = project / "models" / "incomplete"
    model.mkdir(parents=True)
    _, incomplete = _job(project, "incomplete")
    incomplete.write_text(_promotion("incomplete", model).rstrip("\n"), encoding="utf-8")

    assert _recover(project) is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("protocol_version", 1),
        ("module_id", "other-module"),
        ("framework", "other-framework"),
        ("project_name", "Other"),
    ],
)
def test_ignores_jobs_with_wrong_identity(tmp_path, field, value):
    project = tmp_path / PROJECT_NAME
    model = project / "models" / field
    model.mkdir(parents=True)
    _, events = _job(project, field, overrides={field: value})
    events.write_text(_promotion(field, model), encoding="utf-8")

    assert _recover(project) is None


def test_rejects_job_directory_traversal(tmp_path):
    project = tmp_path / PROJECT_NAME
    model = project / "models" / "inside"
    model.mkdir(parents=True)
    _, traversal = _job(
        project, "traversal", overrides={"job_dir": str(tmp_path.resolve())}
    )
    traversal.write_text(_promotion("traversal", model), encoding="utf-8")

    assert _recover(project) is None


def test_rejects_promoted_model_symlink_escape(tmp_path):
    project = tmp_path / PROJECT_NAME
    (project / "models").mkdir(parents=True)

    outside = tmp_path / "outside"
    outside_model = outside / "model"
    outside_model.mkdir(parents=True)
    link = project / "models" / "linked"
    try:
        link.symlink_to(outside_model, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are unavailable")
    _, escaped = _job(project, "escaped")
    escaped.write_text(_promotion("escaped", link), encoding="utf-8")

    assert _recover(project) is None


def test_falls_back_to_older_existing_promotion(tmp_path):
    project = tmp_path / PROJECT_NAME
    valid = project / "models" / "valid"
    deleted = project / "models" / "deleted"
    valid.mkdir(parents=True)
    deleted.mkdir(parents=True)
    _, events = _job(project, "one-job")
    events.write_text(
        _promotion("one-job", valid) + _promotion("one-job", deleted),
        encoding="utf-8",
    )
    deleted.rmdir()

    assert _recover(project) == valid.resolve()


def test_scan_is_limited_to_30_newest_jobs(tmp_path):
    project = tmp_path / PROJECT_NAME
    model = project / "models" / "too-old"
    model.mkdir(parents=True)
    old_job, old_events = _job(project, "old")
    old_events.write_text(_promotion("old", model), encoding="utf-8")
    os.utime(old_job, (1, 1))
    os.utime(old_events, (1, 1))
    for index in range(30):
        job, events = _job(project, f"new-{index:02d}")
        events.write_text("not-json\n", encoding="utf-8")
        modified = index + 2
        os.utime(job, (modified, modified))
        os.utime(events, (modified, modified))

    assert _recover(project) is None


@pytest.mark.parametrize(
    ("field", "value"),
    [("protocol_version", 2), ("job_id", "other-job")],
)
def test_wrong_event_protocol_or_job_id_cannot_complete_promotion(
    tmp_path, field, value
):
    project = tmp_path / PROJECT_NAME
    model = project / "models" / "model"
    model.mkdir(parents=True)
    _, events = _job(project, "job")
    lines = _promotion("job", model).splitlines()
    completed = json.loads(lines[1])
    completed[field] = value
    events.write_text(lines[0] + "\n" + json.dumps(completed) + "\n", encoding="utf-8")

    assert _recover(project) is None


def test_returns_latest_matching_failed_job_and_stage(tmp_path):
    from tts_builder.training_modules.recovery import FailedJob, latest_failed_job

    project = tmp_path / PROJECT_NAME
    job, events = _job(
        project,
        "failed-job",
        overrides={
            "output_root": str((project / "runs").resolve()),
            "stages": ["preprocess", "s2", "s1", "evaluate"],
        },
    )
    events.write_text("{}\n", encoding="utf-8")
    state = project / "runs" / PROJECT_NAME / "pipeline-state.json"
    state.parent.mkdir(parents=True)
    state.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "identity": {
                    "pipeline": (job.parent / "module" / "pipeline.yaml").resolve().as_posix(),
                    "config": (job.parent / "module" / "train.yaml").resolve().as_posix(),
                    "stages": ["preprocess", "s2", "s1", "evaluate"],
                },
                "stages": {
                    "preprocess": "completed",
                    "s2": "completed",
                    "s1": "completed",
                    "evaluate": "failed",
                },
                "failure": {"stage": "evaluate", "type": "OSError", "message": "failed"},
            }
        ),
        encoding="utf-8",
    )

    assert latest_failed_job(project, MODULE, FRAMEWORK, PROJECT_NAME) == FailedJob(
        job.resolve(), "evaluate"
    )


def _complete_bundle(root: Path) -> Path:
    for relative in (
        "model.yaml",
        "metadata.json",
        "weights/s1.ckpt",
        "weights/s2.pth",
        "reference/default.json",
        "reference/default.wav",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    return root


def test_model_history_lists_only_complete_local_models_and_latest_audio(tmp_path):
    from tts_builder.training_modules.recovery import local_model_history

    project = tmp_path / "project"
    acane = _complete_bundle(project / "models" / "Acane")
    lucy = _complete_bundle(project / "models" / "Lucy")
    (project / "models" / "broken").mkdir(parents=True)
    os.utime(acane, (1, 1))
    os.utime(lucy, (2, 2))
    old = project / "outputs" / "Acane" / "gui" / "old.wav"
    new = project / "outputs" / "Acane" / "gui" / "new.wav"
    old.parent.mkdir(parents=True)
    old.write_bytes(b"old")
    new.write_bytes(b"new")
    os.utime(old, (1, 1))
    os.utime(new, (2, 2))

    history = local_model_history(project, MODULE, FRAMEWORK)

    assert [item.model.name for item in history] == ["Lucy", "Acane"]
    assert history[0].model == lucy.resolve()
    assert history[0].latest_audio is None
    assert history[1].model == acane.resolve()
    assert history[1].latest_audio == new.resolve()
