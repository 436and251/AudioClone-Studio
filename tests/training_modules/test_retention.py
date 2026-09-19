from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path


NOW = datetime(2026, 9, 19, 12, tzinfo=timezone.utc)


def _job(
    root: Path,
    job_id: str,
    *,
    project_name: str = "Acane",
    stages: tuple[str, ...] = ("preprocess", "s2", "s1", "evaluate"),
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    directory = root / "jobs" / job_id
    directory.mkdir(parents=True)
    payload = {
        "protocol_version": 2,
        "job_id": job_id,
        "module_id": "voice-pipeline",
        "project_name": project_name,
        "project_root": str(root.resolve()),
        "output_root": str((root / "runs").resolve()),
        "training_data": {"path": str((root / "data.list").resolve()), "kind": "file"},
        "framework": "gpt-sovits-v2proplus",
        "stages": list(stages),
        "device": "cuda:0",
        "precision": "fp16",
        "parameters": {},
        "reference": None,
        "job_dir": str(directory.resolve()),
    }
    (directory / "job.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    return directory


def _events(directory: Path, *items: tuple[str, datetime]) -> None:
    lines = [
        json.dumps(
            {
                "protocol_version": 1,
                "job_id": directory.name,
                "type": kind,
                "timestamp": timestamp.isoformat(),
            }
        )
        for kind, timestamp in items
    ]
    (directory / "events.jsonl").write_text(
        "".join(line + "\n" for line in lines), encoding="utf-8"
    )


def _cleanup(root: Path, *, active_job: Path | None = None):
    from tts_builder.training_modules.retention import cleanup_jobs

    return cleanup_jobs(root, now=NOW, active_job=active_job)


def test_active_and_incomplete_jobs_are_preserved(tmp_path):
    active = _job(tmp_path, "active")
    incomplete = _job(tmp_path, "incomplete")
    _events(active, ("job_completed", NOW - timedelta(days=3)))

    summary = _cleanup(tmp_path, active_job=active)

    assert active.is_dir()
    assert incomplete.is_dir()
    assert summary.removed == 0


def test_evaluated_job_waiting_for_promotion_is_preserved(tmp_path):
    pending = _job(tmp_path, "pending")
    _events(pending, ("job_completed", NOW - timedelta(days=10)))

    summary = _cleanup(tmp_path)

    assert pending.is_dir()
    assert summary.removed == 0


def test_promotion_failure_keeps_pending_candidates(tmp_path):
    pending = _job(tmp_path, "pending")
    _events(
        pending,
        ("job_completed", NOW - timedelta(hours=3)),
        ("promotion_failed", NOW - timedelta(hours=2)),
    )

    _cleanup(tmp_path)

    assert pending.is_dir()


def test_promoted_training_job_is_removed(tmp_path):
    promoted = _job(tmp_path, "promoted")
    _events(
        promoted,
        ("job_completed", NOW - timedelta(hours=2)),
        ("promotion_completed", NOW - timedelta(hours=1)),
    )

    summary = _cleanup(tmp_path)

    assert not promoted.exists()
    assert summary.removed == 1


def test_completed_non_evaluation_job_is_removed(tmp_path):
    completed = _job(tmp_path, "completed", stages=("preprocess", "s2"))
    _events(completed, ("job_completed", NOW))

    _cleanup(tmp_path)

    assert not completed.exists()


def test_completed_inference_job_is_removed_but_output_is_untouched(tmp_path):
    inference = _job(tmp_path, "inference-Acane")
    output = tmp_path / "outputs" / "Acane" / "gui" / "result.wav"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"RIFF-audio")
    _events(
        inference,
        ("promotion_completed", NOW - timedelta(minutes=2)),
        ("inference_completed", NOW - timedelta(minutes=1)),
    )

    _cleanup(tmp_path)

    assert not inference.exists()
    assert output.read_bytes() == b"RIFF-audio"


def test_inference_receipt_without_terminal_event_is_incomplete(tmp_path):
    inference = _job(tmp_path, "inference-Acane")
    _events(inference, ("promotion_completed", NOW - timedelta(days=5)))

    _cleanup(tmp_path)

    assert inference.is_dir()


def test_cancelled_job_is_removed(tmp_path):
    cancelled = _job(tmp_path, "cancelled")
    _events(cancelled, ("job_cancelled", NOW))

    _cleanup(tmp_path)

    assert not cancelled.exists()


def test_only_newest_failure_per_target_survives(tmp_path):
    older = _job(tmp_path, "acane-old")
    newer = _job(tmp_path, "acane-new")
    lucy = _job(tmp_path, "lucy", project_name="Lucy")
    _events(older, ("job_failed", NOW - timedelta(hours=8)))
    _events(newer, ("job_failed", NOW - timedelta(hours=2)))
    _events(lucy, ("job_failed", NOW - timedelta(hours=4)))

    summary = _cleanup(tmp_path)

    assert not older.exists()
    assert newer.is_dir()
    assert lucy.is_dir()
    assert summary.retained_failures == 2


def test_latest_failure_older_than_48_hours_is_removed(tmp_path):
    failed = _job(tmp_path, "failed")
    _events(failed, ("job_failed", NOW - timedelta(hours=48, seconds=1)))

    _cleanup(tmp_path)

    assert not failed.exists()


def test_failure_exactly_48_hours_old_is_retained(tmp_path):
    failed = _job(tmp_path, "failed")
    _events(failed, ("job_failed", NOW - timedelta(hours=48)))

    summary = _cleanup(tmp_path)

    assert failed.is_dir()
    assert summary.retained_failures == 1


def test_missing_terminal_event_is_preserved_past_cutoff(tmp_path):
    interrupted = _job(tmp_path, "interrupted")
    _events(interrupted, ("job_started", NOW - timedelta(days=30)))

    _cleanup(tmp_path)

    assert interrupted.is_dir()


def test_malformed_job_or_journal_is_uncertain(tmp_path):
    bad_job = _job(tmp_path, "bad-job")
    (bad_job / "job.json").write_text("not-json", encoding="utf-8")
    bad_events = _job(tmp_path, "bad-events")
    (bad_events / "events.jsonl").write_text("not-json\n", encoding="utf-8")

    summary = _cleanup(tmp_path)

    assert bad_job.is_dir()
    assert bad_events.is_dir()
    assert summary.uncertain == 2


def test_link_like_job_is_uncertain(tmp_path, monkeypatch):
    linked = _job(tmp_path, "linked")
    _events(linked, ("promotion_completed", NOW))
    original = Path.is_symlink
    monkeypatch.setattr(
        Path,
        "is_symlink",
        lambda path: path == linked or original(path),
    )

    summary = _cleanup(tmp_path)

    assert linked.is_dir()
    assert summary.uncertain == 1


def test_active_job_argument_prevents_terminal_job_deletion(tmp_path):
    active = _job(tmp_path, "active")
    _events(active, ("promotion_completed", NOW))

    summary = _cleanup(tmp_path, active_job=active / "job.json")

    assert active.is_dir()
    assert summary.removed == 0
