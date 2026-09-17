import json
from pathlib import Path


def _event(**changes):
    value = {
        "protocol_version": 1,
        "job_id": "job-1",
        "type": "stage_progress",
        "timestamp": "2026-09-17T00:00:00+00:00",
        "stage": "s2",
        "current": 1,
        "total": 2,
    }
    value.update(changes)
    return value


def _line(value):
    return (json.dumps(value, ensure_ascii=False) + "\n").encode("utf-8")


def test_event_store_tails_split_utf8_and_replays_after_restart(tmp_path: Path):
    from tts_builder.training_modules.event_store import EventStore

    journal = tmp_path / "events.jsonl"
    encoded = _line(_event(message_key="进捗"))
    split = encoded.index("进".encode("utf-8")) + 1
    journal.write_bytes(encoded[:split])
    store = EventStore(journal, tmp_path, expected_job_id="job-1")

    first = store.read_new()
    assert first.events == ()
    assert first.errors == ()
    assert store.offset == split

    with journal.open("ab") as stream:
        stream.write(encoded[split:])
        stream.write(_line(_event(type="future_event", stage=None)))
    second = store.read_new()

    assert [event.type for event in second.events] == ["stage_progress", "future_event"]
    assert second.events[0].message_key == "进捗"
    assert second.errors == ()
    assert store.offset == journal.stat().st_size

    replay = EventStore(journal, tmp_path, expected_job_id="job-1").read_new()
    assert replay.events == second.events


def test_event_store_isolates_malformed_and_oversized_lines(tmp_path: Path):
    from tts_builder.training_modules.event_store import EventStore, MAX_LINE_BYTES

    journal = tmp_path / "events.jsonl"
    journal.write_bytes(
        b"not-json\n"
        + (b"x" * (MAX_LINE_BYTES + 1))
        + b"\n"
        + _line(_event(type="job_completed", stage=None))
    )

    store = EventStore(journal, tmp_path, expected_job_id="job-1")
    events, errors = [], []
    while store.offset < journal.stat().st_size:
        batch = store.read_new()
        events.extend(batch.events)
        errors.extend(batch.errors)

    assert [event.type for event in events] == ["job_completed"]
    assert len(errors) == 2
    assert any("JSON" in error for error in errors)
    assert any("maximum" in error for error in errors)


def test_event_store_rejects_invalid_schema_and_escaped_artifacts(tmp_path: Path):
    from tts_builder.training_modules.event_store import EventStore

    outside = tmp_path.parent / "outside.wav"
    outside.write_bytes(b"wav")
    journal = tmp_path / "events.jsonl"
    journal.write_bytes(
        _line(_event(protocol_version=2))
        + _line(
            _event(
                type="artifact",
                stage=None,
                artifacts=[{"type": "audio", "path": str(outside.resolve())}],
            )
        )
        + _line(_event(type="job_completed", stage=None))
    )

    batch = EventStore(journal, tmp_path, expected_job_id="job-1").read_new()

    assert [event.type for event in batch.events] == ["job_completed"]
    assert len(batch.errors) == 2


def test_event_store_bounds_bytes_processed_per_poll(tmp_path: Path, monkeypatch):
    import tts_builder.training_modules.event_store as event_store

    monkeypatch.setattr(event_store, "_READ_BYTES", 128)
    journal = tmp_path / "events.jsonl"
    journal.write_bytes(b"".join(_line(_event(current=index)) for index in range(10)))
    store = event_store.EventStore(journal, tmp_path, expected_job_id="job-1")

    first = store.read_new()

    assert store.offset == 128
    assert first.events == ()
    events = []
    while store.offset < journal.stat().st_size:
        events.extend(store.read_new().events)
    assert len(events) == 10
