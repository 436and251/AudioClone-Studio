from datetime import datetime
import json
from pathlib import Path

import pytest


def _job(tmp_path: Path) -> tuple[Path, Path, Path]:
    project = tmp_path / "Acane"
    job_dir = project / "jobs" / "job-001"
    job_dir.mkdir(parents=True)
    job = job_dir / "job.json"
    job.write_text(
        json.dumps({
            "protocol_version": 2,
            "job_id": "job-001",
            "project_name": "Acane",
            "project_root": str(project.resolve()),
            "job_dir": str(job_dir.resolve()),
        }),
        encoding="utf-8",
    )
    model = project / "models" / "Acane"
    model.mkdir(parents=True)
    return job, project, model


def test_build_inline_request_uses_fixed_output_namespace_and_atomic_json(tmp_path):
    from tts_builder.training_modules.inference import build_inference_request

    job, project, model = _job(tmp_path)
    request = build_inference_request(
        job, model, "hello", None, "en", "cuda:0",
        datetime(2026, 9, 18, 12, 34, 56),
    )
    payload = json.loads(request.read_text(encoding="utf-8"))

    assert request.parent.parent == job.parent / "inference"
    assert request.parent.name == payload["request_id"]
    assert payload == {
        "protocol_version": 2,
        "request_id": request.parent.name,
        "project_root": str(project.resolve()),
        "project_name": "Acane",
        "model": str(model.resolve()),
        "text": "hello",
        "text_file": None,
        "language": "en",
        "device": "cuda:0",
        "output": str(
            (project / "outputs" / "Acane" / "gui" / "20260918-123456.wav").resolve()
        ),
    }
    assert not list(request.parent.glob("*.tmp"))


def test_external_utf8_txt_is_snapshotted_and_output_collision_is_numbered(tmp_path):
    from tts_builder.training_modules.inference import build_inference_request

    job, project, model = _job(tmp_path)
    external = tmp_path / "outside.txt"
    external.write_text("外部文本", encoding="utf-8")
    first_output = project / "outputs" / "Acane" / "gui" / "20260918-123456.wav"
    first_output.parent.mkdir(parents=True)
    first_output.write_bytes(b"old")

    request = build_inference_request(
        job, model, None, external, "mixed", "cpu",
        datetime(2026, 9, 18, 12, 34, 56),
    )
    payload = json.loads(request.read_text(encoding="utf-8"))
    snapshot = Path(payload["text_file"])

    assert snapshot.parent == request.parent
    assert snapshot.name == "input.txt"
    assert snapshot.read_text(encoding="utf-8") == "外部文本"
    assert payload["text"] is None
    assert Path(payload["output"]).name == "20260918-123456-2.wav"
    external.write_text("changed", encoding="utf-8")
    assert snapshot.read_text(encoding="utf-8") == "外部文本"

    second = build_inference_request(
        job, model, "second", None, "en", "cpu",
        datetime(2026, 9, 18, 12, 34, 56),
    )
    assert Path(json.loads(second.read_text(encoding="utf-8"))["output"]).name == (
        "20260918-123456-3.wav"
    )


@pytest.mark.parametrize(
    ("text", "text_file"),
    [(None, None), ("hello", "input.txt")],
)
def test_request_requires_exactly_one_text_source(tmp_path, text, text_file):
    from tts_builder.training_modules.inference import build_inference_request

    job, _, model = _job(tmp_path)
    source = tmp_path / "input.txt"
    source.write_text("file", encoding="utf-8")

    with pytest.raises(ValueError, match="exactly one"):
        build_inference_request(
            job,
            model,
            text,
            source if text_file else None,
            "en",
            "cpu",
            datetime(2026, 9, 18),
        )


def test_request_rejects_invalid_utf8_and_model_escape_without_partial_directory(tmp_path):
    from tts_builder.training_modules.inference import build_inference_request

    job, project, _ = _job(tmp_path)
    outside_model = tmp_path / "outside-model"
    outside_model.mkdir()
    bad_text = tmp_path / "bad.txt"
    bad_text.write_bytes(b"\xff")

    with pytest.raises(ValueError, match="model"):
        build_inference_request(
            job, outside_model, "hello", None, "en", "cpu", datetime(2026, 9, 18)
        )
    with pytest.raises(ValueError, match="UTF-8"):
        build_inference_request(
            job,
            project / "models" / "Acane",
            None,
            bad_text,
            "en",
            "cpu",
            datetime(2026, 9, 18),
        )
    assert not (job.parent / "inference").exists()
