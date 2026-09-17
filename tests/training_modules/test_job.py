import hashlib
import json
from pathlib import Path
import uuid

import pytest

from tts_builder.training_modules.models import FrameworkDescriptor, ModuleDescriptor


def _descriptors():
    framework = FrameworkDescriptor(
        id="v2ProPlus",
        display_name="GPT-SoVITS v2ProPlus",
        capabilities=("preprocess", "train", "evaluate"),
        fields=(),
    )
    return ModuleDescriptor(1, "gpt-sovits-v2proplus", "1.0.0", (framework,)), framework


def _project(tmp_path: Path):
    project = tmp_path / "Acane"
    dataset = project / "dataset" / "dataset.list"
    dataset.parent.mkdir(parents=True)
    dataset.write_text("clip.wav|Acane|ja|テスト\n", encoding="utf-8")
    return project, dataset


def _values(project: Path):
    return {
        "project_name": "Acane",
        "output_root": project / "runs",
        "device": "cuda:0",
        "precision": "fp16",
        "parameters": {"s2.batch_size": 2},
        "reference": None,
    }


def test_build_job_writes_exact_atomic_voice_pipeline_contract(tmp_path):
    from tts_builder.training_modules.job import build_job

    project, dataset = _project(tmp_path)
    module, framework = _descriptors()

    path = build_job(
        project,
        module,
        framework,
        _values(project),
        ("evaluate", "s1", "preprocess", "s2"),
        dataset,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert set(payload) == {
        "protocol_version", "job_id", "project_name", "project_root",
        "output_root", "dataset_list", "dataset_sha256", "framework",
        "stages", "device", "precision", "parameters", "reference", "job_dir",
    }
    assert path == project.resolve() / "jobs" / payload["job_id"] / "job.json"
    assert payload["project_root"] == str(project.resolve())
    assert payload["output_root"] == str((project / "runs").resolve())
    assert payload["dataset_list"] == str(dataset.resolve())
    assert payload["dataset_sha256"] == hashlib.sha256(dataset.read_bytes()).hexdigest()
    assert payload["framework"] == "v2ProPlus"
    assert payload["stages"] == ["preprocess", "s2", "s1", "evaluate"]
    assert payload["parameters"] == {"s2.batch_size": 2}
    assert payload["reference"] is None
    assert payload["job_dir"] == str(path.parent.resolve())
    assert not (path.parent / ".job.json.tmp").exists()


def test_build_job_refuses_to_overwrite_existing_uuid_directory(tmp_path, monkeypatch):
    from tts_builder.training_modules import job as job_module

    project, dataset = _project(tmp_path)
    module, framework = _descriptors()
    fixed = uuid.UUID("12345678-1234-5678-1234-567812345678")
    monkeypatch.setattr(job_module.uuid, "uuid4", lambda: fixed)

    first = job_module.build_job(
        project, module, framework, _values(project), ("preprocess",), dataset
    )
    with pytest.raises(FileExistsError):
        job_module.build_job(
            project, module, framework, _values(project), ("preprocess",), dataset
        )

    assert json.loads(first.read_text(encoding="utf-8"))["job_id"] == str(fixed)


@pytest.mark.parametrize("escaped", ["dataset", "output"])
def test_build_job_rejects_paths_outside_project(tmp_path, escaped):
    from tts_builder.training_modules.job import build_job

    project, dataset = _project(tmp_path)
    module, framework = _descriptors()
    outside = tmp_path / "outside"
    outside.mkdir()
    values = _values(project)
    if escaped == "dataset":
        dataset = outside / "dataset.list"
        dataset.write_text("outside", encoding="utf-8")
    else:
        values["output_root"] = outside / "runs"

    with pytest.raises(ValueError, match=escaped):
        build_job(project, module, framework, values, ("preprocess",), dataset)


def test_build_job_rejects_dataset_symlink_escape(tmp_path):
    from tts_builder.training_modules.job import build_job

    project, _ = _project(tmp_path)
    outside = tmp_path / "outside.list"
    outside.write_text("outside", encoding="utf-8")
    link = project / "dataset" / "linked.list"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks are unavailable")
    module, framework = _descriptors()

    with pytest.raises(ValueError, match="dataset"):
        build_job(project, module, framework, _values(project), ("preprocess",), link)
