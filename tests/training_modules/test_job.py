import json
from pathlib import Path
import uuid

import pytest

from tts_builder.training_modules.models import (
    FrameworkDescriptor,
    ModuleDescriptor,
    TrainingDataDescriptor,
)


def _descriptors():
    framework = FrameworkDescriptor(
        id="v2ProPlus",
        display_name="GPT-SoVITS v2ProPlus",
        capabilities=("preprocess", "train", "evaluate"),
        training_data=TrainingDataDescriptor("file", (".list",)),
        fields=(),
    )
    return ModuleDescriptor(2, "gpt-sovits-v2proplus", "1.0.0", (framework,)), framework


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
        "protocol_version", "job_id", "module_id", "project_name", "project_root",
        "output_root", "training_data", "framework",
        "stages", "device", "precision", "parameters", "reference", "job_dir",
    }
    assert payload["protocol_version"] == 2
    assert payload["module_id"] == "gpt-sovits-v2proplus"
    assert path == project.resolve() / "jobs" / payload["job_id"] / "job.json"
    assert payload["project_root"] == str(project.resolve())
    assert payload["output_root"] == str((project / "runs").resolve())
    assert payload["training_data"] == {
        "path": str(dataset.resolve()),
        "kind": "file",
    }
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


def test_build_job_accepts_external_training_data(tmp_path):
    from tts_builder.training_modules.job import build_job

    project, dataset = _project(tmp_path)
    module, framework = _descriptors()
    outside = tmp_path / "outside"
    outside.mkdir()
    dataset = outside / "dataset.list"
    dataset.write_text("clip.wav|Acane|ja|test\n", encoding="utf-8")

    path = build_job(
        project, module, framework, _values(project), ("preprocess",), dataset
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["training_data"]["path"] == str(dataset.resolve())


def test_build_job_accepts_external_reference_audio(tmp_path):
    from tts_builder.training_modules.job import build_job

    project, dataset = _project(tmp_path)
    outside = tmp_path / "reference.wav"
    outside.write_bytes(b"wav")
    module, framework = _descriptors()
    values = _values(project)
    values["reference"] = {"audio": outside, "text": "test", "language": "en"}

    path = build_job(
        project, module, framework, values, ("preprocess", "evaluate"), dataset
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["reference"]["audio"] == str(outside.resolve())


def test_build_job_still_rejects_output_outside_project(tmp_path):
    from tts_builder.training_modules.job import build_job

    project, dataset = _project(tmp_path)
    module, framework = _descriptors()
    outside = tmp_path / "outside"
    outside.mkdir()
    values = _values(project)
    values["output_root"] = outside / "runs"

    with pytest.raises(ValueError, match="output"):
        build_job(project, module, framework, values, ("preprocess",), dataset)


def test_build_job_accepts_directory_training_data(tmp_path):
    from tts_builder.training_modules.job import build_job

    project, _ = _project(tmp_path)
    training_data = project / "corpus"
    training_data.mkdir()
    framework = FrameworkDescriptor(
        "directory-framework",
        "Directory Framework",
        ("preprocess",),
        TrainingDataDescriptor("directory", ()),
        (),
    )
    module = ModuleDescriptor(2, "directory-module", "1.0", (framework,))

    path = build_job(
        project, module, framework, _values(project), ("preprocess",), training_data
    )

    assert json.loads(path.read_text(encoding="utf-8"))["training_data"] == {
        "path": str(training_data.resolve()),
        "kind": "directory",
    }


@pytest.mark.parametrize("name", ["dataset.json", "missing.list"])
def test_build_job_rejects_wrong_or_missing_file_training_data(tmp_path, name):
    from tts_builder.training_modules.job import build_job

    project, _ = _project(tmp_path)
    module, framework = _descriptors()
    training_data = project / name
    if training_data.suffix == ".json":
        training_data.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="training data"):
        build_job(
            project,
            module,
            framework,
            _values(project),
            ("preprocess",),
            training_data,
        )
