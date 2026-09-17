import hashlib
import json
from pathlib import Path

import pytest


def _manifest(tmp_path: Path, candidates=("candidate_A", "candidate_B", "candidate_C")):
    entries = []
    for candidate in candidates:
        samples = []
        for language in ("zh", "ja", "en"):
            path = tmp_path / candidate / f"{language}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"{candidate}-{language}".encode())
            samples.append(
                {
                    "language": language,
                    "text": f"{language} text",
                    "wav": f"{candidate}/{language}.wav",
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        entries.append({"candidate": candidate, "samples": samples})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"schema_version": 1, "candidates": entries}), encoding="utf-8"
    )
    return manifest


def test_load_listening_manifest_validates_real_three_language_shape(tmp_path):
    from tts_builder.training_modules.artifacts import load_listening_manifest

    candidates = load_listening_manifest(_manifest(tmp_path), tmp_path)

    assert [candidate.visible_name for candidate in candidates] == ["A", "B", "C"]
    assert [candidate.internal_id for candidate in candidates] == [
        "candidate_A", "candidate_B", "candidate_C",
    ]
    assert [sample.language for sample in candidates[0].samples] == ["zh", "ja", "en"]
    assert all(sample.path.is_absolute() for sample in candidates[0].samples)


@pytest.mark.parametrize("failure", ["traversal", "missing", "hash", "language", "duplicate"])
def test_load_listening_manifest_rejects_unsafe_or_incomplete_entries(tmp_path, failure):
    from tts_builder.training_modules.artifacts import load_listening_manifest

    manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if failure == "traversal":
        outside = tmp_path.parent / "outside.wav"
        outside.write_bytes(b"outside")
        sample = payload["candidates"][0]["samples"][0]
        sample["wav"] = "../outside.wav"
        sample["sha256"] = hashlib.sha256(outside.read_bytes()).hexdigest()
    elif failure == "missing":
        (tmp_path / "candidate_A" / "zh.wav").unlink()
    elif failure == "hash":
        payload["candidates"][0]["samples"][0]["sha256"] = "0" * 64
    elif failure == "language":
        payload["candidates"][0]["samples"][0]["language"] = "ja"
    else:
        payload["candidates"][1]["candidate"] = "candidate_A"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        load_listening_manifest(manifest, tmp_path)


def test_load_listening_manifest_rejects_symlink_escape(tmp_path):
    from tts_builder.training_modules.artifacts import load_listening_manifest

    manifest = _manifest(tmp_path)
    outside = tmp_path.parent / "outside.wav"
    outside.write_bytes(b"outside")
    link = tmp_path / "candidate_A" / "zh.wav"
    link.unlink()
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks are unavailable")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["candidates"][0]["samples"][0]["sha256"] = hashlib.sha256(
        outside.read_bytes()
    ).hexdigest()
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="outside"):
        load_listening_manifest(manifest, tmp_path)
