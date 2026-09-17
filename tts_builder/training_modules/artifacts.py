from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re


_SHA256 = re.compile(r"[0-9a-f]{64}")
_CANDIDATES = {"candidate_A": "A", "candidate_B": "B", "candidate_C": "C"}
_LANGUAGES = ("zh", "ja", "en")


@dataclass(frozen=True, slots=True)
class ListeningSample:
    language: str
    text: str
    path: Path
    sha256: str


@dataclass(frozen=True, slots=True)
class ListeningCandidate:
    internal_id: str
    visible_name: str
    samples: tuple[ListeningSample, ...]


def load_listening_manifest(
    path: Path, allowed_root: Path
) -> tuple[ListeningCandidate, ...]:
    root = Path(allowed_root).resolve()
    manifest = Path(path).resolve()
    if not root.is_dir() or not manifest.is_relative_to(root) or not manifest.is_file():
        raise ValueError("listening manifest is outside the allowed root or missing")
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid listening manifest: {error}") from error
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "candidates"}:
        raise ValueError("invalid listening manifest fields")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValueError("listening manifest schema_version must be 1")
    entries = payload["candidates"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("listening candidates must be a non-empty array")

    candidates = tuple(_candidate(entry, root) for entry in entries)
    identifiers = [candidate.internal_id for candidate in candidates]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("listening candidate IDs must be unique")
    return candidates


def _candidate(value: object, root: Path) -> ListeningCandidate:
    if not isinstance(value, dict) or set(value) != {"candidate", "samples"}:
        raise ValueError("invalid listening candidate")
    identifier = value["candidate"]
    if not isinstance(identifier, str) or identifier not in _CANDIDATES:
        raise ValueError("unsupported listening candidate ID")
    samples = value["samples"]
    if not isinstance(samples, list) or len(samples) != 3:
        raise ValueError("listening candidate must contain three samples")
    parsed = tuple(_sample(sample, root) for sample in samples)
    by_language = {sample.language: sample for sample in parsed}
    if len(by_language) != 3 or set(by_language) != set(_LANGUAGES):
        raise ValueError("listening samples must contain unique zh, ja, and en")
    return ListeningCandidate(
        identifier,
        _CANDIDATES[identifier],
        tuple(by_language[language] for language in _LANGUAGES),
    )


def _sample(value: object, root: Path) -> ListeningSample:
    fields = {"language", "text", "wav", "sha256"}
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("invalid listening sample")
    language = value["language"]
    text = value["text"]
    wav = value["wav"]
    digest = value["sha256"]
    if language not in _LANGUAGES:
        raise ValueError("unsupported listening language")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("listening sample text must not be empty")
    if not isinstance(wav, str) or not wav or Path(wav).is_absolute():
        raise ValueError("listening WAV path must be relative")
    path = (root / wav).resolve()
    if not path.is_relative_to(root):
        raise ValueError("listening WAV is outside the allowed root")
    if path.suffix.lower() != ".wav" or not path.is_file():
        raise ValueError("listening WAV is missing")
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise ValueError("listening WAV sha256 is invalid")
    if _sha256(path) != digest:
        raise ValueError("listening WAV sha256 does not match")
    return ListeningSample(language, text.strip(), path, digest)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
