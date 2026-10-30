from pathlib import Path

import numpy as np
import soundfile as sf
import pytest

from tts_builder.config import BuildConfig
from tts_builder.dataset import load_manifest
from tts_builder.models import AcquiredSource, TranscriptResult, TranscriptSegment, WordStamp
from tts_builder.pipeline import process_source


def _transcript():
    return TranscriptResult(
        language="ja",
        language_probability=0.99,
        segments=[TranscriptSegment(0, 8, "こんにちは。元気です。", 0.9)],
        words=[
            WordStamp(0.2, 4.2, "こんにちは。", 0.9),
            WordStamp(5.0, 8.8, "元気です。", 0.9),
        ],
    )


def _install_audio_stubs(monkeypatch, counters, fail_asr_once=False):
    def acquire(value, work_dir):
        counters["acquire"] += 1
        path = Path(work_dir) / "source.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, np.zeros(44100 * 10, dtype=np.float32), 44100)
        return AcquiredSource(str(path), "source.wav", "source_abcd", value)

    def separate(path, root, **kwargs):
        counters["separate"] += 1
        out = Path(root) / "htdemucs" / "source" / "vocals.flac"
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out, np.zeros(44100 * 10, dtype=np.float32), 44100)
        return out

    def normalize(path, out):
        counters["normalize"] += 1
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out, np.zeros(44100 * 10, dtype=np.float32), 44100)
        return out

    calls = {"n": 0}

    def transcribe(*args, **kwargs):
        counters["asr"] += 1
        calls["n"] += 1
        if fail_asr_once and calls["n"] == 1:
            raise RuntimeError("simulated ASR interruption")
        return _transcript()

    monkeypatch.setattr("tts_builder.pipeline.acquire_source", acquire)
    monkeypatch.setattr("tts_builder.pipeline.separate_vocals", separate)
    monkeypatch.setattr("tts_builder.pipeline.normalize_audio", normalize)
    monkeypatch.setattr("tts_builder.pipeline.transcribe", transcribe)


def test_failed_run_keeps_completed_stages_and_resumes_at_asr(tmp_path, monkeypatch):
    counters = {"acquire": 0, "separate": 0, "normalize": 0, "asr": 0}
    _install_audio_stubs(monkeypatch, counters, fail_asr_once=True)
    cfg = BuildConfig(output=tmp_path / "dataset", speaker="suis", language="ja")

    with pytest.raises(RuntimeError, match="simulated ASR interruption"):
        process_source("https://example.test/video", cfg)

    assert counters == {"acquire": 1, "separate": 1, "normalize": 1, "asr": 1}
    assert list((cfg.output / ".cache").rglob("source.wav"))
    assert list((cfg.output / ".cache").rglob("vocals.flac"))
    assert list((cfg.output / ".cache").rglob("normalized.wav"))

    process_source("https://example.test/video", cfg)

    assert counters == {"acquire": 1, "separate": 1, "normalize": 1, "asr": 2}


def test_success_compacts_cache_but_keeps_source_and_asr_json(tmp_path, monkeypatch):
    counters = {"acquire": 0, "separate": 0, "normalize": 0, "asr": 0}
    _install_audio_stubs(monkeypatch, counters)
    cfg = BuildConfig(output=tmp_path / "dataset", speaker="suis", language="ja")

    process_source("https://example.test/video", cfg)

    cache = cfg.output / ".cache"
    assert list(cache.rglob("source.wav"))
    assert list(cache.rglob("asr.json"))
    assert list(cache.rglob("state.json"))
    assert not list(cache.rglob("vocals.flac"))
    assert not list(cache.rglob("normalized.wav"))
    assert list((cfg.output / "clips").glob("*.wav"))


def test_changed_speaker_reuses_asr_and_replaces_source_manifest(tmp_path, monkeypatch):
    counters = {"acquire": 0, "separate": 0, "normalize": 0, "asr": 0}
    _install_audio_stubs(monkeypatch, counters)
    output = tmp_path / "dataset"

    process_source(
        "https://example.test/video",
        BuildConfig(output=output, speaker="old", language="ja"),
    )
    process_source(
        "https://example.test/video",
        BuildConfig(output=output, speaker="new", language="ja"),
    )

    assert counters == {"acquire": 1, "separate": 1, "normalize": 1, "asr": 1}
    records = load_manifest(output / "manifest.jsonl")
    assert records
    assert {record.speaker for record in records} == {"new"}


def test_different_sources_accumulate_in_one_dataset_with_isolated_caches(tmp_path, monkeypatch):
    from tts_builder.sources.local import acquire_local
    counters = {"acquire": 0, "separate": 0, "normalize": 0, "asr": 0}
    _install_audio_stubs(monkeypatch, counters)
    # Keep real source IDs: even equal file names in different directories must not collide.
    monkeypatch.setattr("tts_builder.pipeline.acquire_source", lambda source, *_: acquire_local(source))
    first, second = tmp_path / "one" / "song.wav", tmp_path / "two" / "song.wav"
    for source in (first, second):
        source.parent.mkdir()
        sf.write(source, np.zeros(44100 * 10, np.float32), 44100)
    cfg = BuildConfig(output=tmp_path / "dataset", speaker="suis", language="ja", skip_separation=True)
    process_source(str(first), cfg)
    first_records = load_manifest(cfg.output / "manifest.jsonl")
    process_source(str(second), cfg)
    combined = load_manifest(cfg.output / "manifest.jsonl")
    assert len(combined) == 2 * len(first_records) > 0
    assert all((cfg.output / row.audio).is_file() for row in combined)
    process_source(str(first), cfg)
    assert len(load_manifest(cfg.output / "manifest.jsonl")) == len(combined)
    assert counters["asr"] == 2
    assert len(list((cfg.output / ".cache").glob("*/state.json"))) == 2
