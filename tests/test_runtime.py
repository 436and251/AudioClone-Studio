import pytest

import tts_builder.runtime as runtime


def test_resolve_binary_uses_system_path(monkeypatch):
    monkeypatch.setattr(runtime.shutil, "which", lambda name: "/path/ffmpeg")

    assert runtime.resolve_binary("ffmpeg") == "/path/ffmpeg"


def test_resolve_binary_has_no_packaged_bin_fallback(monkeypatch):
    monkeypatch.setattr(runtime.shutil, "which", lambda name: None)

    assert not hasattr(runtime, "application_dir")
    with pytest.raises(RuntimeError, match="PATH"):
        runtime.resolve_binary("ffmpeg")
