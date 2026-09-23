import sys

import pytest

from tts_builder.audio2score import runtime


def test_legacy_cli_does_not_load_score_dependencies():
    sys.modules.pop("librosa", None)
    sys.modules.pop("basic_pitch", None)
    import tts_builder.cli  # noqa: F401

    assert "librosa" not in sys.modules
    assert "basic_pitch" not in sys.modules


def test_runtime_error_names_the_single_install_file(monkeypatch):
    monkeypatch.setattr(runtime.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(RuntimeError, match="requirements-audio2score.txt"):
        runtime.require_audio2score_runtime()
