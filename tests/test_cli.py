import subprocess
import sys

from tts_builder.cli import build_parser


def test_cli_defaults_are_simple_and_multilingual():
    parser = build_parser()
    args = parser.parse_args(["audio.m4a", "--speaker", "target"])
    assert args.sources == ["audio.m4a"]
    assert args.speaker == "target"
    assert args.language == "auto"
    assert args.asr_model == "large-v3-turbo"
    assert args.output == "dataset_out"
    assert args.skip_separation is False


def test_cli_accepts_multiple_sources_and_explicit_language():
    parser = build_parser()
    args = parser.parse_args([
        "a.wav", "https://youtu.be/example",
        "--speaker", "target", "--language", "JA",
    ])
    assert len(args.sources) == 2
    assert args.language == "JA"


def test_cli_supports_fresh_rebuild_flag():
    parser = build_parser()
    args = parser.parse_args(["audio.m4a", "--speaker", "target", "--fresh"])
    assert args.fresh is True


def test_cli_does_not_import_pyside6():
    code = (
        "import sys; import tts_builder.cli; "
        "print(any(name == 'PySide6' or name.startswith('PySide6.') for name in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False"
