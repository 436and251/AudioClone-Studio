from pathlib import Path

from tts_builder.separator import build_demucs_command


def test_demucs_command_writes_only_lossless_vocal_stem():
    command = build_demucs_command(
        Path("input.m4a"), Path("out"), model="htdemucs", device="auto", segment_seconds=7
    )
    assert "--two-stems=vocals" in command
    assert "--other-method" in command
    assert command[command.index("--other-method") + 1] == "none"
    assert "--flac" in command
    assert "-d" not in command


def test_vocal_wrapper_keeps_original_arguments(tmp_path):
    args = build_demucs_command(Path("input.m4a"), tmp_path, "htdemucs", "auto", 7)
    assert "--two-stems=vocals" in args
    assert args[args.index("--other-method") + 1] == "none"
    assert "--flac" in args
