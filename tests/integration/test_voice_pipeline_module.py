import os
from pathlib import Path

import pytest

from tts_builder.gui.settings import TrainingModuleSetting
from tts_builder.training_modules.probe import probe_module


def test_real_voice_pipeline_descriptor_handshake():
    python_value = os.getenv("VOICE_PIPELINE_PYTHON")
    root_value = os.getenv("VOICE_PIPELINE_ROOT")
    if not python_value or not root_value:
        pytest.skip("voice-pipeline integration environment not configured")

    setting = TrainingModuleSetting(
        name="GPT-SoVITS",
        project_root=Path(root_value),
        python_executable=Path(python_value),
        module_name="voice_pipeline",
    )

    result = probe_module(setting)

    assert result.available, result.error
    assert result.descriptor is not None
    assert result.descriptor.protocol_version == 2
    assert result.descriptor.module_id
    assert result.descriptor.frameworks
    assert result.descriptor.frameworks[0].training_data.kind == "file"
    assert result.descriptor.frameworks[0].training_data.extensions == (".list",)
