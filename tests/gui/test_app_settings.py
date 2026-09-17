from pathlib import Path
from tts_builder.gui.settings import (
    AppSettings,
    TrainingModuleSetting,
    apply_model_environment,
)


def test_settings_round_trip(tmp_path: Path):
    path = tmp_path / 'config.json'
    settings = AppSettings(
        first_run_completed=True,
        model_root=tmp_path / 'models',
        output_root=tmp_path / 'datasets',
        preferred_asr_model='large-v3-turbo',
        locale='ja',
        training_modules=(
            TrainingModuleSetting(
                name='GPT-SoVITS',
                project_root=tmp_path / 'voice-pipeline',
                python_executable=tmp_path / 'venv' / 'python.exe',
                module_name='voice_pipeline',
            ),
        ),
    )
    settings.save(path)
    loaded = AppSettings.load(path)
    assert loaded == settings


def test_old_settings_load_without_training_configuration(tmp_path: Path):
    path = tmp_path / 'config.json'
    path.write_text(
        '{"first_run_completed":true,"preferred_asr_model":"small"}',
        encoding='utf-8',
    )

    loaded = AppSettings.load(path)

    assert loaded.locale == 'en'
    assert loaded.training_modules == ()
    assert loaded.preferred_asr_model == 'small'


def test_invalid_optional_module_does_not_reset_legacy_settings(tmp_path: Path):
    path = tmp_path / 'config.json'
    path.write_text(
        '{'
        '"first_run_completed":true,'
        '"preferred_asr_model":"small",'
        '"locale":"zh_CN",'
        '"training_modules":['
        '{"name":"broken","project_root":12,"python_executable":null,"module_name":""},'
        '{"name":"GPT-SoVITS","project_root":"D:/voice-pipeline",'
        '"python_executable":"D:/venv/python.exe","module_name":"voice_pipeline"}'
        ']'
        '}',
        encoding='utf-8',
    )

    loaded = AppSettings.load(path)

    assert loaded.first_run_completed is True
    assert loaded.preferred_asr_model == 'small'
    assert loaded.locale == 'zh_CN'
    assert loaded.training_modules == (
        TrainingModuleSetting(
            name='GPT-SoVITS',
            project_root=Path('D:/voice-pipeline'),
            python_executable=Path('D:/venv/python.exe'),
            module_name='voice_pipeline',
        ),
    )


def test_apply_model_environment_sets_separate_caches(tmp_path: Path, monkeypatch):
    root = tmp_path / 'models'
    monkeypatch.delenv('HF_HOME', raising=False)
    monkeypatch.delenv('TORCH_HOME', raising=False)
    apply_model_environment(root)
    import os
    assert os.environ['HF_HOME'] == str(root / 'huggingface')
    assert os.environ['TORCH_HOME'] == str(root / 'torch')

from tts_builder.gui.settings import normalize_model_root, should_update_current_output


def test_normalize_model_root_accepts_huggingface_cache_levels(tmp_path: Path):
    root = tmp_path / 'AI-cache'
    assert normalize_model_root(root) == root
    assert normalize_model_root(root / 'huggingface') == root
    assert normalize_model_root(root / 'huggingface' / 'hub') == root
    repo = root / 'huggingface' / 'hub' / 'models--org--model'
    assert normalize_model_root(repo) == root


def test_default_output_change_does_not_override_task_specific_output(tmp_path: Path):
    old_default = tmp_path / 'default-old'
    new_default = tmp_path / 'default-new'
    current_task = tmp_path / 'speaker-acane'
    assert not should_update_current_output(current_task, old_default)
    assert should_update_current_output(old_default, old_default)
