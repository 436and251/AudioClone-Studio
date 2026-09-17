from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path


def default_config_path() -> Path:
    base = os.getenv("APPDATA")
    if base:
        return Path(base) / "VoiceDatasetBuilder" / "config.json"
    return Path.home() / ".config" / "VoiceDatasetBuilder" / "config.json"


def default_model_root() -> Path:
    base = os.getenv("LOCALAPPDATA")
    if base:
        return Path(base) / "VoiceDatasetBuilder" / "models"
    return Path.home() / ".cache" / "VoiceDatasetBuilder" / "models"


def default_output_root() -> Path:
    return Path.home() / "VoiceDatasetBuilder" / "datasets"


@dataclass(frozen=True)
class TrainingModuleSetting:
    name: str
    project_root: Path
    python_executable: Path
    module_name: str


@dataclass(frozen=True)
class AppSettings:
    first_run_completed: bool = False
    model_root: Path = default_model_root()
    output_root: Path = default_output_root()
    preferred_asr_model: str = "large-v3-turbo"
    locale: str = "en"
    training_modules: tuple[TrainingModuleSetting, ...] = ()

    @classmethod
    def load(cls, path: Path | None = None) -> "AppSettings":
        path = path or default_config_path()
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return cls(
                first_run_completed=bool(data.get("first_run_completed", False)),
                model_root=Path(data.get("model_root") or default_model_root()),
                output_root=Path(data.get("output_root") or default_output_root()),
                preferred_asr_model=str(data.get("preferred_asr_model") or "large-v3-turbo"),
                locale=data.get("locale") if data.get("locale") in {"zh_CN", "en", "ja"} else "en",
                training_modules=_training_modules(data.get("training_modules")),
            )
        except (OSError, ValueError, TypeError):
            return cls()

    def save(self, path: Path | None = None) -> None:
        path = path or default_config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        payload["model_root"] = str(self.model_root)
        payload["output_root"] = str(self.output_root)
        payload["training_modules"] = [
            {
                "name": module.name,
                "project_root": str(module.project_root),
                "python_executable": str(module.python_executable),
                "module_name": module.module_name,
            }
            for module in self.training_modules
        ]
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(path)


def _training_modules(value: object) -> tuple[TrainingModuleSetting, ...]:
    if not isinstance(value, list):
        return ()
    modules = []
    fields = ("name", "project_root", "python_executable", "module_name")
    for entry in value:
        if not isinstance(entry, dict) or any(
            not isinstance(entry.get(field), str) or not entry[field].strip()
            for field in fields
        ):
            continue
        modules.append(
            TrainingModuleSetting(
                name=entry["name"].strip(),
                project_root=Path(entry["project_root"]),
                python_executable=Path(entry["python_executable"]),
                module_name=entry["module_name"].strip(),
            )
        )
    return tuple(modules)



def normalize_model_root(path: Path) -> Path:
    """Normalize a user-selected HF cache path to the app model storage root."""
    path = Path(path).expanduser()
    name = path.name.lower()
    if name.startswith("models--") and path.parent.name.lower() == "hub" and path.parent.parent.name.lower() == "huggingface":
        return path.parent.parent.parent
    if name == "hub" and path.parent.name.lower() == "huggingface":
        return path.parent.parent
    if name == "huggingface":
        return path.parent
    return path


def model_cache_paths(root: Path) -> tuple[Path, Path]:
    root = normalize_model_root(root)
    return root / "huggingface", root / "torch"


def should_update_current_output(current: Path, old_default: Path) -> bool:
    return Path(current).expanduser() == Path(old_default).expanduser()

def apply_model_environment(root: Path) -> None:
    root = normalize_model_root(Path(root)).resolve()
    hf, torch = model_cache_paths(root)
    hf.mkdir(parents=True, exist_ok=True)
    torch.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf)
    os.environ["TORCH_HOME"] = str(torch)
