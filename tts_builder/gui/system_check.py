from __future__ import annotations

import os
import platform
import shutil
from dataclasses import dataclass
from pathlib import Path

from ..runtime import resolve_binary
from .i18n import Translator


@dataclass(frozen=True)
class HardwareInfo:
    os_name: str
    cpu_name: str
    ram_gb: float
    gpu_name: str | None
    cuda_available: bool
    ffmpeg_available: bool
    free_gb: float


@dataclass(frozen=True)
class Recommendation:
    mode: str
    asr_model: str
    message: str


def recommendation_for(info: HardwareInfo, translator: Translator | None = None) -> Recommendation:
    translator = translator or Translator("en")
    if info.cuda_available:
        return Recommendation("cuda", "large-v3-turbo", translator.text("system.cuda_available"))
    return Recommendation("cpu", "small", translator.text("system.cpu_available"))


def detect_system(model_root: Path, translator: Translator | None = None) -> HardwareInfo:
    translator = translator or Translator("en")
    root = Path(model_root).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(root).free / (1024 ** 3)
    ram_gb = _memory_gb()
    gpu_name, cuda = _cuda_info()
    return HardwareInfo(
        os_name=platform.system(),
        cpu_name=platform.processor() or translator.text("system.unknown_cpu"),
        ram_gb=ram_gb,
        gpu_name=gpu_name,
        cuda_available=cuda,
        ffmpeg_available=_ffmpeg_available(),
        free_gb=free_gb,
    )


def _memory_gb() -> float:
    try:
        import psutil
        return round(psutil.virtual_memory().total / (1024 ** 3), 1)
    except Exception:
        if os.name == "posix" and hasattr(os, "sysconf"):
            try:
                return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024 ** 3), 1)
            except Exception:
                pass
    return 0.0


def _cuda_info() -> tuple[str | None, bool]:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0), True
    except Exception:
        pass
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            return "NVIDIA CUDA GPU", True
    except Exception:
        pass
    return None, False


def _ffmpeg_available() -> bool:
    try:
        resolve_binary("ffmpeg")
        return True
    except Exception:
        return False
