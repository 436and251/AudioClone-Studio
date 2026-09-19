from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QDialog

from .first_run import FirstRunDialog
from .main_window import MainWindow
from .settings import AppSettings, TrainingModuleSetting, apply_model_environment
from .studio_window import StudioWindow
from .styles import APP_QSS
from .wheel_guard import install_wheel_guard
from ..training_modules.models import ModuleDescriptor, ProbeResult
from ..training_modules.probe import probe_module


def resource_path(relative_path: str) -> Path:
    return Path(__file__).resolve().parents[2] / relative_path


def set_windows_app_id() -> None:
    if sys.platform == "win32":
        from ctypes import windll

        windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "AudioCloneStudio.Source"
        )


def create_application(argv=None) -> QApplication:
    set_windows_app_id()
    app = QApplication.instance() or QApplication(argv or sys.argv)

    app.setApplicationName("Voice Dataset Builder")
    app.setStyleSheet(APP_QSS)
    install_wheel_guard(app)

    icon_path = resource_path("assets/app.ico")
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    return app


def discover_available_modules(
    settings: AppSettings,
    *,
    probe: Callable[[TrainingModuleSetting], ProbeResult] = probe_module,
) -> tuple[ModuleDescriptor, ...]:
    return tuple(
        descriptor
        for _, descriptor in discover_available_module_bindings(settings, probe=probe)
    )


def discover_available_module_bindings(
    settings: AppSettings,
    *,
    probe: Callable[[TrainingModuleSetting], ProbeResult] = probe_module,
):
    if not settings.training_modules:
        return ()
    with ThreadPoolExecutor(max_workers=min(4, len(settings.training_modules))) as pool:
        results = tuple(pool.map(probe, settings.training_modules))
    return tuple(
        (setting, result.descriptor)
        for setting, result in zip(settings.training_modules, results)
        if result.available and result.descriptor is not None
    )


def choose_window(settings: AppSettings, available_modules: Sequence[ModuleDescriptor]):
    if available_modules:
        return StudioWindow(settings, available_modules)
    return MainWindow(settings)


def main(argv=None) -> int:
    app = create_application(argv)

    settings = AppSettings.load()
    apply_model_environment(settings.model_root)

    if not settings.first_run_completed:
        dialog = FirstRunDialog(settings)

        if dialog.exec() != QDialog.Accepted:
            return 0

        settings = dialog.result_settings()
        settings.save()

        apply_model_environment(settings.model_root)

    modules = discover_available_module_bindings(settings)
    window = choose_window(settings, modules)
    app.setApplicationName(window.windowTitle())

    icon_path = resource_path("assets/app.ico")
    icon = QIcon(str(icon_path))

    if not icon.isNull():
        window.setWindowIcon(icon)

    window.show()

    return app.exec()
