from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from PySide6.QtGui import QResizeEvent
from PySide6.QtWidgets import (
    QApplication, QHBoxLayout, QMainWindow, QScrollArea, QStackedWidget, QWidget,
)

from ..training_modules.models import ModuleDescriptor
from .controller import TaskController
from .dataset_page import DatasetPage
from .i18n import LocaleController
from .navigation import Navigation
from .settings import AppSettings
from .training_page import TrainingPage


class StudioWindow(QMainWindow):
    def __init__(self, settings: AppSettings, modules: Sequence[object],
                 dataset_controller: TaskController | None = None):
        super().__init__()
        self.bindings = _module_bindings(settings, modules)
        self.modules = tuple(module for _, module in self.bindings)
        self.locale_controller = LocaleController(settings.locale, self)
        controller = dataset_controller if dataset_controller is not None else TaskController(self)

        root = QWidget()
        layout = QHBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.navigation = Navigation(self.locale_controller)
        self.pages = QStackedWidget()
        self.dataset_page = DatasetPage(
            settings,
            controller,
            locale_controller=self.locale_controller,
            training_available=True,
        )
        self.training_page = TrainingPage(self.bindings, self.locale_controller)
        self.pages.addWidget(self._scroll(self.dataset_page))
        self.pages.addWidget(self._scroll(self.training_page))
        self.navigation.selection_changed.connect(self.pages.setCurrentIndex)
        self.dataset_page.training_requested.connect(self._continue_training)
        layout.addWidget(self.navigation)
        layout.addWidget(self.pages, 1)
        self.setCentralWidget(root)

        self.locale_controller.locale_changed.connect(self._locale_changed)
        self._locale_changed(self.locale_controller.locale)
        self._fit_to_screen()

    @staticmethod
    def _scroll(page: QWidget) -> QScrollArea:
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setWidget(page)
        return area

    def _locale_changed(self, locale: str) -> None:
        self.setWindowTitle("AudioClone Studio")

    def _continue_training(self, dataset: str) -> None:
        self.training_page.prefill_dataset(Path(dataset))
        self.navigation.set_current(1)

    def _fit_to_screen(self) -> None:
        available = QApplication.primaryScreen().availableGeometry()
        self.setMinimumSize(min(640, available.width()), min(480, available.height()))
        width = min(1440, available.width())
        height = min(900, available.height())
        self.resize(width, height)

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self.navigation.set_compact(self.width() < 900)


def _module_bindings(settings: AppSettings, modules: Sequence[object]):
    bindings = []
    for index, item in enumerate(modules):
        if (
            isinstance(item, tuple)
            and len(item) == 2
            and isinstance(item[1], ModuleDescriptor)
        ):
            bindings.append(item)
        elif isinstance(item, ModuleDescriptor):
            setting = (
                settings.training_modules[index]
                if index < len(settings.training_modules)
                else None
            )
            bindings.append((setting, item))
    return tuple(bindings)
