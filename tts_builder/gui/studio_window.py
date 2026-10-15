from __future__ import annotations

from collections.abc import Sequence
from functools import partial
from pathlib import Path

from PySide6.QtGui import QResizeEvent
from PySide6.QtWidgets import (
    QApplication, QHBoxLayout, QMainWindow, QScrollArea, QStackedWidget, QTabWidget, QWidget,
)

from ..training_modules.models import ModuleDescriptor
from ..training_modules.process import ModuleProcessController
from .controller import TaskController
from .audio2score_page import Audio2ScorePage
from .dataset_page import DatasetPage
from .i18n import LocaleController
from .navigation import Navigation
from .settings import AppSettings, normalize_model_root
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
        self.mining_tabs = QTabWidget()
        self.mining_tabs.addTab(self._scroll(self.dataset_page), "")
        self.audio2score_page = Audio2ScorePage(settings, self.locale_controller)
        self.dataset_page.settings_changed.connect(lambda updated: setattr(self.audio2score_page, "settings", updated))
        self.mining_tabs.addTab(self._scroll(self.audio2score_page), "")
        self.training_page = TrainingPage(
            self.bindings,
            self.locale_controller,
            process_factory=partial(
                ModuleProcessController,
                model_root=normalize_model_root(settings.model_root),
            ),
        )
        self.pages.addWidget(self.mining_tabs)
        self.pages.addWidget(self.training_page)
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
        from .i18n import Translator
        translator = Translator(locale)
        self.mining_tabs.setTabText(0, translator.text("audio2score.tab.dataset"))
        self.mining_tabs.setTabText(1, translator.text("audio2score.tab.score"))

    def _continue_training(self, dataset: str) -> None:
        self.training_page.prefill_dataset(Path(dataset))
        self.training_page.tabs.setCurrentIndex(0)
        self.pages.setCurrentWidget(self.training_page)
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

    def closeEvent(self, event) -> None:
        if not self.audio2score_page.request_close(self.close):
            event.ignore()
            return
        super().closeEvent(event)


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
