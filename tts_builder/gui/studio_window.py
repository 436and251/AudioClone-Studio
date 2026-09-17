from __future__ import annotations

from collections.abc import Sequence

from PySide6.QtGui import QResizeEvent
from PySide6.QtWidgets import (
    QApplication, QHBoxLayout, QLabel, QMainWindow, QScrollArea,
    QStackedWidget, QVBoxLayout, QWidget,
)

from ..training_modules.models import ModuleDescriptor
from .controller import TaskController
from .dataset_page import DatasetPage
from .i18n import LocaleController, Translator
from .navigation import Navigation
from .settings import AppSettings


class StudioWindow(QMainWindow):
    def __init__(self, settings: AppSettings, modules: Sequence[ModuleDescriptor],
                 dataset_controller: TaskController | None = None):
        super().__init__()
        self.modules = tuple(modules)
        self.locale_controller = LocaleController(settings.locale, self)
        controller = dataset_controller if dataset_controller is not None else TaskController(self)

        root = QWidget()
        layout = QHBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.navigation = Navigation(self.locale_controller)
        self.pages = QStackedWidget()
        self.dataset_page = DatasetPage(
            settings, controller, locale_controller=self.locale_controller
        )
        self.training_page = self._training_page()
        self.pages.addWidget(self._scroll(self.dataset_page))
        self.pages.addWidget(self._scroll(self.training_page))
        self.navigation.selection_changed.connect(self.pages.setCurrentIndex)
        layout.addWidget(self.navigation)
        layout.addWidget(self.pages, 1)
        self.setCentralWidget(root)

        self.locale_controller.locale_changed.connect(self._locale_changed)
        self._locale_changed(self.locale_controller.locale)
        self._fit_to_screen()

    def _training_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(30, 24, 30, 24)
        self.training_placeholder = QLabel()
        self.training_placeholder.setObjectName("Subtitle")
        layout.addWidget(self.training_placeholder)
        layout.addStretch(1)
        return page

    @staticmethod
    def _scroll(page: QWidget) -> QScrollArea:
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setWidget(page)
        return area

    def _locale_changed(self, locale: str) -> None:
        self.setWindowTitle("AudioClone Studio")
        self.training_placeholder.setText(
            Translator(locale).text("training.placeholder")
        )

    def _fit_to_screen(self) -> None:
        available = QApplication.primaryScreen().availableGeometry()
        self.setMinimumSize(min(640, available.width()), min(480, available.height()))
        width = min(1180, max(640, int(available.width() * 0.9)), available.width())
        height = min(820, max(560, int(available.height() * 0.9)), available.height())
        self.resize(width, height)

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self.navigation.set_compact(self.width() < 900)
