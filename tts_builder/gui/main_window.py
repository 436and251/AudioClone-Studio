from __future__ import annotations

from PySide6.QtWidgets import QApplication, QMainWindow, QScrollArea, QTabWidget

from .controller import TaskController
from .audio2score_page import Audio2ScorePage
from .dataset_page import DatasetPage
from .i18n import LocaleController, Translator
from .settings import AppSettings


_COMPAT_WIDGETS = (
    "title_label", "subtitle_label", "settings_btn", "source", "speaker",
    "language", "model", "output_label", "output", "output_browse",
    "status", "open_output", "stop", "start", "processing_label",
    "progress", "logs",
)


class MainWindow(QMainWindow):
    def __init__(self, settings: AppSettings, controller: TaskController | None = None,
                 locale_controller: LocaleController | None = None):
        super().__init__()
        controller = controller if controller is not None else TaskController(self)
        self.dataset_page = DatasetPage(
            settings, controller, self, locale_controller=locale_controller
        )
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.dataset_page)
        self.tabs = QTabWidget()
        self.tabs.addTab(scroll, "")
        self.audio2score_page = Audio2ScorePage(settings, self.dataset_page.locale_controller)
        self.dataset_page.settings_changed.connect(lambda updated: setattr(self.audio2score_page, "settings", updated))
        score_scroll = QScrollArea()
        score_scroll.setWidgetResizable(True)
        score_scroll.setWidget(self.audio2score_page)
        self.tabs.addTab(score_scroll, "")
        self.setCentralWidget(self.tabs)
        self._fit_to_screen()
        self.locale_controller.locale_changed.connect(self._locale_changed)
        self._locale_changed(self.locale_controller.locale)
        for name in _COMPAT_WIDGETS:
            setattr(self, name, getattr(self.dataset_page, name))

    @property
    def settings(self) -> AppSettings:
        return self.dataset_page.settings

    @property
    def controller(self) -> TaskController:
        return self.dataset_page.controller

    @property
    def locale_controller(self) -> LocaleController:
        return self.dataset_page.locale_controller

    @property
    def progress_model(self):
        return self.dataset_page.progress_model

    def _locale_changed(self, locale: str) -> None:
        translator = Translator(locale)
        self.setWindowTitle(translator.text("app.title"))
        self.tabs.setTabText(0, translator.text("audio2score.tab.dataset"))
        self.tabs.setTabText(1, translator.text("audio2score.tab.score"))

    def _set_status(self, key: str, **values: object) -> None:
        self.dataset_page._set_status(key, **values)

    def _fit_to_screen(self) -> None:
        available = QApplication.primaryScreen().availableGeometry()
        self.setMinimumSize(min(760, available.width()), min(650, available.height()))
        width = min(860, max(640, int(available.width() * 0.95)), available.width())
        height = min(760, max(560, int(available.height() * 0.95)), available.height())
        self.resize(width, height)

    def closeEvent(self, event) -> None:
        if not self.audio2score_page.request_close(self.close):
            event.ignore()
            return
        super().closeEvent(event)
