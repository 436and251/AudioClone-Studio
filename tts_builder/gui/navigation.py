from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QButtonGroup, QLabel, QPushButton, QVBoxLayout, QWidget

from .i18n import LocaleController, Translator


class Navigation(QWidget):
    selection_changed = Signal(int)

    def __init__(self, locale_controller: LocaleController, parent=None):
        super().__init__(parent)
        self.locale_controller = locale_controller
        self.translator = Translator(locale_controller.locale)
        self.compact = False
        self.current_index = -1

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 24, 16, 24)
        layout.setSpacing(8)
        self.workspace_label = QLabel()
        self.workspace_label.setObjectName("NavigationTitle")
        layout.addWidget(self.workspace_label)

        self.buttons = [QPushButton(), QPushButton()]
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        for index, button in enumerate(self.buttons):
            button.setObjectName("NavigationButton")
            button.setCheckable(True)
            self._group.addButton(button, index)
            layout.addWidget(button)
        layout.addStretch(1)

        self._group.idClicked.connect(self.set_current)
        locale_controller.locale_changed.connect(self._locale_changed)
        self.retranslate_ui(self.translator)
        self.set_current(0)
        self.set_compact(False)

    def _locale_changed(self, locale: str) -> None:
        self.retranslate_ui(Translator(locale))

    def retranslate_ui(self, translator: Translator) -> None:
        self.translator = translator
        self.workspace_label.setText(translator.text("nav.workspace"))
        labels = (translator.text("nav.dataset"), translator.text("nav.training"))
        icons = ("⌕", "◉")
        for button, label, icon in zip(self.buttons, labels, icons):
            button.setText(icon if self.compact else label)
            button.setToolTip(label)
            button.setAccessibleName(label)

    def set_current(self, index: int) -> None:
        if index == self.current_index:
            return
        self.current_index = index
        for button_index, button in enumerate(self.buttons):
            selected = button_index == index
            button.setChecked(selected)
            button.setProperty("selected", selected)
            button.style().unpolish(button)
            button.style().polish(button)
        self.selection_changed.emit(index)

    def set_compact(self, compact: bool) -> None:
        self.compact = compact
        self.workspace_label.setVisible(not compact)
        self.setFixedWidth(72 if compact else 210)
        self.retranslate_ui(self.translator)
