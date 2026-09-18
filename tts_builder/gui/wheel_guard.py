from PySide6.QtCore import QEvent, QObject
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import (
    QAbstractScrollArea,
    QAbstractSpinBox,
    QApplication,
    QComboBox,
)


class WheelGuard(QObject):
    def eventFilter(self, watched, event) -> bool:
        if event.type() != QEvent.Wheel or not isinstance(
            watched, (QComboBox, QAbstractSpinBox)
        ):
            return False
        parent = watched.parentWidget()
        while parent is not None and not isinstance(parent, QAbstractScrollArea):
            parent = parent.parentWidget()
        if parent is not None:
            forwarded = QWheelEvent(
                event.position(),
                event.globalPosition(),
                event.pixelDelta(),
                event.angleDelta(),
                event.buttons(),
                event.modifiers(),
                event.phase(),
                event.inverted(),
            )
            QApplication.sendEvent(parent.viewport(), forwarded)
        return True


def install_wheel_guard(app: QApplication) -> None:
    if getattr(app, "_audio_clone_wheel_guard", None) is not None:
        return
    guard = WheelGuard(app)
    app.installEventFilter(guard)
    app._audio_clone_wheel_guard = guard
