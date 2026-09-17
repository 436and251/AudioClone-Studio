APP_QSS = r"""
* { font-family: "Segoe UI", "Microsoft YaHei UI", sans-serif; font-size: 13px; }
QWidget { background: #121212; color: #FFFFFF; }
QMainWindow, QDialog { background: #121212; }
QLabel#Title { font-size: 28px; font-weight: 700; }
QLabel#Subtitle { color: #B3B3B3; font-size: 14px; }
QLabel#FieldChip {
    background: #242424;
    border: 1px solid #333333;
    border-radius: 10px;
    padding: 10px 12px;
    min-height: 20px;
    color: #FFFFFF;
    font-weight: 500;
}
QFrame#Card { background: #181818; border: 1px solid #282828; border-radius: 14px; }
QLineEdit, QComboBox {
    background: #242424;
    border: 1px solid #333333;
    border-radius: 10px;
    padding: 10px 12px;
    min-height: 20px;
    color: #FFFFFF;
}

QLineEdit:focus, QComboBox:focus {border: 1px solid #1ED760; }

QComboBox { padding: 10px 36px 10px 12px; }

QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 34px;
    border: none;
    background: transparent;
}

QComboBox::down-arrow {
    image: none;
    width: 0;
    height: 0;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid #1ED760;
    margin-right: 12px;
}

QComboBox QAbstractItemView {
    background: #242424;
    color: #FFFFFF;
    border: 1px solid #333333;
    border-radius: 10px;
    padding: 6px;
    outline: 0;
    selection-background-color: #1ED760;
    selection-color: #06130A;
}
QPushButton { background: #2A2A2A; border: none; border-radius: 16px; padding: 9px 16px; font-weight: 600; }
QPushButton:hover { background: #343434; }
QPushButton:disabled { color: #777777; background: #202020; }
QPushButton#Primary { background: #1ED760; color: #06130A; border-radius: 20px; padding: 11px 22px; font-weight: 700; }
QPushButton#Primary:hover { background: #25E46A; }
QPushButton#Danger { color: #FF8A8A; }
QProgressBar { background: #2A2A2A; border: none; border-radius: 3px; height: 6px; text-align: center; color: transparent; }
QProgressBar::chunk { background: #1ED760; border-radius: 3px; }
QPlainTextEdit { background: #101010; border: 1px solid #282828; border-radius: 10px; color: #B3B3B3; padding: 8px; }
QToolButton {
    background: transparent;
    color: #B3B3B3;
    border: none;
    padding: 4px;
}

QToolButton:hover {
    color: #FFFFFF;
}

QToolButton#IconButton {
    background: #2A2A2A;
    color: #FFFFFF;
    border: none;
    border-radius: 22px;
    font-size: 18px;
    font-weight: 700;
    min-width: 44px;
    min-height: 44px;
    max-width: 44px;
    max-height: 44px;
    padding: 0;
}

QToolButton#IconButton:hover {
    background: #343434;
    color: #1ED760;
}

QToolButton#IconButton:pressed {
    background: #1F1F1F;
}

QLabel#NavigationTitle {
    color: #B3B3B3;
    font-size: 12px;
    font-weight: 700;
    padding: 0 8px 8px 8px;
}

QPushButton#NavigationButton {
    background: transparent;
    border: none;
    border-radius: 0;
    border-bottom: 3px solid transparent;
    padding: 12px 10px;
    text-align: left;
}

QPushButton#NavigationButton:hover { background: #1B1B1B; }
QPushButton#NavigationButton[selected="true"] { border-bottom-color: #1ED760; }

QScrollArea { border: none; }
QTabWidget::pane { border: none; }
QTabBar::tab {
    background: transparent;
    border: none;
    border-bottom: 3px solid transparent;
    padding: 10px 14px;
    color: #B3B3B3;
}
QTabBar::tab:hover { color: #FFFFFF; }
QTabBar::tab:selected { color: #FFFFFF; border-bottom-color: #1ED760; }
"""

GREEN = "#1ED760"
MUTED = "#B3B3B3"
FAILED = "#FF6B6B"
PENDING = "#6F6F6F"
