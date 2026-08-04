"""Dark theme: icon color, fallback QSS, accent QSS, and an apply helper."""

ICON = "#cfd3dc"  # icon tint

# Fallback stylesheet used only if qdarktheme is unavailable.
APP_QSS = """
QWidget { background:#1c1c1f; color:#e6e6e6; font-size:13px; }
QLabel { background:transparent; }
QPushButton { background:#333338; border:none; border-radius:6px; padding:6px; color:#eee; }
QPushButton:hover { background:#3d3d44; }
QCheckBox { spacing:6px; background:transparent; }
QSlider::groove:horizontal { height:5px; background:#3a3a40; border-radius:2px; }
QSlider::handle:horizontal { background:#00e676; width:14px; border-radius:7px; margin:-5px 0; }
QSplitter::handle { background:#2a2a2e; }
QSplitter::handle:hover { background:#00e676; }
QComboBox { background:#333338; border-radius:6px; padding:4px 8px; }
QComboBox QAbstractItemView { background:#2b2b2e; selection-background-color:#00695c; }
QScrollBar:vertical { background:#1c1c1f; width:10px; }
QScrollBar::handle:vertical { background:#444; border-radius:5px; }
QTabWidget::pane { border:1px solid #2a2a2e; background:#1c1c1f; border-radius:6px; }
QTabBar::tab { background:#2b2b2e; color:#bbb; padding:6px 14px; border-radius:6px; margin:2px; }
QTabBar::tab:selected { background:#00695c; color:#fff; }
QTabBar::tab:hover { background:#3a3a40; }
QToolButton { background:#333338; border:none; border-radius:6px; padding:4px; }
QToolButton:checked { background:#00695c; }
QListWidget { background:#202024; border:none; border-radius:6px; }
"""

# Accent overrides layered on top of qdarktheme (or APP_QSS in the fallback).
ACCENT_QSS = """
#floatbar { background: rgba(32,33,40,236); border-radius: 10px; }
#navbtn { text-align: left; border: none; border-radius: 8px; padding: 9px 12px; color: #cfd3dc; }
#navbtn:hover { background: #2a2c34; }
#navbtn:checked { background: #2b6cb0; color: #fff; }
QToolButton { border: none; border-radius: 6px; padding: 5px; }
QToolButton:checked { background: #2b6cb0; }
QListWidget { background: #1a1b20; border: 1px solid #2a2c34; border-radius: 8px; padding: 3px; }
QListWidget::item { padding: 3px 5px; border-radius: 4px; }
QListWidget::item:selected { background: #2b6cb0; }
QPushButton#primary { background: #2b6cb0; color: white; font-weight: bold; border-radius: 6px; padding: 7px; }
QPushButton#primary:hover { background: #3182ce; }
QPushButton#primary:disabled { background: #2a2c34; color: #6b6e78; }
QTableView { background: #141518; alternate-background-color: #1a1b20;
             selection-background-color: #2b6cb0; border: 1px solid #2a2c34; border-radius: 6px; }
QHeaderView::section { background: #1f2026; color: #aab0bc; border: none;
                       border-right: 1px solid #2a2c34; padding: 4px 6px; font-weight: bold; }
"""


def apply_theme(app):
    """Apply the dark theme to a QApplication (qdarktheme, or QSS fallback)."""
    from PyQt6.QtGui import QFont
    app.setFont(QFont("Segoe UI", 10))   # tipografia consistente no Windows
    try:
        import qdarktheme
        qdarktheme.setup_theme("dark", additional_qss=ACCENT_QSS)
    except Exception as e:  # pragma: no cover - depends on optional dep
        import logging
        logging.getLogger(__name__).info("qdarktheme n/d (%s) — usando o QSS embutido", e)
        app.setStyleSheet(APP_QSS + ACCENT_QSS)
