"""Test config: force headless Qt so the suite runs without a display (CI + local).

Set BEFORE any PyQt6/pyqtgraph import happens during collection.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
