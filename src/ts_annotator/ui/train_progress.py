"""TrainProgressWindow — a live training panel with Progresso + Resultado tabs.

Progresso (during): a determinate bar with real ETA, the phase line (spatial-CV
fold f/k · epoch e/E), a live loss sparkline (loss per epoch of the current
fit), the per-fold bacc as folds complete, and a collapsible raw log.
Resultado (after): the honest spatial-CV metrics (bacc + delta, macro-F1), F1
per class, cleanlab suspects, and an expandable confusion matrix heatmap.

Dumb view fed by the controller: start() / step(dict) / progress(str) / done().
"""

import time

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


class _ConfusionTable(QTableWidget):
    """Matriz de confusão como heatmap: linha=verdadeiro, coluna=predito."""

    def set_matrix(self, mat, labels):
        mat = np.asarray(mat, float)
        n = len(labels)
        self.setRowCount(n)
        self.setColumnCount(n)
        self.setHorizontalHeaderLabels(labels)
        self.setVerticalHeaderLabels(labels)
        rowsum = mat.sum(1, keepdims=True)
        frac = np.divide(mat, np.where(rowsum == 0, 1, rowsum))   # normaliza por linha
        for i in range(n):
            for j in range(n):
                it = QTableWidgetItem(str(int(mat[i, j])))
                it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                f = frac[i, j]
                # diagonal (acerto) verde; fora (confusão) vermelho — intensidade ∝ fração
                if i == j:
                    it.setBackground(QColor(40, 90, 50, int(40 + 180 * f)))
                elif f > 0:
                    it.setBackground(QColor(120, 40, 40, int(40 + 180 * f)))
                self.setItem(i, j, it)
        self.resizeColumnsToContents()


class TrainProgressWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowTitle("Treino — TSA")
        self.resize(500, 460)
        self._t0 = 0.0
        self._loss_x, self._loss_y = [], []
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 12)
        root.setSpacing(8)

        self.phase = QLabel("preparando…")
        self.phase.setStyleSheet("font-weight:bold; font-size:13px;")
        root.addWidget(self.phase)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setFormat("%p%")
        root.addWidget(self.bar)

        self.tabs = QTabWidget()
        root.addWidget(self.tabs, 1)

        # ---- aba Progresso ----
        prog = QWidget()
        pl = QVBoxLayout(prog)
        pl.setContentsMargins(4, 6, 4, 4)
        self.step_lbl = QLabel("—")
        self.step_lbl.setStyleSheet("color:#c8c8d0;")
        pl.addWidget(self.step_lbl)
        pl.addWidget(QLabel("loss (fit atual)"))
        self.loss_plot = pg.PlotWidget()
        self.loss_plot.setBackground("#141518")
        self.loss_plot.setMaximumHeight(140)
        self.loss_plot.showGrid(x=True, y=True, alpha=0.2)
        self._loss_curve = self.loss_plot.plot(pen=pg.mkPen("#00e676", width=2))
        pl.addWidget(self.loss_plot)
        self.folds_lbl = QLabel("folds: —")
        self.folds_lbl.setStyleSheet("color:#9ccc65;")
        pl.addWidget(self.folds_lbl)
        self._log_btn = QToolButton()
        self._log_btn.setText("▸ log detalhado")
        self._log_btn.setCheckable(True)
        self._log_btn.setStyleSheet("QToolButton{border:none; color:#8a8a92;}")
        self._log_btn.toggled.connect(self._toggle_log)
        pl.addWidget(self._log_btn)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(500)
        self.log.setVisible(False)
        self.log.setStyleSheet("font-family:Consolas,monospace; font-size:11px; "
                               "background:#141518; border:1px solid #2a2c34; border-radius:6px;")
        pl.addWidget(self.log)
        self.tabs.addTab(prog, "Progresso")

        # ---- aba Resultado ----
        resw = QWidget()
        rl = QVBoxLayout(resw)
        rl.setContentsMargins(4, 6, 4, 4)
        self.res_head = QLabel("aguardando o fim do treino…")
        self.res_head.setStyleSheet("font-size:13px;")
        self.res_head.setWordWrap(True)
        rl.addWidget(self.res_head)
        self.f1_lbl = QLabel("")
        self.f1_lbl.setWordWrap(True)
        self.f1_lbl.setStyleSheet("color:#9ccc65;")
        rl.addWidget(self.f1_lbl)
        self._cm_btn = QToolButton()
        self._cm_btn.setText("▸ matriz de confusão")
        self._cm_btn.setCheckable(True)
        self._cm_btn.setStyleSheet("QToolButton{border:none; color:#c8c8d0; font-weight:bold;}")
        self._cm_btn.toggled.connect(self._toggle_cm)
        rl.addWidget(self._cm_btn)
        self._cm_hint = QLabel("linha = classe verdadeira · coluna = predito · diagonal = acerto")
        self._cm_hint.setStyleSheet("color:#777;")
        self._cm_hint.setVisible(False)
        rl.addWidget(self._cm_hint)
        self.confusion = _ConfusionTable()
        self.confusion.setVisible(False)
        rl.addWidget(self.confusion, 1)
        self.tabs.addTab(resw, "Resultado")

        self.close_btn = QPushButton("fechar")
        self.close_btn.clicked.connect(self.hide)
        self.close_btn.setEnabled(False)
        root.addWidget(self.close_btn)

    # ---- controle ----
    def start(self, subtitle=""):
        self._t0 = time.time()
        self._loss_x, self._loss_y = [], []
        self._loss_curve.setData([], [])
        self.log.clear()
        self.phase.setText("treinando…")
        self.step_lbl.setText(subtitle or "—")
        self.folds_lbl.setText("folds: —")
        self.bar.setValue(0)
        self.res_head.setText("aguardando o fim do treino…")
        self.f1_lbl.clear()
        self.confusion.setVisible(False)
        self._cm_btn.setChecked(False)
        self.tabs.setCurrentIndex(0)
        self.close_btn.setEnabled(False)
        self.show()
        self.raise_()
        self.activateWindow()

    def step(self, s):
        frac = s.get("frac")
        if frac is not None:
            self.bar.setValue(int(1000 * min(max(frac, 0.0), 1.0)))
            el = time.time() - self._t0
            eta = (el / frac - el) if frac > 0.02 else 0
            self.phase.setText(f"{s.get('phase','treinando')} · {self._fmt(eta)} restante"
                               if eta else s.get("phase", "treinando"))
        if "epoch" in s:
            fold = f"fold {s['fold']}/{s['k']} · " if "fold" in s else ""
            self.step_lbl.setText(f"{fold}época {s['epoch']}/{s['epochs']}  ·  loss {s['loss']:.3f}")
            self._loss_y.append(s["loss"])
            self._loss_x = list(range(len(self._loss_y)))
            self._loss_curve.setData(self._loss_x, self._loss_y)
        if "fold_bacc" in s:
            cur = self.folds_lbl.text()
            cur = "" if cur.startswith("folds: —") else cur.replace("folds: ", "")
            self.folds_lbl.setText("folds: " + (cur + "  " if cur else "")
                                   + f"[{s['fold']}] {s['fold_bacc']:.2f}")
            self._loss_y = []             # nova curva de loss por fold
            self._loss_curve.setData([], [])

    def progress(self, msg):
        self.log.appendPlainText(msg)

    def done(self, ok, head, f1_html="", confusion=None, labels=None):
        self.bar.setValue(1000)
        self.phase.setText("✔ concluído" if ok else "erro no treino")
        self.res_head.setText(head)
        self.f1_lbl.setText(f1_html)
        if ok and confusion is not None and labels:
            self.confusion.set_matrix(confusion, labels)
        self.tabs.setCurrentIndex(1)      # pula pro Resultado
        self.close_btn.setEnabled(True)

    def _toggle_log(self, on):
        self._log_btn.setText(("▾" if on else "▸") + " log detalhado")
        self.log.setVisible(on)

    def _toggle_cm(self, on):
        self._cm_btn.setText(("▾" if on else "▸") + " matriz de confusão")
        self._cm_hint.setVisible(on)
        self.confusion.setVisible(on)

    @staticmethod
    def _fmt(sec):
        sec = int(sec)
        return f"{sec // 60}m{sec % 60:02d}s" if sec >= 60 else f"{sec}s"
