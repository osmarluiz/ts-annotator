"""ClassifyPanel — the 'Classify' tab: run the active model over the shared work area.

A dumb view: one classify button (the WORK AREA comes from the MapToolbar scope
selector; the controller echoes it on the button text) + the batch-job progress UI
(job_started / job_progress / job_finished, cancelRequested) + overlay opacity/clear.
Small areas run synchronously; large ones run as a resumable background tile job —
that decision lives in the controller, not here.
"""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QButtonGroup,
    QLabel,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)
import qtawesome as qta

from ts_annotator.app.theme import ICON


class ClassifyPanel(QWidget):
    classifyRequested = pyqtSignal()
    cancelRequested = pyqtSignal()
    scopeChosen = pyqtSignal(str)   # onde classificar: all | viewport | rect | grids

    def __init__(self):
        super().__init__()
        L = QVBoxLayout(self)
        L.setContentsMargins(12, 12, 12, 12)
        L.setSpacing(8)
        L.addWidget(QLabel("<b>Classificar — modelo ativo</b>"))
        self.model_note = QLabel("modelo: (nenhum)")
        self.model_note.setStyleSheet("color:#888;")
        L.addWidget(self.model_note)
        # ONDE grava: um raster por modelo — sem mistério de destino
        self.target = QLabel("grava em: —")
        self.target.setStyleSheet("color:#888;")
        self.target.setToolTip("cada modelo tem SEU raster em predictions/ — "
                               "áreas classificadas se acumulam nele, nada é refeito")
        L.addWidget(self.target)
        # quanto da imagem o raster deste modelo já cobre (lido do sidecar)
        self.coverage = QLabel("cobertura: —")
        self.coverage.setStyleSheet("color:#c8c8d0; font-weight:bold;")
        L.addWidget(self.coverage)
        L.addWidget(QLabel("<b>Onde classificar</b>"))
        self._scope_group = QButtonGroup(self)
        self._scope_btns = {}
        for key, label in [("all", "imagem toda"),
                           ("viewport", "vista atual"),
                           ("rect", "desenhar retângulo"),
                           ("grids", "células da grade (clique p/ selecionar)")]:
            rb = QRadioButton(label)
            rb.toggled.connect(lambda on, k=key: on and self.scopeChosen.emit(k))
            self._scope_group.addButton(rb)
            self._scope_btns[key] = rb
            L.addWidget(rb)
        self._scope_btns["viewport"].setChecked(True)
        note = QLabel("área grande roda em background (tiles, retomável)")
        note.setStyleSheet("color:#888;")
        note.setWordWrap(True)
        L.addWidget(note)

        self.btn = QPushButton("  classificar")
        self.btn.setObjectName("primary")
        self.btn.setIcon(qta.icon("mdi6.map-check-outline", color="#ffffff"))
        self.btn.setToolTip("classifica a área de trabalho em full-res; grava em "
                            "predictions/pred_<modelo>.tif (acumula no mesmo raster)")
        self.btn.clicked.connect(self.classifyRequested)
        L.addWidget(self.btn)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setVisible(False)
        L.addWidget(self.progress)

        self.cancel_btn = QPushButton("  cancelar job (retomável)")
        self.cancel_btn.setIcon(qta.icon("mdi6.stop-circle-outline", color=ICON))
        self.cancel_btn.setVisible(False)
        self.cancel_btn.clicked.connect(self.cancelRequested)
        L.addWidget(self.cancel_btn)

        self.status = QLabel("classifica a área de trabalho com o modelo ativo")
        self.status.setStyleSheet("color:#bbb;")
        self.status.setWordWrap(True)
        L.addWidget(self.status)
        L.addStretch()

    def set_status(self, text):
        self.status.setText(text)

    def set_model(self, name):
        self.model_note.setText(f"modelo: {name}")

    def set_coverage(self, text):
        self.coverage.setText(f"cobertura: {text}")

    def set_target(self, filename):
        self.target.setText(f"grava em: {filename}")

    def set_scope_text(self, label):
        self.btn.setText(f"  classificar ({label})")

    def set_scope_selected(self, key):
        """Reflete o escopo escolhido em outro lugar (ex.: chip ÁREA) sem re-emitir."""
        b = self._scope_btns.get(key)
        if b is not None and not b.isChecked():
            b.blockSignals(True)
            b.setChecked(True)
            b.blockSignals(False)

    # ---- job de lote (área grande / imagem toda): estado visual ----
    def job_started(self):
        self.btn.setEnabled(False)
        self.progress.setValue(0)
        self.progress.setVisible(True)
        self.cancel_btn.setVisible(True)

    def job_progress(self, msg, pct):
        self.status.setText(msg)
        self.progress.setValue(pct)

    def job_finished(self):
        self.btn.setEnabled(True)
        self.progress.setVisible(False)
        self.cancel_btn.setVisible(False)
