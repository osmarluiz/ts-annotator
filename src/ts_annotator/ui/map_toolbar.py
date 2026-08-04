"""MapToolbar — session strip + the map's category chips.

Two pieces with distinct roles:

1. **Session strip** (this widget; fixed, full-width, ABOVE the map): the two
   costly-if-wrong states — dataset and active model — always visible, plus
   the nav toggle.
2. **Category chips** (via :class:`ts_annotator.ui.layers_panel.LayersPanel`):
   OBSERVAÇÃO · RESULTADOS · PONTOS · VETORIAIS · ÁREA, always visible at the
   map's top-left, each showing its current state and opening its dropdown.

Still a dumb view with the SAME signals and attribute names as the historical
toolbar (dataset_combo, combo, ov_combo, ov_sld, model_combo, scope_combo,
draw_btn, gcb, gcols, sld, panel_pin), so the controller and the smokes don't
change: the base/overlay/scope SOURCE OF TRUTH lives in hidden combos here,
driven by and mirrored into the chips' dropdowns.
"""

from PyQt6.QtCore import QTimer, pyqtSignal
from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QToolButton, QWidget
import qtawesome as qta

from ts_annotator.app.theme import ICON


class MapToolbar(QWidget):
    datasetChanged = pyqtSignal()
    basemapChanged = pyqtSignal(str)
    overlayChanged = pyqtSignal(str)
    overlayAlphaChanged = pyqtSignal(int)
    layerToggled = pyqtSignal(str, bool)
    gridToggled = pyqtSignal(bool)
    gridColsChanged = pyqtSignal(int)
    pointSizeChanged = pyqtSignal(int)
    navToggled = pyqtSignal(bool)
    modelChanged = pyqtSignal()
    scopeChanged = pyqtSignal(str)   # área de trabalho: viewport | rect | grids | all
    drawToggled = pyqtSignal(bool)   # modo desenhar-retângulo
    tableRequested = pyqtSignal()    # abrir a tabela de pontos (janela ao vivo)
    patternsRequested = pyqtSignal() # abrir a janela de descoberta de padrões

    def __init__(self, view, basemap_groups, overlay_labels, vlayer_names, init_label,
                 grid_cols, dataset_names, dataset_current, classes=None):
        super().__init__()
        self.view = view

        def _cap(combo, w, popup=280):
            combo.setMaximumWidth(w)
            combo.view().setMinimumWidth(popup)
            combo.currentTextChanged.connect(combo.setToolTip)

        # ================= faixa de sessão (no layout da janela, acima do mapa)
        self.setObjectName("sessionbar")
        self.setStyleSheet("#sessionbar{background:#141418; border-bottom:1px solid #2a2a2e;}"
                           "#sessionbar QLabel{color:#8a8a92;}")
        bar = QHBoxLayout(self)
        bar.setContentsMargins(10, 3, 10, 3)
        bar.setSpacing(8)

        bar.addWidget(QLabel("dataset:"))
        self.dataset_combo = QComboBox()
        self.dataset_combo.setToolTip("dataset de trabalho (pontos rotulados) — editável e treinável")
        for _dn in dataset_names:
            self.dataset_combo.addItem(_dn)
        _di = self.dataset_combo.findText(dataset_current)
        if _di >= 0:
            self.dataset_combo.setCurrentIndex(_di)
        self.dataset_combo.currentIndexChanged.connect(lambda _: self.datasetChanged.emit())
        _cap(self.dataset_combo, 190)
        bar.addWidget(self.dataset_combo)

        bar.addSpacing(14)
        bar.addWidget(QLabel("modelo:"))
        self.model_combo = QComboBox()
        self.model_combo.setToolTip("modelo ativo — proposta · cartões · predição · classificação")
        self.model_combo.currentIndexChanged.connect(lambda _: self.modelChanged.emit())
        _cap(self.model_combo, 280, popup=340)
        bar.addWidget(self.model_combo)
        bar.addStretch()

        self.table_btn = QToolButton()
        self.table_btn.setIcon(qta.icon("mdi6.table", color=ICON))
        self.table_btn.setToolTip("tabela de pontos rotulados — janela ao vivo (rotulou, apareceu)")
        self.table_btn.clicked.connect(self.tableRequested)
        bar.addWidget(self.table_btn)

        self.patterns_btn = QToolButton()
        self.patterns_btn.setIcon(qta.icon("mdi6.shape-outline", color=ICON))
        self.patterns_btn.setToolTip("descobrir padrões — janela de clustering não-supervisionado")
        self.patterns_btn.clicked.connect(self.patternsRequested)
        bar.addWidget(self.patterns_btn)

        self.panel_pin = QToolButton()
        self.panel_pin.setIcon(qta.icon("mdi6.menu", color=ICON))
        self.panel_pin.setCheckable(True)
        self.panel_pin.setChecked(True)
        self.panel_pin.setToolTip("mostrar/ocultar o painel de navegação")
        self.panel_pin.toggled.connect(self.navToggled)
        bar.addWidget(self.panel_pin)

        # ================= ESTADO oculto (fonte da verdade; API histórica)
        self.combo = QComboBox(self)
        _cmodel = self.combo.model()
        for _gi, (_gname, _labels) in enumerate(basemap_groups):
            if _gi > 0:
                self.combo.insertSeparator(self.combo.count())
            self.combo.addItem(f"— {_gname} —")
            _cmodel.item(self.combo.count() - 1).setEnabled(False)
            for _lb in _labels:
                self.combo.addItem(_lb)
        _bidx = self.combo.findText(init_label)
        if _bidx >= 0:
            self.combo.setCurrentIndex(_bidx)
        self.combo.hide()

        self.ov_combo = QComboBox(self)
        self.ov_combo.addItem("(nenhum)")
        for _ovl in overlay_labels:
            self.ov_combo.addItem(_ovl)
        self.ov_combo.hide()

        self.scope_combo = QComboBox(self)
        self.scope_combo.addItem("vista atual", "viewport")
        self.scope_combo.addItem("retângulo", "rect")
        self.scope_combo.addItem("células da grade", "grids")
        self.scope_combo.addItem("imagem toda", "all")
        self.scope_combo.hide()

        # ================= chips de categoria (LayersPanel flutua no mapa)
        from ts_annotator.app.config import PRED_LABEL
        from ts_annotator.ui.layers_panel import LayersPanel
        obs_labels, result_labels = [], []
        for _gname, _labels in basemap_groups:
            dst = result_labels if _gname.lower().startswith("result") else obs_labels
            dst += [l for l in _labels if l != PRED_LABEL]
        self.pop = LayersPanel(view, obs_labels, result_labels, PRED_LABEL,
                               vlayer_names, classes or {}, init_label, grid_cols)

        # painel → estado oculto; estado → sinais do app + espelho no painel
        self.pop.baseSelected.connect(self.combo.setCurrentText)
        self.combo.currentTextChanged.connect(self.basemapChanged)
        self.combo.currentTextChanged.connect(self.pop.sync_base)
        def _ov_sel(l):
            txt = l if l else "(nenhum)"
            # se o combo JÁ está nesse texto (ex.: auto-mostrar da classificação
            # setou antes), setCurrentText é no-op e currentTextChanged NÃO dispara
            # → o "sobrepor" ficava inerte. Força o overlayChanged nesse caso.
            if self.ov_combo.currentText() == txt:
                self.overlayChanged.emit(txt)
            else:
                self.ov_combo.setCurrentText(txt)
        self.pop.overlaySelected.connect(_ov_sel)
        self.ov_combo.currentTextChanged.connect(self.overlayChanged)
        self.ov_combo.currentTextChanged.connect(
            lambda t: self.pop.sync_overlay(None if t == "(nenhum)" else t))
        self.pop.scopeSelected.connect(self.set_scope)
        self.scope_combo.currentIndexChanged.connect(
            lambda _: self.scopeChanged.emit(self.scope_combo.currentData()))
        self.pop.overlayAlphaChanged.connect(self.overlayAlphaChanged)
        self.pop.pointSizeChanged.connect(self.pointSizeChanged)
        self.pop.vectorToggled.connect(self.layerToggled)
        self.pop.drawToggled.connect(self.drawToggled)
        self.pop.gridToggled.connect(self.gridToggled)
        self.pop.gridColsChanged.connect(self.gridColsChanged)

        # aliases de compat (controller/smokes)
        self.draw_btn = self.pop.draw_btn
        self.gcb = self.pop.gcb
        self.gcols = self.pop.gcols
        self.sld = self.pop.size_sld
        self.ov_sld = self.pop.ov_sld
        self.help_btn = self.pop.help_btn

        self._orig_resize = view.resizeEvent
        view.resizeEvent = self._on_resize
        QTimer.singleShot(0, self.reposition)

    # ---------- API usada pelo controller ----------
    def set_grid_checked(self, on):
        self.gcb.setChecked(on)

    def set_scope(self, data):
        i = self.scope_combo.findData(data)
        if i >= 0:
            self.scope_combo.setCurrentIndex(i)

    def reposition(self):
        self.pop.reposition()

    def _on_resize(self, ev):
        self._orig_resize(ev)
        self.reposition()
