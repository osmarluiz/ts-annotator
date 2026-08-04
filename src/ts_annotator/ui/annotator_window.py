"""AnnotatorWindow — the map-native annotator UI, assembled from an AppContext.

Assembly + wiring ONLY: the window builds the widgets (MapCanvas, MapToolbar and
the four nav panels), connects their signals to the :class:`AnnotatorController`
(``app/controller.py``) — which owns the cross-cutting logic and the shared state —
and lays everything out. For backward compatibility (smokes, external scripts) the
window exposes aliases both to the widgets (``self.slist``, ``self.goal_combo``, …)
and to the controller's handler methods (``self.on_click``, ``self.do_propose``, …).
"""

import qtawesome as qta
from PyQt6.QtCore import QEvent, QSize, Qt, QTimer
from PyQt6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QButtonGroup,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ts_annotator.app.config import PRED_LABEL
from ts_annotator.app.controller import AnnotatorController
from ts_annotator.app.theme import ICON
from ts_annotator.ui.map_toolbar import MapToolbar
from ts_annotator.ui.review_panel import ReviewPanel
from ts_annotator.ui.train_progress import TrainProgressWindow
from ts_annotator.ui.patterns_window import PatternsWindow
from ts_annotator.ui.classify_panel import ClassifyPanel
from ts_annotator.ui.train_panel import TrainPanel
from ts_annotator.ui.annotate_panel import AnnotatePanel
from ts_annotator.ui.map_canvas import MapCanvas

# handlers do controller espelhados como atributos da janela — os builds conectam
# sinais a self.<nome> e código externo (smokes) chama pela janela
_CONTROLLER_METHODS = (
    "on_dataset_change", "toggle_layer", "refresh_overlay", "set_overlay", "set_overlay_alpha",
    "toggle_cell", "refresh_markers", "select_point", "predict_point", "on_click", "on_label",
    "on_class_added", "on_remove", "navigate", "_propose_area", "fill_suggestions", "do_propose",
    "on_sugg_select", "advance_suggestion", "skip_suggestion", "undo_last", "on_hover",
    "do_discover", "discover_split", "discover_enter", "discover_up", "discover_root",
    "apply_patterns", "clear_patterns", "go_to_pattern", "propose_from_pattern", "zoom_to_point_rc",
    "_proc_hover", "_on_basemap", "set_grid", "set_cols", "set_size", "set_draw", "update_goals",
    "on_scope_change", "on_draw_toggled", "update_scope_labels",
    "on_points_visible", "on_class_visible", "on_triage", "on_pts_alpha",
    "set_layer_color", "set_layer_fill", "on_class_color", "on_pred_class_visible", "open_class_manager",
    "refresh_review_list", "on_review_point", "zoom_to_point", "list_model_files", "load_active_model",
    "refresh_models", "on_model_change", "classify_scope", "classify_all",
    "_on_alpha", "do_train",
)


class AnnotatorWindow(QWidget):
    """Top-level annotator widget: nav | (map over curve) | class cards, + floating toolbar."""

    GRID_COLS = 10
    PRED_LABEL = PRED_LABEL  # basemap dinâmico (definição única em app/config.py)

    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        # --- refs que os builds/layout usam ---
        self.view = ctx.view
        self.curve_view = ctx.curve_view
        self.panel = ctx.panel
        self.datasets = ctx.datasets
        self.class_src = ctx.class_src
        self.basemaps = ctx.basemaps
        self.basemap_groups = ctx.basemap_groups
        self.vlayer_paths = ctx.vlayer_paths
        self.classes = ctx.classes

        # --- controller: lógica cruzada + estado compartilhado ---
        self.controller = AnnotatorController(ctx, self)
        self.state = self.controller.state
        self.tstate = self.controller.tstate
        self.pred_state = self.controller.pred_state
        for _m in _CONTROLLER_METHODS:
            setattr(self, _m, getattr(self.controller, _m))

        # camada "Predição (modelo)" entra no grupo Resultado; a FONTE é dinâmica
        # (segue o modelo ativo) e é resolvida em _refresh_pred_basemap
        for _gname, _labels in self.basemap_groups:
            if _gname.startswith("Resultado"):
                if self.PRED_LABEL not in _labels:   # ctx pode ser reusado (2ª janela)
                    _labels.append(self.PRED_LABEL)
                break
        else:
            self.basemap_groups.append(("Resultado (produto)", [self.PRED_LABEL]))

        self.setWindowTitle("TSA — anotador")
        self._build_canvas()
        self._build_toolbar()
        self._build_annotate()
        self._build_review()
        self._build_train()
        self._build_classify()
        self._build_nav()
        self._assemble()
        self._wire()
        self.controller.init_workers()

        QTimer.singleShot(0, self.fit)

    @property
    def store(self):
        """Dataset ativo (muda no dropdown do topo) — fonte da verdade é o controller."""
        return self.controller.store

    # =====================================================================
    # Map canvas (pyqtgraph overlays + grid/draw-rect tools; ver MapCanvas)
    # =====================================================================
    def _build_canvas(self):
        self.canvas = MapCanvas(self.view, self.class_src.width, self.class_src.height,
                                self.GRID_COLS, self.basemaps)
        c = self.canvas
        # aliases: controller e smokes usam os itens direto
        self.markers = c.markers
        self.prop_marker = c.prop_marker
        self.hover_marker = c.hover_marker
        self.cursor_marker = c.cursor_marker
        self.pred_overlay = c.pred_overlay
        self.overlay_item = c.overlay_item
        self.rect_roi = c.rect_roi
        self.overlay_state = c.overlay_state
        self.state["cells"] = c.cells

    # =====================================================================
    # Floating toolbar
    # =====================================================================
    def _build_toolbar(self):
        self.toolbar = MapToolbar(
            self.view, self.basemap_groups, list(self.basemaps) + [self.PRED_LABEL],
            sorted(self.vlayer_paths), self.ctx.init_label, self.GRID_COLS,
            list(self.datasets), self.ctx.dataset_default, classes=self.classes)
        tb = self.toolbar
        tb.datasetChanged.connect(self.on_dataset_change)
        tb.basemapChanged.connect(self._on_basemap)
        tb.overlayChanged.connect(self.set_overlay)
        tb.overlayAlphaChanged.connect(self.set_overlay_alpha)
        tb.layerToggled.connect(self.toggle_layer)
        tb.gridToggled.connect(self.set_grid)
        tb.gridColsChanged.connect(self.set_cols)
        tb.pointSizeChanged.connect(self.set_size)
        tb.navToggled.connect(lambda on: self.nav.setVisible(on))
        tb.modelChanged.connect(self.on_model_change)
        tb.tableRequested.connect(self.open_table)
        tb.patternsRequested.connect(lambda: self.patterns_win.show_panel())
        tb.scopeChanged.connect(self.on_scope_change)
        tb.drawToggled.connect(self.on_draw_toggled)
        # camada de PONTOS + extras do painel de camadas
        pop = tb.pop
        pop.pointsVisibleToggled.connect(self.on_points_visible)
        pop.classVisibleToggled.connect(self.on_class_visible)
        pop.triageChanged.connect(self.on_triage)
        pop.pointsAlphaChanged.connect(self.on_pts_alpha)
        pop.vectorColorChanged.connect(self.set_layer_color)
        pop.vectorFillChanged.connect(self.set_layer_fill)
        pop.classColorChanged.connect(self.on_class_color)
        pop.predClassVisibleToggled.connect(self.on_pred_class_visible)
        pop.manageClassesRequested.connect(self.open_class_manager)
        pop.clearTempOverlay.connect(lambda: self.pred_overlay.setVisible(False))
        pop.openTableRequested.connect(self.show_review_from_map)
        # aliases: escopo/desenhar são da barra (área de trabalho compartilhada)
        self.area_combo = tb.scope_combo
        self.draw_btn = tb.draw_btn

    # =====================================================================
    # Annotate page
    # =====================================================================
    def _build_annotate(self):
        self.annotate = AnnotatePanel(list(self.classes))
        ap = self.annotate
        ap.proposeRequested.connect(self.do_propose)
        ap.skipRequested.connect(self.skip_suggestion)
        ap.goalChanged.connect(self.update_goals)
        # escopo compartilhado: o "onde" da aba Rotular dirige o mesmo scope_combo
        # (e reflete de volta mudanças da aba Classificar / chip ÁREA)
        ap.scopeChosen.connect(self.toolbar.set_scope)
        self.toolbar.scopeChanged.connect(ap.set_scope_selected)
        self.patterns_win = PatternsWindow(parent=self)
        pw = self.patterns_win
        pw.discoverRequested.connect(self.do_discover)
        pw.goToPattern.connect(self.go_to_pattern)
        pw.proposeFromPattern.connect(self.propose_from_pattern)
        pw.splitNode.connect(self.discover_split)
        pw.enterNode.connect(self.discover_enter)
        pw.goUp.connect(self.discover_up)
        pw.goToRoot.connect(self.discover_root)
        pw.applyToMap.connect(self.apply_patterns)
        pw.clearFromMap.connect(self.clear_patterns)
        # aliases: controller e smokes referenciam estes widgets direto
        self.goal_combo = ap.goal_combo
        self.tcls_combo = ap.tcls_combo
        self.count_spin = ap.count_spin
        self.pb = ap.pb
        self.propose_bar = ap.propose_bar
        self.scount = ap.scount
        self.slist = ap.slist

        self.slist.currentItemChanged.connect(self.on_sugg_select)
        self.update_goals()
        self.annot_page = ap

    # =====================================================================
    # Review page
    # =====================================================================
    def _build_review(self):
        """A tabela de pontos é uma JANELA própria (estilo tabela de atributos do
        ArcGIS): abre ao lado do mapa e atualiza AO VIVO a cada rótulo — não é
        exclusiva de um 'modo revisar'."""
        self.review = ReviewPanel(self.classes, fx=self.ctx.fx,
                                  feature_cols=getattr(self.ctx, "review_features", None))
        self.review.pointSelected.connect(self.on_review_point)
        self.review.pointActivated.connect(self.zoom_to_point)
        self.review_win = QWidget(self, Qt.WindowType.Window)
        self.review_win.setWindowTitle("Tabela de pontos — TSA (ao vivo)")
        rl = QVBoxLayout(self.review_win)
        rl.setContentsMargins(6, 6, 6, 6)
        rl.addWidget(self.review)
        self.review_win.resize(600, 680)

    def open_table(self):
        self.refresh_review_list()
        self.review_win.show()
        self.review_win.raise_()
        self.review_win.activateWindow()

    # =====================================================================
    # Train page
    # =====================================================================
    def _build_train(self):
        self.train = TrainPanel()
        self.train_win = TrainProgressWindow(self)   # janela pop de progresso
        self.train.trainRequested.connect(self.do_train)
        self.train.reviewSuspectsRequested.connect(self.show_review_suspects)
        # aliases: controller e smokes referenciam estes widgets direto
        self.lr_spin = self.train.lr_spin
        self.ep_spin = self.train.ep_spin
        self.fold_spin = self.train.fold_spin
        self.train_btn = self.train.train_btn
        self.train_status = self.train.train_status
        self.train_metrics = self.train.train_metrics
        self.train_page = self.train

    def _build_classify(self):
        self.classify = ClassifyPanel()
        self.classify.classifyRequested.connect(self.classify_scope)
        # escopo (4 opções) vive na aba Classify → dirige o scope_combo do toolbar
        # (que já aciona on_scope_change: grade, retângulo, etc.). Sync de volta pro
        # caso do chip ÁREA mudar o escopo por fora.
        self.classify.scopeChosen.connect(self.toolbar.set_scope)
        self.toolbar.scopeChanged.connect(self.classify.set_scope_selected)
        self.classify_page = self.classify

    # =====================================================================
    # Atalhos de teclado (filtro global — precisa ser método da janela/QObject)
    # =====================================================================
    def eventFilter(self, obj, ev):
        t = ev.type()
        if t == QEvent.Type.KeyPress:
            # quem está DIGITANDO (diálogo "nova classe", spins, combos) fica em paz
            fw = QApplication.focusWidget()
            if isinstance(fw, (QLineEdit, QAbstractSpinBox, QComboBox)):
                return False
            k = ev.key()
            if k == Qt.Key.Key_Space:
                self.state["pan"] = True
                self.view.setCursor(Qt.CursorShape.OpenHandCursor)
                return False
            txt = ev.text()
            if txt and txt in "1234567890":            # 1-9,0 = rotular com o card
                cls = self.panel.class_for_key(txt)
                if cls is not None:
                    self.on_label(cls)
                    return True
            if k in (Qt.Key.Key_Return, Qt.Key.Key_Enter):   # Enter = próxima sugestão
                self.skip_suggestion()
                return True
            if k == Qt.Key.Key_Backspace:              # Backspace = remover rótulo atual
                self.on_remove()
                return True
            if k == Qt.Key.Key_Z and ev.modifiers() & Qt.KeyboardModifier.ControlModifier:
                self.undo_last()                        # Ctrl+Z = desfazer
                return True
            if k == Qt.Key.Key_P:                       # P = liga/desliga a Predição por cima
                cur = self.toolbar.ov_combo.currentText()
                new = "(nenhum)" if cur == self.PRED_LABEL else self.PRED_LABEL
                self.toolbar.ov_combo.setCurrentText(new)
                self.set_status("predição sobreposta: OFF" if new == "(nenhum)"
                                else "predição sobreposta: ON (P alterna)")
                return True
        elif t == QEvent.Type.KeyRelease and ev.key() == Qt.Key.Key_Space:
            self.state["pan"] = False
            self.view.unsetCursor()
        return False

    # =====================================================================
    # Nav rail + assembly + wiring
    # =====================================================================
    def _build_nav(self):
        nav = QWidget()
        self.nav = nav
        nav.setMinimumWidth(210)
        nav.setMaximumWidth(300)
        nav_l = QVBoxLayout(nav)
        nav_l.setContentsMargins(8, 8, 8, 8)
        nav_l.setSpacing(4)
        nav_group = QButtonGroup(nav)
        nav_group.setExclusive(True)

        stack = QStackedWidget()
        self.stack = stack
        stack.addWidget(self.annot_page)
        stack.addWidget(self.train_page)
        stack.addWidget(self.classify_page)

        def _nav_btn(text, icon_name, idx):
            b = QToolButton()
            b.setObjectName("navbtn")
            b.setText("  " + text)
            b.setIcon(qta.icon(icon_name, color=ICON, color_active="#ffffff"))
            b.setIconSize(QSize(20, 20))
            b.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            b.setCheckable(True)
            b.setMinimumHeight(38)
            b.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            b.clicked.connect(lambda: self.stack.setCurrentIndex(idx))
            nav_group.addButton(b)
            nav_l.addWidget(b)
            return b

        btn_annot = _nav_btn("Rotular", "mdi6.pencil-outline", 0)
        _nav_btn("Treinar", "mdi6.brain", 1)
        _nav_btn("Classificar", "mdi6.map-check-outline", 2)
        btn_annot.setChecked(True)
        nav_l.addWidget(stack, 1)

    def show_review_from_map(self):
        """Mesma consulta, duas vistas: leva a triagem do MAPA como modalidade da tabela."""
        pop = self.toolbar.pop
        if pop.f_susp.isChecked():
            self.review.set_mode("susp")
        elif pop.f_disc.isChecked():
            self.review.set_mode("disc")
        self.open_table()

    def show_review_suspects(self):
        """Tabela já ordenada por suspeita do cleanlab (pior primeiro)."""
        self.review.sort_by_cleanlab()
        self.open_table()

    def _assemble(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        # faixa de sessão (dataset/modelo) ACIMA do mapa — sempre visível, zero
        # pixels de mapa roubados; as ferramentas de mapa flutuam no próprio mapa
        lay.addWidget(self.toolbar)

        center = QSplitter(Qt.Orientation.Vertical)  # mapa em cima, curva embaixo (ajustável)
        center.addWidget(self.view)
        center.addWidget(self.curve_view)
        center.setSizes([720, 240])
        center.setStretchFactor(0, 1)
        center.setHandleWidth(6)

        splitter = QSplitter(Qt.Orientation.Horizontal)  # nav esq | centro | cards dir
        splitter.addWidget(self.nav)
        splitter.addWidget(center)
        splitter.addWidget(self.panel)
        splitter.setSizes([270, 940, 360])
        splitter.setStretchFactor(1, 1)
        splitter.setHandleWidth(6)
        lay.addWidget(splitter, 1)

        # status bar: o feedback que ia só pro console agora tem casa na UI
        self.status_label = QLabel("pronto")
        self.status_label.setStyleSheet(
            "background:#1a1a1e; color:#bbb; padding:3px 10px; border-top:1px solid #2b2b2e;")
        lay.addWidget(self.status_label)
        self.resize(1500, 950)

    def set_status(self, text):
        self.status_label.setText(str(text))

    def _wire(self):
        # overlay basemap: refresh on range change (debounced)
        self.overlay_timer = QTimer(self.view)
        self.overlay_timer.setSingleShot(True)
        self.overlay_timer.timeout.connect(self.refresh_overlay)
        self.view.vb.sigRangeChanged.connect(
            lambda *_: self.overlay_timer.start(120) if self.overlay_state["label"] else None)

        # hover highlight
        self.hover_timer = QTimer(self.view)
        self.hover_timer.timeout.connect(self._proc_hover)
        self.hover_timer.start(60)

        # spacebar pan (global event filter)
        QApplication.instance().installEventFilter(self)
        self.view.glw.scene().sigMouseMoved.connect(self.on_hover)
        self.view.pixelClicked.connect(self.on_click)

        # markers: update only when the range stops changing (debounced)
        self.marker_timer = QTimer(self.view)
        self.marker_timer.setSingleShot(True)
        self.marker_timer.timeout.connect(self.refresh_markers)
        self.view.vb.sigRangeChanged.connect(lambda *_: self.marker_timer.start(120))

        # class picker
        self.panel.classChosen.connect(self.on_label)
        self.panel.removeRequested.connect(self.on_remove)
        self.panel.classAdded.connect(self.on_class_added)

        self.update_scope_labels()   # botões nascem ecoando o escopo ("vista")

    def show(self):
        super().show()
        self.toolbar.reposition()

    def closeEvent(self, ev):
        self.controller.shutdown()
        if hasattr(self.view, "stop"):
            self.view.stop()   # tile worker: o closeEvent da view não dispara (não é top-level)
        super().closeEvent(ev)

    def fit(self):
        g = self.view.vb.geometry()
        W, H = self.class_src.width, self.class_src.height
        wa = max(g.width(), 1) / max(g.height(), 1)
        da = W / H
        sy, sx = (H * 1.1, H * 1.1 * wa) if wa > da else (W * 1.1 / wa, W * 1.1)
        self.view.vb.setRange(xRange=(W / 2 - sx / 2, W / 2 + sx / 2),
                              yRange=(H / 2 - sy / 2, H / 2 + sy / 2), padding=0)
        self.refresh_markers()
        self.refresh_review_list()
