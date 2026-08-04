"""LayersPanel — the map's five categories as always-visible CHIPS + dropdowns.

Design (co-designed with the owner): the top of the map shows five chips, each
naming a category AND its current state — the map's vocabulary at a glance:

    OBSERVAÇÃO · NDVI │ RESULTADOS · — │ PONTOS · 39.996 │ VETORIAIS · 0 │ ÁREA · vista │ ?

Clicking a chip opens that category's dropdown — a small DRAGGABLE panel with
a title bar and a close button that STAYS OPEN while you work on the map
(re-click the chip or hit ✕ to close; several can be open at once):

- **OBSERVAÇÃO** — which signal am I looking at? (exclusive base radios)
- **RESULTADOS** — what do models say? Pipeline products + this app's active
  prediction, each assignable to the roles *fundo*/*sobrepor*.
- **PONTOS** — the labeled points as a layer: visibility, legend-as-filter
  with live counts, triage filters, size/opacity, jump to the Review table.
- **VETORIAIS** — vector outlines, checkbox + color swatch each.
- **ÁREA** — the shared work area + draw toggle + grid (conditional controls).

This class is a QObject FACADE: it builds the chip bar and the dropdown frames
(floating children of the map view) and exposes the same signal and attribute
names the rest of the app uses; the MapToolbar keeps the hidden state combos.
"""

from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QColorDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
)
import qtawesome as qta

from ts_annotator.app.config import order_classes
from ts_annotator.app.theme import ICON

_DROP_QSS = (
    "QFrame#drop{background:rgba(19,19,23,247); border:1px solid #2e2e36; border-radius:10px;}"
    "QFrame#drop QLabel{color:#9a9aa2;}"
    "QFrame#drop QLabel#droptitle{color:#e8e8ee; font-weight:bold; letter-spacing:1px;}"
    "QToolButton{border:1px solid #3a3a40; border-radius:4px; padding:1px 7px; color:#bbb;}"
    "QToolButton:checked{background:#2d5a88; color:white; border-color:#2d5a88;}")
_CHIP_QSS = (
    "QFrame#chips{background:rgba(19,19,23,238); border:1px solid #2e2e36; border-radius:9px;}"
    "QFrame#chips QToolButton{border:none; border-radius:6px; padding:4px 10px; color:#b6b6c0;}"
    "QFrame#chips QToolButton:hover{background:#26262e;}"
    "QFrame#chips QToolButton:checked{background:#2d5a88; color:white;}"
    "QFrame#chips QFrame#sep{background:#2e2e36; max-width:1px; margin-top:5px; margin-bottom:5px;}")


class _Drop(QFrame):
    """Dropdown de categoria: título + ✕, ARRASTÁVEL, persiste aberto."""

    def __init__(self, view, title, on_close):
        super().__init__(view)
        self.setObjectName("drop")
        self.setStyleSheet(_DROP_QSS)
        self.setMinimumWidth(250)
        self._drag = None
        self.user_moved = False
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 6, 12, 10)
        outer.setSpacing(5)
        head = QHBoxLayout()
        t = QLabel(title)
        t.setObjectName("droptitle")
        head.addWidget(t)
        head.addStretch()
        x = QToolButton()
        x.setText("✕")
        x.setStyleSheet("QToolButton{border:none; color:#8a8a92; padding:0 4px;}"
                        "QToolButton:hover{color:white;}")
        x.clicked.connect(on_close)
        head.addWidget(x)
        outer.addLayout(head)
        self.body = QVBoxLayout()
        self.body.setSpacing(5)
        outer.addLayout(self.body)

    # arraste pelo corpo/cabeçalho (sobre o mapa)
    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            self._drag = ev.position().toPoint()
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._drag is not None:
            self.user_moved = True
            p = self.mapToParent(ev.position().toPoint() - self._drag)
            pw = self.parentWidget()
            x = max(0, min(p.x(), pw.width() - self.width()))
            y = max(0, min(p.y(), pw.height() - self.height()))
            self.move(x, y)
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        self._drag = None
        super().mouseReleaseEvent(ev)


class LayersPanel(QObject):
    baseSelected = pyqtSignal(str)
    overlaySelected = pyqtSignal(object)        # str | None
    overlayAlphaChanged = pyqtSignal(int)
    clearTempOverlay = pyqtSignal()
    pointsVisibleToggled = pyqtSignal(bool)
    classVisibleToggled = pyqtSignal(str, bool)
    classColorChanged = pyqtSignal(str, str)          # (classe, novo hex) — pontos+predição+legenda
    predClassVisibleToggled = pyqtSignal(str, bool)   # (classe, visível no raster de predição)
    manageClassesRequested = pyqtSignal()             # abrir o gerenciador (renomear/mesclar/remover)
    triageChanged = pyqtSignal(bool, bool)      # (só ≠ modelo, só suspeitos)
    openTableRequested = pyqtSignal()
    pointSizeChanged = pyqtSignal(int)
    pointsAlphaChanged = pyqtSignal(int)
    vectorToggled = pyqtSignal(str, bool)
    vectorColorChanged = pyqtSignal(str, str)
    vectorFillChanged = pyqtSignal(str, str)
    scopeSelected = pyqtSignal(str)             # viewport | rect | grids | all
    drawToggled = pyqtSignal(bool)
    gridToggled = pyqtSignal(bool)
    gridColsChanged = pyqtSignal(int)

    _SCOPES = [("vista atual", "viewport"), ("retângulo", "rect"),
               ("células da grade", "grids"), ("imagem toda", "all")]
    _CHIP_NAMES = {"obs": "OBSERVAÇÃO", "res": "RESULTADOS", "pts": "PONTOS",
                   "vec": "VETORIAIS", "area": "ÁREA"}

    def __init__(self, view, obs_labels, result_labels, pred_label, vlayer_names,
                 classes, init_label, grid_cols):
        super().__init__()
        self.view = view
        self._pred_label = pred_label
        self._classes = dict(classes)
        self._hidden = set()
        self._n_vec_on = 0

        # ---------------- chips (sempre visíveis; estado no rótulo)
        self.chip_bar = QFrame(view)
        self.chip_bar.setObjectName("chips")
        self.chip_bar.setStyleSheet(_CHIP_QSS)
        cb_lay = QHBoxLayout(self.chip_bar)
        cb_lay.setContentsMargins(5, 3, 5, 3)
        cb_lay.setSpacing(2)
        self._chips = {}
        self.frames = {}
        keys = ("obs", "res", "pts", "vec", "area")
        for i, key in enumerate(keys):
            b = QToolButton()
            b.setCheckable(True)
            b.toggled.connect(lambda on, k=key: self._open(k, on))
            cb_lay.addWidget(b)
            self._chips[key] = b
            sep = QFrame()
            sep.setObjectName("sep")
            sep.setFrameShape(QFrame.Shape.VLine)
            cb_lay.addWidget(sep)
        self.help_btn = QToolButton()
        self.help_btn.setText("?")
        self.help_btn.setToolTip(
            "<b>Atalhos</b><br>"
            "<b>1–9, 0</b> — rotular o ponto atual com a classe do card<br>"
            "<b>Enter</b> — próxima sugestão (pular)<br>"
            "<b>Backspace</b> — remover o rótulo do ponto atual<br>"
            "<b>Ctrl+Z</b> — desfazer o último rótulo/remoção<br>"
            "<b>Espaço (segurar)</b> — clique vira pan (não rotula)<br>"
            "<b>arrastar</b> — move o mapa; com 'desenhar' ligado, cria o retângulo<br>"
            "<b>clique com a grade ligada</b> — seleciona/desseleciona célula")
        cb_lay.addWidget(self.help_btn)

        def _mk(key):
            f = _Drop(view, self._CHIP_NAMES[key],
                      lambda: self._chips[key].setChecked(False))
            self.frames[key] = f
            f.hide()
            return f.body

        # ---------------- OBSERVAÇÃO
        lay = _mk("obs")
        self._base_group = QButtonGroup(self)
        self._base_group.setExclusive(True)
        self._base_btns = {}
        for lb in obs_labels:
            rb = QRadioButton(lb)
            rb.toggled.connect(lambda on, l=lb: on and self.baseSelected.emit(l))
            self._base_group.addButton(rb)
            self._base_btns[lb] = rb
            lay.addWidget(rb)

        # ---------------- RESULTADOS
        lay = _mk("res")
        self._over_btns = {}
        if result_labels:
            lay.addWidget(QLabel("do pipeline (importados)"))
            for lb in result_labels:
                lay.addLayout(self._result_row(lb, QLabel(lb)))
        lay.addWidget(QLabel("deste app"))
        self.pred_info = QLabel("(sem modelo)")
        self.pred_info.setStyleSheet("color:#c8c8d0;")
        lay.addLayout(self._result_row(pred_label, self.pred_info))
        a_row = QHBoxLayout()
        a_row.addWidget(QLabel("opacidade da sobreposição"))
        self.ov_sld = QSlider(Qt.Orientation.Horizontal)
        self.ov_sld.setRange(0, 100)
        self.ov_sld.setValue(60)
        self.ov_sld.setFixedWidth(80)
        self.ov_sld.valueChanged.connect(self.overlayAlphaChanged)
        a_row.addWidget(self.ov_sld)
        lay.addLayout(a_row)
        clr = QPushButton("limpar overlay temporário")
        clr.setToolTip("apaga a pintura volátil do 'classificar área' (não mexe no raster salvo)")
        clr.clicked.connect(self.clearTempOverlay)
        lay.addWidget(clr)
        lay.addWidget(QLabel("classes na predição (desmarque p/ ocultar)"))
        self._pred_legend_box = QVBoxLayout()
        self._pred_legend_box.setSpacing(2)
        lay.addLayout(self._pred_legend_box)
        self._pred_rows = {}
        self._pred_hidden = set()

        # ---------------- PONTOS
        lay = _mk("pts")
        self.pts_visible = QCheckBox("mostrar pontos")
        self.pts_visible.setChecked(True)
        self.pts_visible.toggled.connect(self._on_pts_visible)
        lay.addWidget(self.pts_visible)
        self.active_label = QLabel("—")
        self.active_label.setStyleSheet("color:#c8c8d0;")
        lay.addWidget(self.active_label)
        lay.addWidget(QLabel("legenda (clique = filtra o mapa)"))
        self._legend_box = QVBoxLayout()
        self._legend_box.setSpacing(2)
        lay.addLayout(self._legend_box)
        self._legend_rows = {}
        mgr = QPushButton("⚙ gerenciar classes (renomear / mesclar / remover)")
        mgr.setToolTip("renomear, mesclar ou remover classes — atualiza pontos, "
                       "legenda e tabela; retreine depois pra atualizar a predição")
        mgr.clicked.connect(self.manageClassesRequested)
        lay.addWidget(mgr)
        tri = QHBoxLayout()
        self.f_disc = QCheckBox("só ≠ modelo")
        self.f_susp = QCheckBox("só suspeitos")
        self.f_susp.setToolTip("suspeitos do cleanlab (find_label_issues no treino) — "
                               "mesmo conjunto do contador; nota<0.5 é na tabela")
        for cb in (self.f_disc, self.f_susp):
            cb.toggled.connect(self._on_triage)
            tri.addWidget(cb)
        tri.addStretch()
        lay.addLayout(tri)
        open_tb = QPushButton("abrir na tabela →")
        open_tb.clicked.connect(self.openTableRequested)
        lay.addWidget(open_tb)
        sz = QHBoxLayout()
        sz.addWidget(QLabel("tamanho"))
        self.size_sld = QSlider(Qt.Orientation.Horizontal)
        self.size_sld.setRange(4, 30)
        self.size_sld.setValue(12)
        self.size_sld.valueChanged.connect(self.pointSizeChanged)
        sz.addWidget(self.size_sld)
        sz.addWidget(QLabel("opac."))
        self.pt_alpha = QSlider(Qt.Orientation.Horizontal)
        self.pt_alpha.setRange(20, 100)
        self.pt_alpha.setValue(100)
        self.pt_alpha.valueChanged.connect(self.pointsAlphaChanged)
        sz.addWidget(self.pt_alpha)
        lay.addLayout(sz)

        # ---------------- VETORIAIS
        lay = _mk("vec")
        if not vlayer_names:
            lay.addWidget(QLabel("(dropar .gpkg/.geojson em layers/)"))
        for nm in vlayer_names:
            row = QHBoxLayout()
            cb = QCheckBox(nm)
            cb.toggled.connect(lambda on, n=nm: self._on_vec(n, on))
            row.addWidget(cb, 1)
            sw = QPushButton()
            sw.setFixedSize(16, 16)
            sw.setStyleSheet("background:#00e5ff; border-radius:3px;")
            sw.setToolTip("cor do contorno")
            sw.clicked.connect(lambda _, n=nm, b=sw: self._pick_color(n, b))
            row.addWidget(sw)
            fw = QPushButton()
            fw.setFixedSize(16, 16)
            fw.setStyleSheet("background:transparent; border:1px solid #666; border-radius:3px;")
            fw.setToolTip("cor do preenchimento (com transparência)")
            fw.clicked.connect(lambda _, n=nm, b=fw: self._pick_fill(n, b))
            row.addWidget(fw)
            lay.addLayout(row)

        # ---------------- ÁREA (escopo + desenhar + grade)
        lay = _mk("area")
        self._scope_group = QButtonGroup(self)
        self._scope_group.setExclusive(True)
        self._scope_btns = {}
        for label, data in self._SCOPES:
            rb = QRadioButton(label)
            rb.toggled.connect(lambda on, d=data: on and self.scopeSelected.emit(d))
            self._scope_group.addButton(rb)
            self._scope_btns[data] = rb
            lay.addWidget(rb)
        self._scope_btns["viewport"].setChecked(True)
        self.draw_btn = QToolButton()
        self.draw_btn.setText(" desenhar")
        self.draw_btn.setIcon(qta.icon("mdi6.vector-rectangle", color=ICON))
        self.draw_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.draw_btn.setCheckable(True)
        self.draw_btn.setToolTip("ligado: arraste no mapa p/ desenhar o retângulo (o mapa não move)")
        self.draw_btn.toggled.connect(self._on_draw)
        self.draw_btn.setVisible(False)          # só em escopo retângulo
        lay.addWidget(self.draw_btn)
        g_row = QHBoxLayout()
        self.gcb = QCheckBox("grade (clique seleciona células)")
        self.gcb.toggled.connect(self._on_grid)
        g_row.addWidget(self.gcb)
        self.gcols = QSpinBox()
        self.gcols.setRange(4, 40)
        self.gcols.setValue(grid_cols)
        self.gcols.setToolTip("nº de colunas do grid")
        self.gcols.valueChanged.connect(self.gridColsChanged)
        self.gcols.setVisible(False)             # só com a grade ligada
        g_row.addWidget(self.gcols)
        g_row.addStretch()
        lay.addLayout(g_row)

        self.sync_base(init_label)
        self._chip("res", "—")
        self._chip("pts", "…")
        self._chip("vec", "0")
        self._chip("area", "vista")

    # ------------------------------------------------------------- chips
    def _chip(self, key, state):
        s = state if len(state) <= 18 else state[:17] + "…"
        self._chips[key].setText(f"{self._CHIP_NAMES[key]} · {s}")
        self.chip_bar.adjustSize()

    def _open(self, key, on):
        """Abre/fecha o dropdown do chip — UM por vez (abrir fecha o anterior);
        arrastável enquanto aberto (a posição do usuário é respeitada)."""
        if on:
            for k, b in self._chips.items():
                if k != key and b.isChecked():
                    b.blockSignals(True)
                    b.setChecked(False)
                    b.blockSignals(False)
                    self.frames[k].hide()
        fr = self.frames[key]
        fr.setVisible(on)
        if on:
            fr.adjustSize()
            if not fr.user_moved:
                x = self.chip_bar.x() + self._chips[key].x()
                x = min(x, self.view.width() - fr.width() - 10)
                fr.move(max(10, x), self.chip_bar.y() + self.chip_bar.height() + 6)
            fr.raise_()

    def reposition(self):
        self.chip_bar.adjustSize()
        self.chip_bar.move(10, 8)
        self.chip_bar.raise_()
        self.chip_bar.show()
        for k, b in self._chips.items():
            if b.isChecked():
                self._open(k, True)

    # ------------------------------------------------------------- internos
    def _result_row(self, label, head_widget):
        row = QHBoxLayout()
        row.addWidget(head_widget, 1)
        fundo = QToolButton()
        fundo.setText("fundo")
        fundo.setCheckable(True)
        self._base_group.addButton(fundo)
        fundo.toggled.connect(lambda on, l=label: on and self.baseSelected.emit(l))
        self._base_btns[label] = fundo
        row.addWidget(fundo)
        over = QToolButton()
        over.setText("sobrepor")
        over.setCheckable(True)
        over.clicked.connect(lambda on, l=label: self._on_over(l, on))
        self._over_btns[label] = over
        row.addWidget(over)
        return row

    def _on_over(self, label, on):
        for l, b in self._over_btns.items():
            if l != label:
                b.blockSignals(True)
                b.setChecked(False)
                b.blockSignals(False)
        self.overlaySelected.emit(label if on else None)

    def _on_pts_visible(self, on):
        self.pointsVisibleToggled.emit(on)
        self._refresh_pts_chip()

    def _on_triage(self, _):
        self.triageChanged.emit(self.f_disc.isChecked(), self.f_susp.isChecked())
        self._refresh_pts_chip()

    def _on_vec(self, name, on):
        self._n_vec_on += 1 if on else -1
        self._chip("vec", str(max(0, self._n_vec_on)))
        self.vectorToggled.emit(name, on)

    def _on_draw(self, on):
        self.draw_btn.setText(" desenhando…" if on else " desenhar")
        self.drawToggled.emit(on)

    def _on_grid(self, on):
        self.gcols.setVisible(on)
        self.gridToggled.emit(on)

    def _pick_color(self, name, btn):
        c = QColorDialog.getColor()
        if c.isValid():
            btn.setStyleSheet(f"background:{c.name()}; border-radius:3px;")
            self.vectorColorChanged.emit(name, c.name())

    def _pick_fill(self, name, btn):
        c = QColorDialog.getColor(QColor(0, 229, 255, 90),
                                  options=QColorDialog.ColorDialogOption.ShowAlphaChannel)
        if c.isValid():
            btn.setStyleSheet(f"background:{c.name()}; border:1px solid #666; border-radius:3px;")
            self.vectorFillChanged.emit(name, c.name(QColor.NameFormat.HexArgb))

    def _pick_class_color(self, cls, btn):
        c = QColorDialog.getColor()
        if not c.isValid():
            return
        self._classes[cls] = c.name()
        btn.setStyleSheet(f"background:{c.name()}; border:1px solid #555; border-radius:3px;")
        pr = self._pred_rows.get(cls)   # espelha o pontinho na legenda da predição
        if pr:
            pr[1].setStyleSheet(f"color:{c.name()}; font-size:13px;")
        self.classColorChanged.emit(cls, c.name())

    def _on_pred_legend(self, cls, on):
        (self._pred_hidden.discard(cls) if on else self._pred_hidden.add(cls))
        self.predClassVisibleToggled.emit(cls, on)

    def _refresh_pts_chip(self):
        if not self.pts_visible.isChecked():
            self._chip("pts", "ocultos")
            return
        n = getattr(self, "_n_pts", "…")
        flt = " ⚑" if (self.f_disc.isChecked() or self.f_susp.isChecked() or self._hidden) else ""
        self._chip("pts", f"{n}{flt}")

    def _on_legend(self, cls, on):
        (self._hidden.discard(cls) if on else self._hidden.add(cls))
        self.classVisibleToggled.emit(cls, on)
        self._refresh_pts_chip()

    @staticmethod
    def _clear_layout(layout):
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
            elif item.layout() is not None:
                LayersPanel._clear_layout(item.layout())

    def rebuild_classes(self, classes):
        """Após renomear/mesclar/remover: zera as legendas (pontos + predição) e o
        estado de ocultas; update_points repopula com as classes novas."""
        self._classes = dict(classes)
        self._clear_layout(self._legend_box)
        self._clear_layout(self._pred_legend_box)
        self._legend_rows = {}
        self._pred_rows = {}
        self._hidden &= set(classes)
        self._pred_hidden &= set(classes)

    # ------------------------------------------------------------- sync API
    def sync_base(self, label):
        for l, b in self._base_btns.items():
            b.blockSignals(True)
            b.setChecked(l == label)
            b.blockSignals(False)
        self._chip("obs", label)

    def sync_overlay(self, label):
        for l, b in self._over_btns.items():
            b.blockSignals(True)
            b.setChecked(l == label)
            b.blockSignals(False)
        self._chip("res", label if label else "—")

    def sync_scope(self, data, cells=0):
        b = self._scope_btns.get(data)
        if b is not None:
            b.blockSignals(True)
            b.setChecked(True)
            b.blockSignals(False)
        self.draw_btn.setVisible(data == "rect")
        short = {"viewport": "vista", "rect": "retângulo",
                 "grids": f"células: {cells}", "all": "imagem toda"}.get(data, data)
        self._chip("area", short)

    def set_pred_info(self, text, available):
        self.pred_info.setText(text)
        for b in (self._base_btns.get(self._pred_label), self._over_btns.get(self._pred_label)):
            if b is not None:
                b.setEnabled(available)

    def update_points(self, active_name, n_active, class_counts):
        """Cabeçalho + legenda com contagens vivas + chip."""
        self.active_label.setText(f"dataset: {active_name} ({n_active})")
        self._n_pts = n_active
        self._refresh_pts_chip()
        for cls in class_counts:
            self._classes.setdefault(cls, "#cccccc")   # classe criada em runtime
        for cls in order_classes(self._classes):
            color = self._classes[cls]
            n = class_counts.get(cls, 0)
            if cls not in self._legend_rows:
                row = QHBoxLayout()
                cb = QCheckBox()
                cb.setChecked(cls not in self._hidden)
                cb.toggled.connect(lambda on, c=cls: self._on_legend(c, on))
                row.addWidget(cb)
                sw = QPushButton()                   # swatch CLICÁVEL = editar a cor da classe
                sw.setFixedSize(14, 14)
                sw.setStyleSheet(f"background:{color}; border:1px solid #555; border-radius:3px;")
                sw.setToolTip("clique p/ mudar a cor da classe (pontos + predição + legenda)")
                sw.clicked.connect(lambda _, c=cls, b=sw: self._pick_class_color(c, b))
                row.addWidget(sw)
                lab = QLabel(cls)
                lab.setStyleSheet("color:#d8d8e0;")
                row.addWidget(lab, 1)
                cnt = QLabel()                       # contagem alinhada à direita, dim
                cnt.setStyleSheet("color:#7a7d88;")
                cnt.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                row.addWidget(cnt)
                self._legend_box.addLayout(row)
                self._legend_rows[cls] = (cb, cnt, sw)
            self._legend_rows[cls][1].setText(f"{n:,}".replace(",", "."))
        # legenda da PREDIÇÃO (mesmas classes; checkbox = mostrar no raster classificado)
        for cls in order_classes(self._classes):
            color = self._classes[cls]
            if cls not in self._pred_rows:
                row = QHBoxLayout()
                cb = QCheckBox()
                cb.setChecked(cls not in self._pred_hidden)
                cb.toggled.connect(lambda on, c=cls: self._on_pred_legend(c, on))
                row.addWidget(cb)
                dot = QLabel("●")
                dot.setStyleSheet(f"color:{color}; font-size:13px;")
                row.addWidget(dot)
                lab = QLabel(cls)
                lab.setStyleSheet("color:#d8d8e0;")
                row.addWidget(lab, 1)
                self._pred_legend_box.addLayout(row)
                self._pred_rows[cls] = (cb, dot)
