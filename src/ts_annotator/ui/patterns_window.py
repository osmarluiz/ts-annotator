"""PatternsWindow — navigate the user-driven divisive pattern tree.

A breadcrumb (raiz › padrão 2 › …) plus the cards of the CURRENT node's
children: each card is a mini NDVI curve of its MEDOID, its prevalence (% of the
whole area), and actions — "dividir em [k]" (drill into this group and split it
further), "ir" (zoom the map to the medoid and load it, ready to label), and
"propor deste" (find more pixels like it). Dumb view: the controller drives the
tree and hands it the node to render; it emits splitNode(k) / enterNode(i) /
goUp() / goToPattern(i) / proposeFromPattern(i).
"""

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ts_annotator.core.features import to_ndvi
from ts_annotator.render import CLUSTER_PALETTE


class _PatternCard(QFrame):
    def __init__(self, i, node, on_enter, on_go, on_propose):
        super().__init__()
        color = CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)]
        self.setStyleSheet("QFrame{background:#1a1b20; border:1px solid #2a2c34; border-radius:8px;}"
                           f"QFrame#swatch{{background:{color}; border-radius:3px;}}")
        L = QHBoxLayout(self)
        L.setContentsMargins(8, 6, 8, 6)
        sw = QFrame()
        sw.setObjectName("swatch")
        sw.setFixedWidth(6)
        L.addWidget(sw)

        plot = pg.PlotWidget()
        plot.setFixedSize(168, 92)
        plot.setBackground("#141518")
        plot.setYRange(-0.1, 1.0)
        plot.getAxis("bottom").setStyle(showValues=False)
        plot.showGrid(x=False, y=True, alpha=0.15)
        nd = to_ndvi(np.asarray(node.medoid["curve"], float))
        x = np.arange(len(nd))
        mask = np.isfinite(nd)
        plot.plot(x[mask], nd[mask], pen=pg.mkPen("#00e676", width=2),
                  fillLevel=0, brush=pg.mkBrush(0, 230, 118, 40))
        L.addWidget(plot)

        info = QVBoxLayout()
        info.setSpacing(3)
        pct = node.frac * 100
        info.addWidget(QLabel(f"<b>Padrão {i + 1}</b>  ·  {pct:.0f}% da área  ·  {node.size} px"))
        bar = QProgressBar()
        bar.setRange(0, 100)
        bar.setValue(int(round(pct)))
        bar.setTextVisible(False)
        bar.setFixedHeight(6)
        info.addWidget(bar)
        m = node.medoid
        info.addWidget(QLabel(f"<span style='color:#888'>medóide r{m['row']} c{m['col']}</span>"))

        r1 = QHBoxLayout()
        r1.addWidget(QLabel("dividir em"))
        spin = QSpinBox()
        spin.setRange(2, 8)
        spin.setValue(3)
        spin.setFixedWidth(48)
        r1.addWidget(spin)
        div = QPushButton("→")
        div.setFixedWidth(30)
        div.setToolTip("entra neste grupo e o subdivide (drill-down)")
        div.clicked.connect(lambda: on_enter(i, spin.value()))
        r1.addWidget(div)
        r1.addStretch()
        info.addLayout(r1)

        r2 = QHBoxLayout()
        go = QPushButton("ir →")
        go.setToolTip("zoom no pixel representativo + carrega a curva (rotule com 1–9)")
        go.clicked.connect(lambda: on_go(i))
        r2.addWidget(go)
        prop = QPushButton("propor deste")
        prop.setToolTip("acha mais pixels parecidos com este padrão → lista de sugestões")
        prop.clicked.connect(lambda: on_propose(i))
        r2.addWidget(prop)
        info.addLayout(r2)
        L.addLayout(info, 1)


class PatternsWindow(QWidget):
    discoverRequested = pyqtSignal(str, int, int)   # (escopo de área, step de densidade, k inicial)
    splitNode = pyqtSignal(int)          # dividir o nó ATUAL em k (k = nº)
    enterNode = pyqtSignal(int, int)     # entrar no filho i e dividir em k
    goUp = pyqtSignal()
    goToRoot = pyqtSignal()
    goToPattern = pyqtSignal(int)
    proposeFromPattern = pyqtSignal(int)
    applyToMap = pyqtSignal()            # pintar os padrões do nível no mapa
    clearFromMap = pyqtSignal()

    def __init__(self, months=None, parent=None):
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowTitle("Descobrir padrões — TSA")
        self.resize(460, 660)
        self._months = months
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)

        # ============ seção DESCOBRIR (configurar + disparar) ============
        title = QLabel("DESCOBRIR PADRÕES")
        title.setStyleSheet("font-weight:bold; letter-spacing:1px; color:#c8c8d0;")
        root.addWidget(title)
        sub = QLabel("Agrupa as curvas de uma região e mostra os padrões que mais "
                     "se repetem — não-supervisionado. Serve pra <b>explorar</b> a "
                     "imagem antes de rotular (não entra no dataset).")
        sub.setStyleSheet("color:#9a9aa4;")
        sub.setWordWrap(True)
        root.addWidget(sub)

        # passo 1 — ONDE analisar (área própria da descoberta, não o chip do mapa)
        arow = QHBoxLayout()
        n1 = QLabel("<b>1.</b> onde")
        n1.setFixedWidth(52)
        arow.addWidget(n1)
        self.area_combo = QComboBox()
        for label, key in [("vista atual do mapa", "viewport"),
                           ("retângulo desenhado no mapa", "rect"),
                           ("células do grid selecionadas", "grids"),
                           ("imagem inteira", "all")]:
            self.area_combo.addItem(label, key)
        self.area_combo.setToolTip("região onde os padrões serão buscados; para "
                                   "'retângulo'/'células' desenhe/selecione no mapa antes")
        arow.addWidget(self.area_combo, 1)
        root.addLayout(arow)

        # passo 2 — densidade + nº de padrões
        cfg = QHBoxLayout()
        n2 = QLabel("<b>2.</b> como")
        n2.setFixedWidth(52)
        cfg.addWidget(n2)
        cfg.addWidget(QLabel("1 amostra a cada"))
        self.density_spin = QSpinBox()
        self.density_spin.setRange(1, 500)
        self.density_spin.setValue(10)
        self.density_spin.setSuffix(" px")
        self.density_spin.setToolTip("densidade da amostragem na área; passo pequeno em área "
                                     "grande é limitado por um teto de segurança")
        cfg.addWidget(self.density_spin)
        cfg.addSpacing(8)
        cfg.addWidget(QLabel("em"))
        self.k_spin = QSpinBox()
        self.k_spin.setRange(2, 8)
        self.k_spin.setValue(6)
        self.k_spin.setToolTip("nº de padrões no primeiro nível (subdivida depois)")
        cfg.addWidget(self.k_spin)
        cfg.addWidget(QLabel("padrões"))
        cfg.addStretch()
        root.addLayout(cfg)

        # passo 3 — disparar
        self.discover_btn = QPushButton("  3.  Descobrir padrões")
        self.discover_btn.setObjectName("primary")
        self.discover_btn.clicked.connect(
            lambda: self.discoverRequested.emit(
                self.area_combo.currentData(), self.density_spin.value(), self.k_spin.value()))
        root.addWidget(self.discover_btn)
        self.busy = QProgressBar()
        self.busy.setRange(0, 0)          # indeterminado ("trabalhando…")
        self.busy.setTextVisible(True)
        self.busy.setFormat("descobrindo padrões…")
        self.busy.setFixedHeight(14)
        self.busy.setVisible(False)
        root.addWidget(self.busy)
        self.discover_info = QLabel("")
        self.discover_info.setStyleSheet("color:#888;")
        self.discover_info.setWordWrap(True)
        root.addWidget(self.discover_info)
        sep0 = QFrame()
        sep0.setFrameShape(QFrame.Shape.HLine)
        sep0.setStyleSheet("color:#2a2c34;")
        root.addWidget(sep0)

        # breadcrumb
        self.crumb = QLabel("")
        self.crumb.setStyleSheet("color:#c8c8d0;")
        self.crumb.setWordWrap(True)
        root.addWidget(self.crumb)
        nav = QHBoxLayout()
        self.up_btn = QPushButton("↑ subir")
        self.up_btn.clicked.connect(self.goUp)
        nav.addWidget(self.up_btn)
        self.root_btn = QPushButton("raiz")
        self.root_btn.clicked.connect(self.goToRoot)
        nav.addWidget(self.root_btn)
        nav.addWidget(QLabel("redividir em"))
        self.resplit = QSpinBox()
        self.resplit.setRange(2, 8)
        self.resplit.setValue(6)
        self.resplit.setFixedWidth(48)
        nav.addWidget(self.resplit)
        rb = QPushButton("↺")
        rb.setFixedWidth(30)
        rb.setToolTip("redivide o nó atual com outro k")
        rb.clicked.connect(lambda: self.splitNode.emit(self.resplit.value()))
        nav.addWidget(rb)
        nav.addStretch()
        root.addLayout(nav)

        self.head = QLabel("Defina <b>onde</b> e <b>como</b> acima e clique "
                           "<b>Descobrir</b>. Os padrões aparecem como cards aqui; "
                           "cada card é uma curva típica — 'ir' te leva a um pixel "
                           "dele pra rotular, 'dividir' o refina.")
        self.head.setStyleSheet("color:#9a9aa4;")
        self.head.setWordWrap(True)
        root.addWidget(self.head)

        mrow = QHBoxLayout()
        self.map_btn = QPushButton("mostrar no mapa")
        self.map_btn.setToolTip("pinta os padrões deste nível no mapa (cores = cards)")
        self.map_btn.clicked.connect(self.applyToMap)
        mrow.addWidget(self.map_btn)
        clr = QPushButton("limpar do mapa")
        clr.clicked.connect(self.clearFromMap)
        mrow.addWidget(clr)
        root.addLayout(mrow)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setStyleSheet("QScrollArea{border:none;}")
        self._host = QWidget()
        self._box = QVBoxLayout(self._host)
        self._box.setSpacing(6)
        self._box.addStretch()
        self._scroll.setWidget(self._host)
        root.addWidget(self._scroll, 1)

    def show_node(self, node, depth):
        """Renderiza os FILHOS do nó atual + o breadcrumb (depth = nível na árvore)."""
        while self._box.count() > 1:
            it = self._box.takeAt(0)
            w = it.widget()
            if w is not None:
                w.setParent(None)   # remoção SÍNCRONA (setParent zera it.widget() → guarda antes)
                w.deleteLater()
        self.crumb.setText("📍 " + (node.path.replace(".", " › ") if node else "—"))
        self.up_btn.setEnabled(bool(node and node.parent))
        children = node.children if node else None
        if not children:
            self.head.setText("grupo folha (sem subdivisão) — use 'ir' ou 'propor deste', "
                              "ou redivida acima")
            return
        self.head.setText(f"<b>{len(children)} padrões</b> neste nível "
                          f"(% = prevalência na ÁREA · medóide = pixel representativo)")
        for i, ch in enumerate(children):
            card = _PatternCard(i, ch, self._enter, self.goToPattern.emit,
                                self.proposeFromPattern.emit)
            self._box.insertWidget(self._box.count() - 1, card)

    def set_busy(self, on: bool):
        """Feedback visível enquanto a descoberta (síncrona) roda."""
        self.busy.setVisible(on)
        self.discover_btn.setEnabled(not on)

    def show_panel(self):
        self.show()
        self.raise_()
        self.activateWindow()

    def _enter(self, i, k):
        self.enterNode.emit(i, k)
