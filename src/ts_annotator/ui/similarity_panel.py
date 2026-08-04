"""CurveView (curva + bandas) + SimilarityPanel (cards de classe) — separados p/ layout flexível.

A curva fica num widget próprio (vai abaixo do mapa, largura cheia, altura ajustável);
os cards de classe (similaridade + rótulo) num painel próprio (lateral). Ambos
atualizados pelo controlador (demo): curve_view.set_curve(...) + panel.set_scores(...).
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ts_annotator.app.config import order_classes
from ts_annotator.core.features import to_ndvi

MONTHS = ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep"]
BAND_STYLE = {
    "NDVI": (None, "#00e676"), "B": (0, "#42a5f5"), "G": (1, "#9ccc65"),
    "R": (2, "#ef5350"), "NIR": (3, "#ce93d8"),
}
PALETTE = ["#e57373", "#ba68c8", "#7986cb", "#4db6ac", "#aed581", "#ffd54f",
           "#ff8a65", "#a1887f", "#90a4ae", "#f06292", "#4fc3f7"]


class CurveView(QWidget):
    """Curva NDVI/bandas (vai abaixo do mapa)."""

    def __init__(self, months=None, parent=None):
        super().__init__(parent)
        self.months = months or MONTHS
        self._curve = None
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 2, 4, 2)
        sel = QHBoxLayout()
        sel.addWidget(QLabel("Show:"))
        self._checks = {}
        for name in BAND_STYLE:
            cb = QCheckBox(name)
            cb.setChecked(name == "NDVI")
            cb.setStyleSheet(f"color:{BAND_STYLE[name][1]}; font-weight:bold;")
            cb.stateChanged.connect(self._replot)
            sel.addWidget(cb)
            self._checks[name] = cb
        sel.addStretch()
        self.toggle_row = sel          # exposto p/ subclasses (ex.: InspectCurveView) anexarem controles
        root.addLayout(sel)
        self.plot = pg.PlotWidget()
        self.plot.setBackground("#141416")
        self.plot.setYRange(-0.1, 1.0)
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.plot.getAxis("bottom").setTicks([list(enumerate(self.months))])
        self.plot.addLegend(offset=(-8, 8))
        root.addWidget(self.plot)

    def set_curve(self, curve):
        self._curve = curve
        self._replot()

    def _replot(self):
        self.plot.clear()
        if self._curve is None:
            return
        curve = np.asarray(self._curve, float)
        x = np.arange(curve.shape[0])
        for name, cb in self._checks.items():
            if not cb.isChecked():
                continue
            idx, color = BAND_STYLE[name]
            y = to_ndvi(curve) if idx is None else curve[:, idx]
            m = np.isfinite(y)
            if idx is None:
                self.plot.plot(x[m], y[m], pen=pg.mkPen(color, width=3), name=name, fillLevel=0,
                               brush=pg.mkBrush(0, 230, 118, 45), symbol="o", symbolSize=6, symbolBrush=color)
            else:
                self.plot.plot(x[m], y[m], pen=pg.mkPen(color, width=2), name=name)

    def clear(self):
        self._curve = None
        self.plot.clear()


class ClassCard(QPushButton):
    """Card por classe: 2 barras — predição (cor) + similaridade (cinza).

    ``key``: atalho de teclado (dígito) mostrado antes do nome; None = sem atalho.
    ``indent``/``rail_color``: filha de um grupo entra deslocada, com um trilho
    vertical na cor da família ligando-a ao cabeçalho do pai.
    ``top_level``: classe única (sem super multi-membro) — ganha a MESMA faixa de
    cor forte à esquerda que os cabeçalhos, marcando "nível de superclasse".
    """

    def __init__(self, name, color, key=None, indent=0, rail_color=None, top_level=False):
        super().__init__()
        self.name = name
        self.key = key
        self.indent = indent
        self.rail_color = QColor(rail_color) if rail_color else None
        self.top_level = top_level
        self.color = QColor(color)
        self.sim = None
        self.pred = None
        self.top = False
        self.labeled = False
        self.show_sim = True
        self.show_pred = True
        self.setFixedHeight(42)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_state(self, sim, pred, top, labeled, show_sim, show_pred):
        self.sim, self.pred, self.top, self.labeled = sim, pred, top, labeled
        self.show_sim, self.show_pred = show_sim, show_pred
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        full = self.rect()
        # trilho vertical na cor da família, ligando a filha ao cabeçalho do pai
        if self.rail_color is not None and self.indent > 0:
            p.setPen(Qt.PenStyle.NoPen)
            rc = QColor(self.rail_color)
            rc.setAlpha(150)
            p.setBrush(rc)
            railx = 2 + self.indent - 10
            p.drawRoundedRect(QRectF(railx, full.top() + 1, 3, full.height() - 2), 1.5, 1.5)
        r = QRectF(full.adjusted(2 + self.indent, 1, -2, -1))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#34343a" if self.labeled else "#2b2b2e"))
        p.drawRoundedRect(r, 9, 9)
        # classe de topo (única): faixa de cor forte à esquerda, como os cabeçalhos
        text_x = r.x() + 30
        if self.top_level:
            p.setBrush(self.color)
            p.drawRoundedRect(QRectF(r.x() + 3, r.y() + 5, 5, r.height() - 10), 2, 2)
            chip = QRectF(r.x() + 14, r.y() + r.height() / 2 - 7, 14, 14)
            p.drawRoundedRect(chip, 4, 4)
            text_x = r.x() + 36
        else:
            chip = QRectF(r.x() + 8, r.y() + r.height() / 2 - 8, 16, 16)
            p.setBrush(self.color)
            p.drawEllipse(chip)
        f = p.font()
        p.setPen(QColor("#ffffff"))
        f.setBold(True)
        f.setPointSize(10)
        p.setFont(f)
        nm = ("✔ " + self.name) if self.labeled else self.name
        p.drawText(QRectF(text_x, r.y(), r.width() * 0.42, r.height()),
                   int(Qt.AlignmentFlag.AlignVCenter), nm)
        # barras à direita: pred (cor, em cima) + sim (cinza, embaixo)
        bx = r.x() + r.width() * 0.46
        bw = r.width() * 0.34
        p.setPen(Qt.PenStyle.NoPen)
        if self.show_pred and self.pred is not None:
            c = QColor(self.color)
            c.setAlpha(215)
            p.setBrush(c)
            p.drawRoundedRect(QRectF(bx, r.y() + r.height() * 0.20, max(2.0, bw * self.pred / 100.0), 7), 3, 3)
        if self.show_sim and self.sim is not None:
            p.setBrush(QColor(150, 150, 150, 170))
            p.drawRoundedRect(QRectF(bx, r.y() + r.height() * 0.58, max(2.0, bw * self.sim / 100.0), 7), 3, 3)
        # textos % à direita
        f.setBold(False)
        f.setPointSize(8)
        p.setFont(f)
        p.setPen(QColor("#e0e0e0"))
        if self.show_pred and self.pred is not None:
            p.drawText(QRectF(r.x(), r.y(), r.width() - 9, r.height() * 0.5),
                       int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter), f"P {self.pred}%")
        if self.show_sim and self.sim is not None:
            p.drawText(QRectF(r.x(), r.y() + r.height() * 0.5, r.width() - 9, r.height() * 0.5),
                       int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter), f"S {self.sim}%")
        # bordas
        if self.labeled:
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(QColor("#ffffff"), 3))
            p.drawRoundedRect(r, 9, 9)
        elif self.top:
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(QColor(self.color).lighter(140), 2))
            p.drawRoundedRect(r, 9, 9)
        p.end()


class SuperCard(QPushButton):
    """Cabeçalho de superclasse como CARD full-width: faixa de cor forte à esquerda,
    seta de recolher, chip, NOME (n) e a barra de probabilidade do modelo no nível
    super. Clique na SETA recolhe/expande; clique no CORPO rotula como a superclasse.
    """

    def __init__(self, sup, color, count):
        super().__init__()
        self.sup = sup
        self.color = QColor(color)
        self.count = count
        self.collapsed = False
        self.pred = None
        self.top = False
        self.setFixedHeight(40)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._toggle_cb = None
        self._label_cb = None
        self._ARROW_HIT = 30

    def set_state(self, pred, top, collapsed):
        self.pred, self.top, self.collapsed = pred, top, collapsed
        self.update()

    def mousePressEvent(self, ev):
        x = ev.position().x() if hasattr(ev, "position") else ev.x()
        if x < self._ARROW_HIT and self._toggle_cb:
            self._toggle_cb()
        elif self._label_cb:
            self._label_cb()

    # atalhos p/ teste (evita simular evento de mouse por pixel)
    def toggle_click(self):
        if self._toggle_cb:
            self._toggle_cb()

    def label_click(self):
        if self._label_cb:
            self._label_cb()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect().adjusted(2, 1, -2, -1))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#33333b"))                       # levemente mais claro = "pai"
        p.drawRoundedRect(r, 9, 9)
        # faixa de cor FORTE à esquerda (marca de superclasse)
        p.setBrush(self.color)
        p.drawRoundedRect(QRectF(r.x() + 3, r.y() + 5, 5, r.height() - 10), 2, 2)
        # seta ▾/▸ (zona de clique = recolher)
        f = p.font()
        p.setPen(QColor("#c8c8d0"))
        f.setBold(True)
        f.setPointSize(9)
        p.setFont(f)
        p.drawText(QRectF(r.x() + 12, r.y(), 16, r.height()),
                   int(Qt.AlignmentFlag.AlignVCenter), "▾" if not self.collapsed else "▸")
        # chip quadrado forte
        p.setBrush(self.color)
        p.drawRoundedRect(QRectF(r.x() + 30, r.y() + r.height() / 2 - 7, 14, 14), 4, 4)
        # NOME (n) em maiúsculas
        p.setPen(QColor("#ffffff"))
        f.setPointSize(10)
        p.setFont(f)
        p.drawText(QRectF(r.x() + 52, r.y(), r.width() * 0.5, r.height()),
                   int(Qt.AlignmentFlag.AlignVCenter), f"{self.sup.upper()}  ({self.count})")
        # barra + texto da probabilidade do modelo no nível super
        if self.pred is not None:
            bx = r.x() + r.width() * 0.62
            bw = r.width() * 0.20
            c = QColor(self.color)
            c.setAlpha(230)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(c)
            p.drawRoundedRect(QRectF(bx, r.y() + r.height() / 2 - 3.5, max(2.0, bw * self.pred / 100.0), 7), 3, 3)
            f.setBold(False)
            f.setPointSize(8)
            p.setFont(f)
            p.setPen(QColor("#e8e8e8"))
            p.drawText(QRectF(r.x(), r.y(), r.width() - 10, r.height()),
                       int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter), f"P {self.pred}%")
        # destaque quando o super é o top do modelo
        if self.top:
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(QColor(self.color).lighter(140), 2))
            p.drawRoundedRect(r, 9, 9)
        p.end()


class SimilarityPanel(QWidget):
    """Cards de classe (similaridade + rótulo) — painel lateral."""

    classChosen = pyqtSignal(str)
    removeRequested = pyqtSignal()
    classAdded = pyqtSignal(str)

    def __init__(self, classes: dict, parent=None):
        super().__init__(parent)
        self.classes = dict(classes)
        self._sim = {}
        self._pred = {}
        self._show_sim = True
        self._show_pred = True
        self._labeled = None
        root = QVBoxLayout(self)
        root.setSpacing(6)

        self.info = QLabel("")
        self.info.setStyleSheet("color:#bbb;")
        self.info.setVisible(False)          # sem placeholder: só ocupa espaço quando há info real
        root.addWidget(self.info)

        head = QHBoxLayout()
        head.addWidget(QLabel("<b>Classes</b>"))
        head.addStretch()
        self.pred_toggle = QCheckBox("predição")
        self.pred_toggle.setChecked(True)
        self.pred_toggle.setStyleSheet("color:#dcdcaa;")
        self.pred_toggle.stateChanged.connect(lambda _: self._refresh())
        head.addWidget(self.pred_toggle)
        self.sim_toggle = QCheckBox("similaridade")
        self.sim_toggle.setChecked(True)
        self.sim_toggle.setToolTip(
            "Barra 'S': quão parecida a curva é com cada classe considerando "
            "APENAS os pontos que VOCÊ rotulou neste dataset. Classe sem nenhum "
            "ponto rotulado não aparece (não há com o que comparar).")
        self.sim_toggle.stateChanged.connect(lambda _: self._refresh())
        head.addWidget(self.sim_toggle)
        root.addLayout(head)

        self._host = QWidget()
        self._cards_box = QVBoxLayout(self._host)
        self._cards_box.setSpacing(4)
        self._cards_box.setContentsMargins(0, 0, 0, 0)
        self._cards_box.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._host)
        scroll.setStyleSheet("QScrollArea{border:none;}")
        root.addWidget(scroll, stretch=1)

        self._cards = {}
        self._headers = {}          # super -> SuperCard (cabeçalho)
        self._super_cards = {}      # super -> [cards dos filhos]
        self._collapsed = set()     # supers colapsados
        self._class_super = {}      # classe -> super (setado por set_hierarchy)
        self._super_colors = {}
        self._rebuild_cards()

        addb = QPushButton("+ classe")
        addb.clicked.connect(self._add_class_dialog)
        root.addWidget(addb)
        self.remove_btn = QPushButton("✕ remover rótulo")
        self.remove_btn.setStyleSheet("background:#5a2020; padding:6px; border-radius:6px;")
        self.remove_btn.clicked.connect(lambda: self.removeRequested.emit())
        self.remove_btn.setVisible(False)
        root.addWidget(self.remove_btn)

    _KEY_ORDER = "1234567890"   # atalhos: 1..9 e 0 pros 10 primeiros cards

    def _make_card(self, name, color, indent=0, rail_color=None, top_level=False):
        i = len(self._cards)
        key = self._KEY_ORDER[i] if i < len(self._KEY_ORDER) else None
        card = ClassCard(name, color, key=key, indent=indent,
                         rail_color=rail_color, top_level=top_level)
        card.clicked.connect(lambda _, n=name: self.classChosen.emit(n))
        self._cards[name] = card
        self._cards_box.insertWidget(self._cards_box.count() - 1, card)
        return card

    # ---- hierarquia: grupos colapsáveis por superclasse ----
    def set_hierarchy(self, class_super, super_colors=None):
        """Define classe->super (e cores de super) e reconstrói os cards agrupados."""
        self._class_super = dict(class_super or {})
        if super_colors:
            self._super_colors = dict(super_colors)
        self._rebuild_cards()

    def _rebuild_cards(self):
        """Reconstrói os cards AGRUPADOS por superclasse: cabeçalho-card colapsável por
        super multi-membro (▾ CULTIVO → 1c/2c/3c com trilho); classes únicas viram card
        de topo (faixa forte) no mesmo nível dos cabeçalhos."""
        for w in list(self._cards.values()) + list(self._headers.values()):
            self._cards_box.removeWidget(w)
            w.setParent(None)
        self._cards = {}
        self._headers = {}          # super -> SuperCard (cabeçalho)
        self._super_cards = {}      # super -> [ClassCard filhos]
        groups = {}
        for name in order_classes(self.classes):
            groups.setdefault(self._class_super.get(name, name), []).append(name)
        # grupos multi-membro PRIMEIRO (cabeçalho + filhas indentadas); classes únicas
        # DEPOIS, como card de topo. order_classes já joga nao_sei p/ o fim.
        multi = [(s, m) for s, m in groups.items() if not (len(m) == 1 and m[0] == s)]
        atomic = [(s, m) for s, m in groups.items() if len(m) == 1 and m[0] == s]
        for sup, members in multi:
            rail = self._super_colors.get(sup, "#c8c8d0")
            self._make_header(sup, members)
            self._super_cards[sup] = []
            for m in members:
                card = self._make_card(m, self.classes[m], indent=18, rail_color=rail)
                self._super_cards[sup].append(card)
                card.setVisible(sup not in self._collapsed)
        for sup, members in atomic:
            self._make_card(members[0], self.classes[members[0]], top_level=True)

    def _make_header(self, sup, members):
        """Cabeçalho-card da superclasse (SuperCard): seta recolhe, corpo rotula no super."""
        col = self._super_colors.get(sup, "#c8c8d0")
        card = SuperCard(sup, col, len(members))
        card.collapsed = sup in self._collapsed
        card.setToolTip(f"clique para ROTULAR como '{sup}' (superclasse); clique na seta p/ recolher")
        card._toggle_cb = lambda s=sup: self._toggle_super(s)
        card._label_cb = lambda s=sup: self.classChosen.emit(s)
        self._headers[sup] = card
        self._cards_box.insertWidget(self._cards_box.count() - 1, card)

    def _toggle_super(self, sup):
        self._collapsed.discard(sup) if sup in self._collapsed else self._collapsed.add(sup)
        vis = sup not in self._collapsed
        for c in self._super_cards.get(sup, []):
            c.setVisible(vis)
        hdr = self._headers.get(sup)
        if hdr:
            hdr.collapsed = not vis
            hdr.update()

    def class_for_key(self, digit: str):
        """Classe do atalho de teclado (dígito '1'..'9','0'), ou None."""
        for name, card in self._cards.items():
            if card.key == digit:
                return name
        return None

    def set_classes(self, classes: dict):
        """Reconstrói os cards (agrupados por super) após renomear/mesclar/remover."""
        self.classes = dict(classes)
        self._rebuild_cards()

    def refresh_colors(self):
        """Repinta os chips após edição de cor (o controller atualiza self.classes antes)."""
        for name, card in self._cards.items():
            if name in self.classes:
                card.color = QColor(self.classes[name])
                card.update()
        for sup, hdr in self._headers.items():          # cabeçalhos: cor forte da família
            col = self._super_colors.get(sup)
            if col:
                hdr.color = QColor(col)
                hdr.update()

    def _add_class_dialog(self):
        name, ok = QInputDialog.getText(self, "Nova classe", "Nome da classe:")
        name = name.strip()
        if ok and name and name not in self.classes:
            self.classes[name] = PALETTE[len(self._cards) % len(PALETTE)]
            self._rebuild_cards()                 # nova classe = super atômico (sem grupo)
            self.classAdded.emit(name)

    def _refresh(self):
        self._show_sim = self.sim_toggle.isChecked()
        self._show_pred = self.pred_toggle.isChecked()
        pred_src = self._pred if (self._pred and self._show_pred) else {}
        src = pred_src if pred_src else (self._sim if self._show_sim else {})
        top = max(src, key=src.get) if src else None
        for name, card in self._cards.items():
            sup = self._class_super.get(name, name)
            under_group = sup in self._super_cards          # filho de um super com cabeçalho
            super_pred = self._pred.get(sup) if under_group else None
            sim = int(round(100 * self._sim[name])) if name in self._sim else None
            if under_group and super_pred is not None:
                pred = None    # modelo prevê NO SUPER → barra vai pro cabeçalho, não no card fino
            else:
                pred = int(round(100 * self._pred[name])) if name in self._pred else None
            card.set_state(sim, pred, name == top, name == self._labeled,
                           self._show_sim, self._show_pred)
        # cabeçalhos: probabilidade do MODELO no nível super (quando ele prevê nesse nível)
        for sup, hdr in self._headers.items():
            sp = self._pred.get(sup) if self._show_pred else None
            pred = int(round(100 * sp)) if sp is not None else None
            hdr.set_state(pred, sup == top, sup in self._collapsed)

    def set_scores(self, sim_scores, pred_scores=None, novel=False, cycle_n=None,
                   existing_class=None, quality=None):
        self._sim = sim_scores or {}
        self._pred = pred_scores or {}
        self._labeled = existing_class
        bits = []
        if cycle_n is not None:
            bits.append(f"ciclos (vale): {int(cycle_n)}")
        if quality is not None:
            bits.append(f"cleanlab: {quality:.2f}")
        if self._pred:
            t = max(self._pred, key=self._pred.get)
            bits.append(f"modelo: {t} {int(round(100 * self._pred[t]))}%")
        elif existing_class is None and novel:
            bits.append("⚠ novidade")
        txt = "   ·   ".join(bits) if bits else "ponto novo"
        self.info.setText(txt)
        self.info.setVisible(bool(txt))
        self._refresh()
        self.remove_btn.setVisible(existing_class is not None)

    def clear(self):
        self._sim = {}
        self._pred = {}
        self._labeled = None
        self.info.setText("")
        self.info.setVisible(False)
        self._refresh()
        self.remove_btn.setVisible(False)
