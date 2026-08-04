"""AnnotatePanel — the 'Annotate' tab: active-learning goal/count controls and the
suggestion list.

A dumb view: it emits proposeRequested / skipRequested / goalChanged. Proposing
(SelectionEngine) lives in the controller; the WORK AREA (viewport/rect/cells/all)
is the shared scope selected on the MapToolbar — this tab only supplies the goal
and consumes the suggestion list (slist/scount, filled by the controller).
update_goals(has_model) is view-only: it enables/disables the goals that need a model.
"""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
import qtawesome as qta

from ts_annotator.app.config import GOAL_GROUPS, GOALS, goal_data


class AnnotatePanel(QWidget):
    proposeRequested = pyqtSignal()
    skipRequested = pyqtSignal()
    goalChanged = pyqtSignal()
    scopeChosen = pyqtSignal(str)   # onde buscar: viewport | rect | grids | all (escopo compartilhado)

    def __init__(self, classes):
        super().__init__()
        L = QVBoxLayout(self)
        L.setContentsMargins(12, 12, 12, 12)
        L.setSpacing(6)

        # ============ seção 1: SUGERIR PONTOS (configurar a busca) ============
        title = QLabel("SUGERIR PONTOS")
        title.setStyleSheet("font-weight:bold; letter-spacing:1px; color:#c8c8d0;")
        L.addWidget(title)

        L.addWidget(QLabel("estratégia"))
        self.goal_combo = QComboBox()
        # dropdown AGRUPADO por requisito: cada grupo abre com um cabeçalho não
        # selecionável (— … —), depois as estratégias daquele requisito. _real_idx
        # guarda quais linhas são estratégias de verdade (p/ update_goals / auto-seleção).
        by_req = {}
        for g in GOALS:
            by_req.setdefault(g[3], []).append(g)
        self._real_idx = []
        for gtitle, reqs in GOAL_GROUPS:
            self._add_header(gtitle)
            for req in reqs:
                for (_lbl, _metric, _order, _req, _hint) in by_req.get(req, []):
                    idx = self.goal_combo.count()
                    self.goal_combo.addItem(_lbl, goal_data(_lbl, _metric, _order, _req))
                    self.goal_combo.setItemData(idx, _hint, Qt.ItemDataRole.ToolTipRole)
                    self._real_idx.append(idx)
        if self._real_idx:
            self.goal_combo.setCurrentIndex(self._real_idx[0])
        L.addWidget(self.goal_combo)
        self.model_note = QLabel("")
        self.model_note.setWordWrap(True)
        self.model_note.setStyleSheet("color:#e0a458;")
        L.addWidget(self.model_note)

        where_row = QHBoxLayout()
        where_row.addWidget(QLabel("onde"))
        self.scope_combo = QComboBox()
        for _key, _lb in [("viewport", "vista atual"), ("rect", "retângulo"),
                          ("grids", "células da grade"), ("all", "imagem toda")]:
            self.scope_combo.addItem(_lb, _key)
        self.scope_combo.setToolTip("área onde propor/buscar (mesmo escopo da aba Classificar)")
        self.scope_combo.currentIndexChanged.connect(
            lambda _: self.scopeChosen.emit(self.scope_combo.currentData()))
        where_row.addWidget(self.scope_combo, 1)
        L.addLayout(where_row)

        tc_row = QHBoxLayout()
        self.tcls_label = QLabel("da classe:")
        tc_row.addWidget(self.tcls_label)
        self.tcls_combo = QComboBox()
        self.tcls_combo.addItems(list(classes))
        tc_row.addWidget(self.tcls_combo, 1)
        L.addLayout(tc_row)

        act = QHBoxLayout()
        act.addWidget(QLabel("quantos"))
        self.count_spin = QSpinBox()
        self.count_spin.setRange(1, 100)
        self.count_spin.setValue(10)
        act.addWidget(self.count_spin)
        act.addSpacing(8)
        self.pb = QPushButton("  Propor")
        self.pb.setObjectName("primary")
        self.pb.setIcon(qta.icon("mdi6.auto-fix", color="#ffffff"))
        self.pb.setToolTip("propõe/busca na área escolhida em 'onde'")
        self.pb.clicked.connect(self.proposeRequested)
        act.addWidget(self.pb, 1)
        L.addLayout(act)
        self.propose_bar = QProgressBar()
        self.propose_bar.setRange(0, 100)
        self.propose_bar.setTextVisible(False)
        self.propose_bar.setVisible(False)
        L.addWidget(self.propose_bar)

        # separador fino entre configurar e a fila
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color:#2a2c34;")
        L.addWidget(sep)

        # ============ seção 2: FILA (trabalhar as sugestões) ============
        self.scount = QLabel("FILA — 0 sugestão(ões)")
        self.scount.setStyleSheet("font-weight:bold; letter-spacing:1px; color:#c8c8d0;")
        L.addWidget(self.scount)
        self.slist = QListWidget()
        self.slist.setToolTip("clique numa sugestão p/ ver a curva; rotule pelos cards ou teclas 1–9")
        L.addWidget(self.slist, 1)
        skip_btn = QPushButton("pular →")
        skip_btn.setToolTip("ir p/ a próxima sugestão sem rotular (tecla Enter)")
        skip_btn.clicked.connect(self.skipRequested)
        L.addWidget(skip_btn)

        # ============ rodapé: status ============
        self.since_label = QLabel("rótulos desde o último treino: 0")
        self.since_label.setStyleSheet("color:#888;")
        L.addWidget(self.since_label)

        self.goal_combo.currentIndexChanged.connect(lambda _: self.goalChanged.emit())

    def _add_header(self, text):
        """Insere uma linha de CABEÇALHO não selecionável (agrupa o dropdown)."""
        idx = self.goal_combo.count()
        self.goal_combo.addItem(text, None)
        item = self.goal_combo.model().item(idx)
        item.setFlags(Qt.ItemFlag.ItemIsEnabled & ~Qt.ItemFlag.ItemIsEnabled)  # nem enabled nem selectable
        item.setData("#8a8a92", Qt.ItemDataRole.ForegroundRole)

    def update_goals(self, has_model, has_curve=False):
        """Todas as estratégias ficam SELECIONÁVEIS; o requisito não atendido vira um
        aviso inline (botão ativo, bloqueio real só ao propor — UX uniforme entre
        'precisa de modelo' e 'precisa de escolha'). Rótulos das de modelo ganham um
        sufixo discreto quando não há modelo."""
        mdl = self.goal_combo.model()
        for i in self._real_idx:
            data = self.goal_combo.itemData(i)
            suffix = "  — sem modelo" if (data["req"] == "model" and not has_model) else ""
            mdl.item(i).setText(data["label"] + suffix)
        cur = self.goal_combo.currentData()
        req = cur["req"] if cur else None
        is_class = req == "class"
        self.tcls_combo.setVisible(is_class)
        self.tcls_label.setVisible(is_class)
        # aviso inline do requisito da estratégia ATUAL (vazio quando atendido)
        note = ""
        if req == "model" and not has_model:
            note = "treine um modelo p/ esta estratégia (ela classifica os candidatos na hora)"
        elif req == "curve" and not has_curve:
            note = "clique numa curva no mapa antes de propor (ela é o protótipo)"
        elif req == "class":
            note = "escolha a classe-alvo abaixo"
        self.model_note.setText(note)

    def set_scope_selected(self, key):
        """Reflete o escopo escolhido em outro lugar (aba Classificar / chip ÁREA)."""
        i = self.scope_combo.findData(key)
        if i >= 0 and i != self.scope_combo.currentIndex():
            self.scope_combo.blockSignals(True)
            self.scope_combo.setCurrentIndex(i)
            self.scope_combo.blockSignals(False)

    def set_since_train(self, n):
        self.since_label.setText(f"rótulos desde o último treino: {n}")
