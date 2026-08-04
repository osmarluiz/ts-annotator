"""ReviewPanel — the points table, the app's review instrument.

Model/view (QAbstractTableModel + QSortFilterProxyModel + QTableView): Qt only
materializes visible rows, so 40k points cost the same as 40.

Columns are a LIST OF SPECS built at construction, so nothing domain-specific is
hardcoded in the generic core:
  - always: id, classe (your label), flag (!= model / suspect)
  - model-dependent (hidden until a training run makes metrics): pred, conf, cleanlab
  - domain features (optional, from the launcher/project): e.g. RIDE passes
    ciclos + amplitude read off the curve. A generic project shows none —
    "cycles" is not universal to time series.

Review MODALITIES filter + sort for a task; the model-dependent ones disable
until there's a model. Double-click a row zooms the map to the point. The panel
holds a REFERENCE to the store's live point list and emits pointSelected(pid) /
pointActivated(pid).
"""

from PyQt6.QtCore import QAbstractTableModel, QModelIndex, QSortFilterProxyModel, Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QTableView,
    QVBoxLayout,
    QWidget,
)

_SORT_ROLE = Qt.ItemDataRole.UserRole + 1
_PID_ROLE = Qt.ItemDataRole.UserRole + 2


class _PointsModel(QAbstractTableModel):
    def __init__(self, cols, fx=None):
        super().__init__()
        self._pts = []
        self._colors = {}
        self._fx = fx
        self._cols = cols       # [{"key","label",("fidx"),("width"),("model")}]

    def set_points(self, points, colors):
        self.beginResetModel()
        self._pts = points
        self._colors = colors
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._pts)

    def columnCount(self, parent=QModelIndex()):
        return len(self._cols)

    def headerData(self, s, orient, role=Qt.ItemDataRole.DisplayRole):
        if orient == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return self._cols[s]["label"]
        return None

    # ---- features do domínio (lazy, cacheadas no ponto por índice) ----
    def _feat(self, p, fidx):
        cache = p.setdefault("_feat", {})
        if fidx not in cache:
            try:
                import numpy as np
                f = self._fx.extract(np.asarray(p["curve"], float)) if (self._fx and "curve" in p) else None
                cache[fidx] = float(f[fidx]) if f is not None else None
            except Exception:
                cache[fidx] = None
        return cache[fidx]

    @staticmethod
    def _disc(p):
        # compara no NÍVEL do treino: _ytrain (super se colapsado) senão a classe fina.
        # Sem isto, modelo treinado por super marcaria todo ponto fino como discordante.
        ref = p.get("_ytrain", p.get("class"))
        return bool(p.get("_pred")) and p["_pred"] != ref

    @staticmethod
    def _susp(p):
        # "suspeito" = ponto que o cleanlab (find_label_issues) marcou como
        # provável erro de rótulo — MESMO conjunto que o contador do treino.
        return bool(p.get("_issue"))

    @staticmethod
    def _lowq(p):
        # nota de qualidade baixa (get_label_quality_scores < 0.5): conjunto
        # MAIOR e contínuo, separado dos issues do cleanlab (ver _susp).
        return p.get("_clean", 1.0) < 0.5

    def _cell(self, p, col, sort=False):
        k = col["key"]
        if k == "id":
            return p["id"]
        if k == "class":
            return p["class"]
        if k == "ytrain":
            # rótulo do ponto NO NÍVEL do treino (super se colapsado); é com ESTE que
            # o pred é comparado — deixa explícito que classe fina ≠ pred não é erro
            return p.get("_ytrain", "")
        if k == "pred":
            return p.get("_pred", "" if not sort else "")
        if k == "conf":
            return p.get("_predp", -1.0) if sort else (f"{p['_predp']:.0%}" if "_predp" in p else "")
        if k == "clean":
            return p.get("_clean", 2.0) if sort else (f"{p['_clean']:.2f}" if "_clean" in p else "")
        if k == "feat":
            v = self._feat(p, col["fidx"])
            if sort:
                return v if v is not None else -1e9
            if v is None:
                return ""
            return str(int(round(v))) if float(v).is_integer() else f"{v:.2f}"
        if k == "flag":
            if sort:
                return (2 if self._disc(p) else 0) + (1 if self._susp(p) else 0)
            return ("≠" if self._disc(p) else "") + ("⚑" if self._susp(p) else "")
        return ""

    def data(self, idx, role=Qt.ItemDataRole.DisplayRole):
        if not idx.isValid():
            return None
        p = self._pts[idx.row()]
        col = self._cols[idx.column()]
        if role == _PID_ROLE:
            return p["id"]
        if role == Qt.ItemDataRole.DisplayRole:
            return self._cell(p, col)
        if role == _SORT_ROLE:
            return self._cell(p, col, sort=True)
        if role == Qt.ItemDataRole.ForegroundRole:
            if col["key"] == "flag" and self._disc(p):
                return QColor("#ff6e6e")
            if col["key"] == "class":
                return QColor(self._colors.get(p["class"], "#e0e0e0"))
            if col["key"] == "pred" and self._disc(p):
                return QColor("#ff9a9a")
        if role == Qt.ItemDataRole.TextAlignmentRole and col["key"] in ("conf", "clean", "feat", "flag"):
            return int(Qt.AlignmentFlag.AlignCenter)
        return None

    def point_at(self, row):
        return self._pts[row]

    def has_metrics(self):
        return any("_clean" in p for p in self._pts)


class _ReviewProxy(QSortFilterProxyModel):
    def __init__(self):
        super().__init__()
        self.setSortRole(_SORT_ROLE)
        self.cls = None
        self.mode = "all"

    def filterAcceptsRow(self, row, parent):
        p = self.sourceModel().point_at(row)
        if self.cls is not None and p["class"] != self.cls:
            return False
        m = self.mode
        if m == "disc":
            return _PointsModel._disc(p)
        if m == "susp":
            return _PointsModel._susp(p)
        if m == "lowq":
            return _PointsModel._lowq(p)
        if m == "lowconf":
            return "_predp" in p and p["_predp"] < 0.5
        if m == "agree":
            return "_predp" in p and not _PointsModel._disc(p) and p["_predp"] >= 0.8
        if m == "nometrics":
            return "_clean" not in p
        return True


class ReviewPanel(QWidget):
    pointSelected = pyqtSignal(int)
    pointActivated = pyqtSignal(int)

    _MODES = [("todos", "all", False),
              ("suspeitos (cleanlab)", "susp", True),
              ("baixa qualidade (nota<0.5)", "lowq", True),
              ("≠ modelo (discordantes)", "disc", True),
              ("baixa confiança do modelo", "lowconf", True),
              ("concordantes OK (validar)", "agree", True),
              ("sem métricas (novos)", "nometrics", False)]

    def __init__(self, classes, fx=None, feature_cols=None):
        """feature_cols: [(label, feature_idx)] extras do domínio (do launcher/projeto)."""
        super().__init__()
        # monta a lista de colunas (nada de domínio hardcoded aqui)
        self.cols = [
            {"key": "id", "label": "id", "width": 60},
            {"key": "class", "label": "classe", "width": 116},
            {"key": "ytrain", "label": "nível", "width": 100, "model": True},
            {"key": "pred", "label": "pred", "width": 104, "model": True},
            {"key": "conf", "label": "conf.", "width": 54, "model": True},
            {"key": "clean", "label": "cleanlab", "width": 70, "model": True},
        ]
        for label, fidx in (feature_cols or []):
            self.cols.append({"key": "feat", "label": label, "fidx": fidx, "width": 62})
        self.cols.append({"key": "flag", "label": "", "width": 40})
        self._flag_col = len(self.cols) - 1
        self._clean_col = next(i for i, c in enumerate(self.cols) if c["key"] == "clean")
        self._conf_col = next(i for i, c in enumerate(self.cols) if c["key"] == "conf")

        L = QVBoxLayout(self)
        L.setContentsMargins(10, 10, 10, 10)
        L.addWidget(QLabel("<b>Revisar pontos</b> — clique no cabeçalho p/ ordenar · "
                           "2 cliques dão zoom"))

        row = QHBoxLayout()
        row.addWidget(QLabel("modo:"))
        self.rmode = QComboBox()
        for label, key, _needs in self._MODES:
            self.rmode.addItem(label, key)
        self.rmode.setToolTip("modalidade de revisão — filtra e ordena p/ a tarefa; "
                              "os modos por modelo liberam após treinar")
        row.addWidget(self.rmode, 1)
        self.rfilter = QComboBox()
        self.rfilter.addItem("todas as classes", None)
        for c in classes:
            self.rfilter.addItem(c, c)
        row.addWidget(self.rfilter, 1)
        L.addLayout(row)

        self.rcount = QLabel("0 ponto(s)")
        self.rcount.setStyleSheet("color:#888;")
        L.addWidget(self.rcount)

        self.model = _PointsModel(self.cols, fx)
        self.proxy = _ReviewProxy()
        self.proxy.setSourceModel(self.model)
        self.table = QTableView()
        self.table.setModel(self.proxy)
        self.table.setSortingEnabled(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setShowGrid(False)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setDefaultSectionSize(22)
        for i, col in enumerate(self.cols):
            self.table.setColumnWidth(i, col["width"])
        self.table.horizontalHeader().setSectionResizeMode(
            self._flag_col, QHeaderView.ResizeMode.Stretch)
        L.addWidget(self.table, 1)

        hint = QLabel("↓/↑ navegam · o mapa segue · 2 cliques dão zoom · corrija com 1–9 · Backspace remove")
        hint.setStyleSheet("color:#777;")
        L.addWidget(hint)

        self.table.selectionModel().currentRowChanged.connect(self._on_sel)
        self.table.doubleClicked.connect(self._on_dbl)
        self.rfilter.currentIndexChanged.connect(self._on_filter)
        self.rmode.currentIndexChanged.connect(self._on_mode)

    def set_points(self, points, colors):
        cur = self.table.currentIndex()
        pid = self.proxy.data(cur.siblingAtColumn(0), _PID_ROLE) if cur.isValid() else None
        row = cur.row() if cur.isValid() else -1
        self.model.set_points(points, colors)
        self._sync_model_dependent()
        if pid is not None:
            sm = self.table.selectionModel()
            sm.blockSignals(True)
            restored = False
            for r in range(self.proxy.rowCount()):
                if self.proxy.data(self.proxy.index(r, 0), _PID_ROLE) == pid:
                    self.table.setCurrentIndex(self.proxy.index(r, 0))
                    restored = True
                    break
            if not restored and self.proxy.rowCount():
                self.table.setCurrentIndex(
                    self.proxy.index(min(max(row, 0), self.proxy.rowCount() - 1), 0))
            sm.blockSignals(False)
        self._update_count()

    def _sync_model_dependent(self):
        has = self.model.has_metrics()
        for i, col in enumerate(self.cols):
            if col.get("model"):
                self.table.setColumnHidden(i, not has)
        mdl = self.rmode.model()
        for i, (_lbl, _key, needs) in enumerate(self._MODES):
            en = (not needs) or has
            mdl.item(i).setEnabled(en)
            mdl.item(i).setText(self._MODES[i][0] + ("" if en else "  (precisa de modelo)"))
        if not mdl.item(self.rmode.currentIndex()).isEnabled():
            self.rmode.setCurrentIndex(0)

    def _update_count(self):
        pts = self.model._pts
        n_disc = sum(1 for p in pts if _PointsModel._disc(p))
        n_susp = sum(1 for p in pts if _PointsModel._susp(p))
        n_lowq = sum(1 for p in pts if _PointsModel._lowq(p))
        shown = self.proxy.rowCount()
        extra = f"  ·  mostrando {shown}" if shown != len(pts) else ""
        self.rcount.setText(f"{len(pts):,} ponto(s)  ·  ≠ modelo: {n_disc}  ·  "
                            f"suspeitos: {n_susp}  ·  nota<0.5: {n_lowq}{extra}".replace(",", "."))

    def _on_filter(self, *_):
        self.proxy.cls = self.rfilter.currentData()
        self.proxy.invalidateFilter()
        self._update_count()

    def _on_mode(self, *_):
        mode = self.rmode.currentData()
        self.proxy.mode = mode
        self.proxy.invalidateFilter()
        if mode in ("susp", "lowq"):
            self.table.sortByColumn(self._clean_col, Qt.SortOrder.AscendingOrder)
        elif mode in ("disc", "agree"):
            self.table.sortByColumn(self._conf_col, Qt.SortOrder.DescendingOrder)
        elif mode == "lowconf":
            self.table.sortByColumn(self._conf_col, Qt.SortOrder.AscendingOrder)
        self._update_count()

    def set_mode(self, mode):
        i = self.rmode.findData(mode)
        if i >= 0 and self.rmode.model().item(i).isEnabled():
            self.rmode.setCurrentIndex(i)

    def _on_sel(self, cur, _prev=None):
        if cur.isValid():
            pid = self.proxy.data(cur.siblingAtColumn(0), _PID_ROLE)
            if pid is not None:
                self.pointSelected.emit(int(pid))

    def _on_dbl(self, idx):
        pid = self.proxy.data(idx.siblingAtColumn(0), _PID_ROLE)
        if pid is not None:
            self.pointActivated.emit(int(pid))

    def rebuild_class_filter(self, classes):
        """Re-sincroniza o combo de filtro por classe após renomear/mesclar/remover."""
        cur = self.rfilter.currentData()
        self.rfilter.blockSignals(True)
        self.rfilter.clear()
        self.rfilter.addItem("todas as classes", None)
        for c in classes:
            self.rfilter.addItem(c, c)
        idx = self.rfilter.findData(cur) if cur in classes else 0
        self.rfilter.setCurrentIndex(max(0, idx))
        self.rfilter.blockSignals(False)
        self.proxy.cls = self.rfilter.currentData()
        self.proxy.invalidateFilter()

    def sort_by_cleanlab(self):
        self.rfilter.setCurrentIndex(0)
        self.set_mode("susp")

    def select_pid(self, pid):
        sm = self.table.selectionModel()
        sm.blockSignals(True)
        for r in range(self.proxy.rowCount()):
            idx = self.proxy.index(r, 0)
            if self.proxy.data(idx, _PID_ROLE) == pid:
                self.table.setCurrentIndex(idx)
                self.table.scrollTo(idx)
                break
        sm.blockSignals(False)
