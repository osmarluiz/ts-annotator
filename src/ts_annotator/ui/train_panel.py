"""TrainPanel — the 'Train' tab: hyper-params + train button.

A dumb view: it exposes the chosen hyper-params (params()) and emits trainRequested.
Training itself (worker, versioned save, metrics writeback) lives in the controller.
Running the model over an area lives in the Classify tab (single home for inference).
"""

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
import qtawesome as qta

from ts_annotator.app.theme import ICON


class TrainPanel(QWidget):
    trainRequested = pyqtSignal()
    reviewSuspectsRequested = pyqtSignal()

    def __init__(self):
        super().__init__()
        L = QVBoxLayout(self)
        L.setContentsMargins(12, 12, 12, 12)
        L.setSpacing(8)
        L.addWidget(QLabel("<b>Treinar</b>"))

        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("modelo"))
        self.model_combo = QComboBox()
        # o catálogo é a IT nativa + o que o tsai (opcional) exportar + os
        # estimadores sklearn/xgboost; sem os opcionais a lista só encolhe
        from ts_annotator.core.architectures import available
        nice = {"it": "InceptionTime (IT)",
                "random_forest": "Random Forest (scikit-learn)",
                "extra_trees": "Extra Trees (scikit-learn)",
                "hist_gradient_boosting": "Gradient Boosting (scikit-learn)",
                "xgboost": "XGBoost"}
        for a in available():
            self.model_combo.addItem(nice.get(a, a), a)
        model_row.addWidget(self.model_combo, 1)
        L.addLayout(model_row)

        lr_row = QHBoxLayout()
        lr_row.addWidget(QLabel("learning rate"))
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setDecimals(4)
        self.lr_spin.setRange(0.0001, 0.1)
        self.lr_spin.setSingleStep(0.0005)
        self.lr_spin.setValue(0.001)
        lr_row.addWidget(self.lr_spin)
        L.addLayout(lr_row)

        ep_row = QHBoxLayout()
        ep_row.addWidget(QLabel("epochs"))
        self.ep_spin = QSpinBox()
        self.ep_spin.setRange(10, 600)
        self.ep_spin.setValue(120)
        ep_row.addWidget(self.ep_spin)
        L.addLayout(ep_row)

        fold_row = QHBoxLayout()
        fold_row.addWidget(QLabel("folds (spatial-CV)"))
        self.fold_spin = QSpinBox()
        self.fold_spin.setRange(2, 10)
        self.fold_spin.setValue(5)
        fold_row.addWidget(self.fold_spin)
        L.addLayout(fold_row)

        # blocos espaciais: 0 = a fórmula automática de sempre
        blk_row = QHBoxLayout()
        blk_row.addWidget(QLabel("blocos (0=auto)"))
        self.blk_spin = QSpinBox()
        self.blk_spin.setRange(0, 500)
        self.blk_spin.setValue(0)
        self.blk_spin.setToolTip("nº de blocos espaciais da CV. 0 usa a fórmula automática.")
        blk_row.addWidget(self.blk_spin)
        L.addLayout(blk_row)

        # zona morta: bloco sozinho NÃO garante separação (células vizinhas se
        # tocam). O buffer descarta do treino quem estiver perto da validação.
        buf_row = QHBoxLayout()
        buf_row.addWidget(QLabel("buffer (px)"))
        self.buf_spin = QSpinBox()
        self.buf_spin.setRange(0, 100000)
        self.buf_spin.setValue(0)
        self.buf_spin.setToolTip("separação mínima garantida entre treino e validação. "
                                 "0 mantém o comportamento anterior.")
        buf_row.addWidget(self.buf_spin)
        L.addLayout(buf_row)

        # nível de treino: classes finas | superclasses | personalizado (colapsa por super)
        lvl_row = QHBoxLayout()
        lvl_row.addWidget(QLabel("nível"))
        self.level_combo = QComboBox()
        self.level_combo.addItem("classes finas", "fine")
        self.level_combo.addItem("superclasses", "super")
        self.level_combo.addItem("personalizado", "custom")
        self.level_combo.setToolTip("finas = cada classe; superclasses = colapsa todos os grupos; "
                                    "personalizado = escolha quais grupos colapsar")
        self.level_combo.currentIndexChanged.connect(self._on_level)
        lvl_row.addWidget(self.level_combo, 1)
        L.addLayout(lvl_row)
        self._collapse_box = QVBoxLayout()      # 1 checkbox por super multi-membro (só no personalizado)
        L.addLayout(self._collapse_box)
        self._super_cb = {}

        # o treino exige >=30 pontos com curva válida — mostrar o saldo ANTES do clique
        self.pts_label = QLabel("")
        self.pts_label.setStyleSheet("color:#888;")
        L.addWidget(self.pts_label)

        self.train_btn = QPushButton("  treinar")
        self.train_btn.setObjectName("primary")
        self.train_btn.setIcon(qta.icon("mdi6.brain", color="#ffffff"))
        self.train_btn.clicked.connect(self.trainRequested)
        L.addWidget(self.train_btn)

        self.train_status = QLabel("")
        self.train_status.setStyleSheet("color:#bbb;")
        self.train_status.setWordWrap(True)
        L.addWidget(self.train_status)

        self.train_metrics = QLabel("")
        self.train_metrics.setWordWrap(True)
        self.train_metrics.setStyleSheet("color:#9ccc65;")
        L.addWidget(self.train_metrics)

        # atalho pro fluxo cleanlab: aparece após o treino, já abre o Revisar filtrado
        self.review_btn = QPushButton("  revisar os suspeitos →")
        self.review_btn.setIcon(qta.icon("mdi6.flag-outline", color=ICON))
        self.review_btn.setToolTip("abre a aba Revisar ordenada por suspeita do cleanlab (pior primeiro)")
        self.review_btn.setVisible(False)
        self.review_btn.clicked.connect(self.reviewSuspectsRequested)
        L.addWidget(self.review_btn)
        L.addStretch()

    def params(self):
        """(learning rate, epochs, folds) escolhidos."""
        return self.lr_spin.value(), self.ep_spin.value(), self.fold_spin.value()

    def cv_config(self):
        """(n_blocks, buffer_px) da spatial-CV. n_blocks None = automático."""
        return (self.blk_spin.value() or None), float(self.buf_spin.value())

    def set_superclasses(self, super_members):
        """super_members: {super: [classes...]}. Cria um checkbox de colapso por super
        MULTI-membro (os de 1 membro não têm o que colapsar)."""
        while self._collapse_box.count():
            it = self._collapse_box.takeAt(0)
            if it.widget() is not None:
                it.widget().setParent(None)
        self._super_cb = {}
        for s, members in super_members.items():
            if len(members) <= 1:
                continue
            cb = QCheckBox(f"colapsar {s}  ({' + '.join(members)})")
            cb.setChecked(True)
            self._super_cb[s] = cb
            self._collapse_box.addWidget(cb)
        self._on_level()

    def _on_level(self, *_):
        custom = self.level_combo.currentData() == "custom"
        for cb in self._super_cb.values():
            cb.setVisible(custom)

    def collapse_config(self):
        """Conjunto de superclasses a COLAPSAR no treino, conforme o nível escolhido."""
        mode = self.level_combo.currentData()
        if mode == "fine":
            return set()
        if mode == "super":
            return set(self._super_cb)          # todos os grupos multi-membro
        return {s for s, cb in self._super_cb.items() if cb.isChecked()}  # personalizado

    def model_type(self):
        """Arquitetura escolhida (só 'it' por enquanto; extensível)."""
        return self.model_combo.currentData()

    def set_points_info(self, n_with_curve):
        self.pts_label.setText(f"{n_with_curve} pontos válidos (mín. 30)")

    def show_review_button(self, n_suspects):
        self.review_btn.setText(f"  revisar os {n_suspects} suspeitos →")
        self.review_btn.setVisible(n_suspects > 0)

    def set_status(self, text):
        self.train_status.setText(text)

    def set_metrics(self, text):
        self.train_metrics.setText(text)
