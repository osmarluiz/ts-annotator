"""ClassManagerDialog — renomear / mesclar / remover classes do dataset ativo.

View burra: lista as classes (cor + nome + contagem) e chama de volta os callables
do controller (que fazem o trabalho pesado — reescrever pontos, atualizar UI,
persistir no project.yaml). Após cada operação, relê o estado e se redesenha.

Callables esperados:
  get_state() -> (classes: dict[nome->hex], counts: dict[nome->int])
  do_rename(old, new) -> (ok: bool, msg: str)      # new existente = mesclar
  do_remove(cls, dest) -> (ok: bool, msg: str)     # dest=None se a classe está vazia
"""
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QIcon, QPixmap
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)


class ClassManagerDialog(QDialog):
    def __init__(self, get_state, do_rename, do_remove, non_training=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Gerenciar classes")
        self.setMinimumWidth(360)
        self._get_state = get_state
        self._do_rename = do_rename
        self._do_remove = do_remove
        self._non_training = set(non_training or ())

        L = QVBoxLayout(self)
        L.addWidget(QLabel("<b>Classes do dataset ativo</b>"))
        L.addWidget(QLabel("<span style='color:#9a9aa2'>renomear · mesclar (renomear p/ "
                           "uma existente) · remover. O modelo/predição só refletem "
                           "depois de <b>retreinar</b>.</span>"))
        self.list = QListWidget()
        self.list.itemSelectionChanged.connect(self._sync_buttons)
        L.addWidget(self.list, 1)

        row = QHBoxLayout()
        self.b_rename = QPushButton("Renomear…")
        self.b_merge = QPushButton("Mesclar em…")
        self.b_remove = QPushButton("Remover…")
        self.b_rename.clicked.connect(self._rename)
        self.b_merge.clicked.connect(self._merge)
        self.b_remove.clicked.connect(self._remove)
        for b in (self.b_rename, self.b_merge, self.b_remove):
            row.addWidget(b)
        L.addLayout(row)
        close = QPushButton("Fechar")
        close.clicked.connect(self.accept)
        L.addWidget(close)

        self.status = QLabel("")
        self.status.setStyleSheet("color:#8fbf8f;")
        self.status.setWordWrap(True)
        L.addWidget(self.status)

        self._reload()

    # -------- estado --------
    def _reload(self):
        self.classes, self.counts = self._get_state()
        sel = self._selected()
        self.list.clear()
        for name, color in self.classes.items():
            n = self.counts.get(name, 0)
            tag = "  · fora do treino" if name in self._non_training else ""
            it = QListWidgetItem(f"{name}   ({n} pts){tag}")
            pm = QPixmap(14, 14)
            pm.fill(QColor(color))
            it.setIcon(QIcon(pm))
            it.setData(Qt.ItemDataRole.UserRole, name)
            self.list.addItem(it)
            if name == sel:
                self.list.setCurrentItem(it)
        self._sync_buttons()

    def _selected(self):
        it = self.list.currentItem()
        return it.data(Qt.ItemDataRole.UserRole) if it else None

    def _others(self, exclude):
        return [c for c in self.classes if c != exclude]

    def _sync_buttons(self):
        has = self._selected() is not None
        multi = len(self.classes) > 1
        self.b_rename.setEnabled(has)
        self.b_merge.setEnabled(has and multi)
        self.b_remove.setEnabled(has)

    def _apply(self, ok, msg):
        self.status.setText(msg)
        if ok:
            self._reload()

    # -------- ações --------
    def _rename(self):
        old = self._selected()
        if not old:
            return
        new, ok = QInputDialog.getText(self, "Renomear classe",
                                       f"Novo nome para '{old}':", text=old)
        new = new.strip()
        if not ok or not new or new == old:
            return
        if new in self.classes:
            if QMessageBox.question(self, "Mesclar?",
                                    f"Já existe '{new}'. Mesclar '{old}' em '{new}'?\n"
                                    f"Os {self.counts.get(old,0)} pontos de '{old}' viram '{new}'."
                                    ) != QMessageBox.StandardButton.Yes:
                return
        self._apply(*self._do_rename(old, new))

    def _merge(self):
        src = self._selected()
        others = self._others(src)
        if not src or not others:
            return
        dst, ok = QInputDialog.getItem(self, "Mesclar em",
                                       f"Mesclar '{src}' em qual classe?", others, 0, False)
        if not ok or not dst:
            return
        self._apply(*self._do_rename(src, dst))

    def _remove(self):
        cls = self._selected()
        if not cls:
            return
        n = self.counts.get(cls, 0)
        if n == 0:
            if QMessageBox.question(self, "Remover", f"Remover a classe vazia '{cls}'?"
                                    ) != QMessageBox.StandardButton.Yes:
                return
            self._apply(*self._do_remove(cls, None))
            return
        # tem pontos: perguntar destino (outra classe OU nao_sei)
        dests = self._others(cls)
        for nt in self._non_training:
            if nt not in dests:
                dests.append(nt)
        if not dests:
            self.status.setText("não há outra classe para mover os pontos — crie/rotule uma antes.")
            return
        dst, ok = QInputDialog.getItem(
            self, "Remover classe",
            f"'{cls}' tem {n} pontos. Mover para qual classe antes de remover?",
            dests, 0, False)
        if not ok or not dst:
            return
        self._apply(*self._do_remove(cls, dst))
