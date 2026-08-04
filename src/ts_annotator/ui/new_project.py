"""NewProjectDialog — wizard de "novo projeto" (Fatia 1 da UI de dados).

View burra: coleta pasta-destino, imagens da série e parâmetros do cubo, e chama
:func:`~ts_annotator.core.project_init.init_project` (toda a lógica/validação mora
lá, testável sem GUI). Ao aceitar, ``self.project_dir`` aponta pro projeto criado —
o :func:`~ts_annotator.app.main.main` chama ``load_workspace`` nele em seguida.

O projeto abre mesmo sem basemaps prontos: o loader cai no fundo cru da série
(mês central). Gerar visualizações/pirâmides pela UI é a Fatia 2.
"""
from __future__ import annotations

import os

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from ts_annotator.core.project_init import ProjectInitError, find_images, init_project


class NewProjectDialog(QDialog):
    """Cria um projeto novo. ``self.project_dir`` = pasta criada (ou None se cancelou)."""

    def __init__(self, parent=None, start_dir=None):
        super().__init__(parent)
        self.setWindowTitle("Novo projeto")
        self.setMinimumWidth(560)
        self.project_dir = None
        self._start = start_dir or os.path.expanduser("~")

        L = QVBoxLayout(self)
        L.addWidget(QLabel("<b>Novo projeto de anotação</b>"))
        L.addWidget(_hint("Escolha onde salvar o projeto e as imagens da série "
                          "(um raster por mês). O projeto abre direto das imagens — "
                          "os basemaps podem ser gerados depois."))

        form = QFormLayout()
        # pasta-destino
        self.ed_dir = QLineEdit()
        b_dir = QPushButton("Escolher…")
        b_dir.clicked.connect(self._pick_dir)
        form.addRow("Pasta do projeto:", _with_button(self.ed_dir, b_dir))
        # nome / período
        self.ed_name = QLineEdit()
        self.ed_name.setPlaceholderText("(padrão: nome da pasta)")
        form.addRow("Nome:", self.ed_name)
        self.ed_period = QLineEdit()
        self.ed_period.setPlaceholderText("ex.: 2023 ou 2022-2023")
        form.addRow("Período:", self.ed_period)
        L.addLayout(form)

        # imagens da série
        L.addWidget(QLabel("<b>Imagens da série</b> "
                           "<span style='color:#9a9aa2'>(ordem = cronológica pelo nome)</span>"))
        row = QHBoxLayout()
        b_folder = QPushButton("Adicionar pasta…")
        b_files = QPushButton("Adicionar arquivos…")
        b_clear = QPushButton("Limpar")
        b_folder.clicked.connect(self._add_folder)
        b_files.clicked.connect(self._add_files)
        b_clear.clicked.connect(self._clear_imgs)
        for b in (b_folder, b_files, b_clear):
            row.addWidget(b)
        row.addStretch(1)
        L.addLayout(row)
        self.lst = QListWidget()
        self.lst.setMaximumHeight(150)
        L.addWidget(self.lst)
        self.lbl_count = _hint("nenhuma imagem selecionada")
        L.addWidget(self.lbl_count)

        # parâmetros do cubo (defaults do cubo B,G,R,NIR do RIDE)
        adv = QFormLayout()
        self.ed_bands = QLineEdit("1,2,3,4")
        self.ed_rgb = QLineEdit("3,2,1")
        self.ed_rgb.setPlaceholderText("bandas R,G,B (1-based) — vazio = automático")
        self.ed_nodata = QLineEdit("65535")
        self.ed_scale = QLineEdit("10000")
        adv.addRow("Bandas:", self.ed_bands)
        adv.addRow("RGB (fundo):", self.ed_rgb)
        adv.addRow("Nodata:", self.ed_nodata)
        adv.addRow("Escala:", self.ed_scale)
        L.addLayout(adv)
        L.addWidget(_hint("Bandas = índices lidos de cada raster. RGB = quais delas "
                          "viram o fundo colorido. Nodata/Escala conforme o cubo."))

        self._imgs = []
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                              | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self._create)
        bb.rejected.connect(self.reject)
        self.btn_ok = bb.button(QDialogButtonBox.StandardButton.Ok)
        self.btn_ok.setText("Criar e abrir")
        L.addWidget(bb)
        self._sync()

    # -------- imagens --------
    def _set_imgs(self, imgs):
        # dedup preservando ordem
        seen, out = set(), []
        for p in imgs:
            ap = os.path.abspath(p)
            if ap not in seen:
                seen.add(ap)
                out.append(ap)
        self._imgs = out
        self.lst.clear()
        for p in out:
            self.lst.addItem(os.path.basename(p))
        n = len(out)
        self.lbl_count.setText(f"{n} imagem(ns) selecionada(s)"
                               + ("" if n == 12 else "  · o padrão do tool é 12 meses"))
        self._sync()

    def _add_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Pasta com as imagens da série", self._start)
        if not d:
            return
        try:
            imgs = find_images(d)
        except ProjectInitError as e:
            QMessageBox.warning(self, "Imagens", str(e))
            return
        if not imgs:
            QMessageBox.information(self, "Imagens", "Nenhum .tif/.tiff nessa pasta.")
            return
        self._start = d
        self._set_imgs(self._imgs + imgs)

    def _add_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Imagens da série", self._start, "Rasters (*.tif *.tiff)")
        if files:
            self._start = os.path.dirname(files[0])
            self._set_imgs(self._imgs + list(files))

    def _clear_imgs(self):
        self._set_imgs([])

    # -------- destino --------
    def _pick_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Pasta do projeto", self._start)
        if d:
            self.ed_dir.setText(d)
            if not self.ed_name.text().strip():
                self.ed_name.setPlaceholderText(f"(padrão: {os.path.basename(d)})")
            self._sync()

    def _sync(self):
        self.btn_ok.setEnabled(bool(self.ed_dir.text().strip()) and bool(self._imgs))

    # -------- criar --------
    def _create(self):
        pdir = self.ed_dir.text().strip()
        if not pdir or not self._imgs:
            return
        try:
            nodata = int(self.ed_nodata.text().strip() or "65535")
            scale = float(self.ed_scale.text().strip() or "10000")
        except ValueError:
            QMessageBox.warning(self, "Parâmetros", "Nodata/Escala devem ser numéricos.")
            return
        yml = os.path.join(os.path.abspath(pdir), "project.yaml")
        overwrite = False
        if os.path.exists(yml):
            if QMessageBox.question(
                    self, "Sobrescrever?",
                    f"Já existe um project.yaml em:\n{pdir}\n\nSobrescrever?"
                    ) != QMessageBox.StandardButton.Yes:
                return
            overwrite = True
        try:
            self.project_dir = init_project(
                pdir, self._imgs,
                name=self.ed_name.text().strip() or None,
                period=self.ed_period.text().strip(),
                bands=self.ed_bands.text().strip(),
                nodata=nodata, scale=scale,
                rgb=self.ed_rgb.text().strip(),
                overwrite=overwrite)
        except ProjectInitError as e:
            QMessageBox.warning(self, "Não deu pra criar", str(e))
            return
        self.accept()


def _hint(text):
    lab = QLabel(f"<span style='color:#9a9aa2'>{text}</span>")
    lab.setWordWrap(True)
    return lab


def _with_button(edit, button):
    from PyQt6.QtWidgets import QWidget
    w = QWidget()
    lay = QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.addWidget(edit, 1)
    lay.addWidget(button)
    return w
