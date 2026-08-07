"""TiledMapView — tiled pyramid (slippy-map) rendering p/ navegação gigapixel butter-smooth.

Em vez de uma imagem que é refeita a cada gesto, mantém uma GRADE de tiles por
nível de overview: pan carrega só os tiles novos (os existentes ficam); ao trocar
de nível, os tiles do nível antigo seguem embaixo (z menor) enquanto os novos
carregam em cima. Cache LRU. Nada nunca "refaz a imagem toda".
"""
from __future__ import annotations

import math
from collections import OrderedDict

import pyqtgraph as pg
from PyQt6.QtCore import (
    QObject,
    QPointF,
    QRectF,
    Qt,
    QThread,
    QTimer,
    pyqtSignal,
    pyqtSlot,
)
from PyQt6.QtGui import QPainterPath, QPolygonF
from PyQt6.QtWidgets import QGraphicsPathItem, QToolButton, QVBoxLayout, QWidget

from ts_annotator.core.raster_source import RasterSource

pg.setConfigOptions(imageAxisOrder="row-major", background="#111111", antialias=False)


class _VectorOverlay:
    """Uma camada vetorial no mapa: CONTORNO (PlotDataItem) + PREENCHIMENTO
    (QGraphicsPathItem por baixo). Duas cores independentes. O fill só é
    recomputado quando tem pincel (cor) definido — sem custo se for só contorno."""

    def __init__(self, vb, layer, pen, brush=None):
        self.layer = layer
        self.fill = QGraphicsPathItem()
        self.fill.setPen(pg.mkPen(None))
        self.fill.setBrush(brush or pg.mkBrush(None))
        self.fill.setZValue(49)
        self.outline = pg.PlotDataItem(pen=pen or pg.mkPen("#00e5ff", width=1),
                                       connect="finite")
        self.outline.setZValue(50)
        self._has_fill = brush is not None and brush.style() != Qt.BrushStyle.NoBrush
        vb.addItem(self.fill)
        vb.addItem(self.outline)

    def setVisible(self, on):
        self.fill.setVisible(on)
        self.outline.setVisible(on)

    def setPen(self, pen):
        self.outline.setPen(pen)

    def setBrush(self, brush):
        self.fill.setBrush(brush)
        self._has_fill = brush is not None and brush.style() != Qt.BrushStyle.NoBrush
        if not self._has_fill:
            self.fill.setPath(QPainterPath())

    def update_geom(self, x0, y0, x1, y1):
        xs, ys = self.layer.query_outlines(x0, y0, x1, y1)
        self.outline.setData(xs, ys)
        if not self._has_fill:
            return
        path = QPainterPath()
        for cols, rows in self.layer.query_polys(x0, y0, x1, y1):
            path.addPolygon(QPolygonF([QPointF(float(c), float(r))
                                       for c, r in zip(cols, rows)]))
        self.fill.setPath(path)


class _TileWorker(QObject):
    """Lê 1 tile (handle próprio, thread-safe) e emite (key, arr, win)."""

    done = pyqtSignal(object, object, object)

    def __init__(self):
        super().__init__()
        self._src = None

    @pyqtSlot(object, object, str)
    def read(self, key, params, path):
        try:
            if self._src is None or self._src.path != path:
                if self._src is not None:
                    self._src.close()
                self._src = RasterSource(path)
            arr, win = self._src.read_view(*params)
            self.done.emit(key, arr, win)
        except Exception:
            self.done.emit(key, None, None)

    @pyqtSlot()
    def invalidate(self):
        """Fecha o handle: o mesmo path pode ter dado novo (ex.: pred raster crescendo)."""
        if self._src is not None:
            self._src.close()
            self._src = None


class TiledMapView(QWidget):
    pixelClicked = pyqtSignal(int, int)  # (row, col)
    _reqTile = pyqtSignal(object, object, str)
    _invTile = pyqtSignal()
    _reqOvTile = pyqtSignal(object, object, str)
    _invOvTile = pyqtSignal()

    TILE = 256       # tamanho do tile em pixels de nível
    MAX_TILES = 320  # teto do cache LRU de tiles na cena

    def __init__(self, source: RasterSource, colorize, parent=None):
        super().__init__(parent)
        self.source = source
        self.colorize = colorize
        self.factors = [1] + list(source.overviews)  # decimações: [1,2,4,8,16,32]

        self.glw = pg.GraphicsLayoutWidget()
        self.vb = self.glw.addViewBox()
        self.vb.setAspectLocked(True)
        self.vb.invertY(True)
        self.vb.setRange(xRange=(0, source.width), yRange=(0, source.height), padding=0)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.glw)

        self.tiles: OrderedDict = OrderedDict()  # key=(gen,L,tx,ty) -> ImageItem (LRU)
        self.requested: set = set()
        self._overlays = []
        self._gen = 0  # invalida tiles após troca de source

        self._thread = QThread(self)
        self._worker = _TileWorker()
        self._worker.moveToThread(self._thread)
        self._reqTile.connect(self._worker.read)
        self._invTile.connect(self._worker.invalidate)
        self._worker.done.connect(self._on_tile)
        self._thread.start()

        # ---- predição como OVERLAY TILED: mesmo motor de pirâmide, 2º worker/handle.
        # Renderiza o pred raster em QUALQUER zoom (read_view decima do full-res) e
        # atualiza AO VIVO reabrindo o handle (refresh_tiled_overlay) — substitui o
        # overlay decimado (handle stale) e o pred_overlay volátil p/ a predição.
        self.ov_source = None
        self.ov_colorize = None
        self.ov_tiles: OrderedDict = OrderedDict()
        self.ov_requested: set = set()
        self._ov_gen = 0
        self._ov_alpha = 0.6
        self._ov_thread = QThread(self)
        self._ov_worker = _TileWorker()
        self._ov_worker.moveToThread(self._ov_thread)
        self._reqOvTile.connect(self._ov_worker.read)
        self._invOvTile.connect(self._ov_worker.invalidate)
        self._ov_worker.done.connect(self._on_ov_tile)
        self._ov_thread.start()

        self.vb.sigRangeChanged.connect(self._update)
        self.glw.scene().sigMouseClicked.connect(self._on_click)

        # --- botões de zoom (+/−/enquadrar) — pra quem NÃO tem scroll do mouse
        # (ex.: acesso remoto). Flutuam no canto inferior-direito do mapa.
        self.zoom_ctrl = QWidget(self)
        _zl = QVBoxLayout(self.zoom_ctrl)
        _zl.setContentsMargins(0, 0, 0, 0)
        _zl.setSpacing(3)
        for _txt, _fn, _tip in (("+", self.zoom_in, "aproximar"),
                                ("−", self.zoom_out, "afastar"),
                                ("⤢", self.zoom_fit, "enquadrar tudo")):
            b = QToolButton()
            b.setText(_txt)
            b.setToolTip(_tip)
            b.setFixedSize(32, 32)
            b.clicked.connect(_fn)
            b.setStyleSheet(
                "QToolButton{background:rgba(20,20,26,225); color:#e0e0e6;"
                "border:1px solid #3a3a40; border-radius:6px; font-size:18px;}"
                "QToolButton:hover{background:#2d5a88; color:white;}")
            _zl.addWidget(b)
        self.zoom_ctrl.adjustSize()
        self.zoom_ctrl.raise_()

        QTimer.singleShot(0, self._update)

    # ---------- zoom por botão (sem scroll) ----------
    def zoom_in(self):
        self.vb.scaleBy((0.7, 0.7))     # range menor = mais perto

    def zoom_out(self):
        self.vb.scaleBy((1 / 0.7, 1 / 0.7))

    def zoom_fit(self):
        self.vb.setRange(xRange=(0, self.source.width),
                         yRange=(0, self.source.height), padding=0)

    def _place_zoom(self):
        if hasattr(self, "zoom_ctrl"):
            m = 14
            self.zoom_ctrl.move(self.width() - self.zoom_ctrl.width() - m,
                                self.height() - self.zoom_ctrl.height() - m)
            self.zoom_ctrl.raise_()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._place_zoom()

    # ---------- pirâmide ----------
    def _pick_level(self) -> int:
        r = self.vb.viewRect()
        g = self.vb.geometry()
        need = r.width() / max(64.0, g.width())  # full-res px por px de tela
        L = 0
        for i, f in enumerate(self.factors):
            if f <= need:
                L = i
        return L

    def _update(self, *args):
        L = self._pick_level()
        d = self.factors[L]
        ts = self.TILE * d  # tamanho do tile em px full-res
        r = self.vb.viewRect()
        nx = int(math.ceil(self.source.width / ts))
        ny = int(math.ceil(self.source.height / ts))
        tx0 = max(0, int((r.x() - ts) // ts))
        ty0 = max(0, int((r.y() - ts) // ts))
        tx1 = min(nx, int(math.ceil((r.x() + r.width() + ts) / ts)))
        ty1 = min(ny, int(math.ceil((r.y() + r.height() + ts) / ts)))
        wanted = [(tx, ty) for tx in range(tx0, tx1) for ty in range(ty0, ty1)]
        # fonte SEM overviews + zoom-out: a vista pode pedir dezenas de milhares de
        # tiles full-res (ex.: pred raster antes do job completo) — mais que o cache
        # LRU comporta, e os tiles seriam descartados antes de aparecer (thrash
        # infinito). Orçamento por update, do centro da vista pra fora.
        budget = self.MAX_TILES - 64
        if len(wanted) > budget:
            cx = (r.x() + r.width() / 2) / ts
            cy = (r.y() + r.height() / 2) / ts
            wanted.sort(key=lambda t: (t[0] + 0.5 - cx) ** 2 + (t[1] + 0.5 - cy) ** 2)
            wanted = wanted[:budget]
        for tx, ty in wanted:
            key = (self._gen, L, tx, ty)
            if key in self.tiles:
                self.tiles.move_to_end(key)
                continue
            if key in self.requested:
                continue
            self.requested.add(key)
            self._reqTile.emit(
                key, (tx * ts, ty * ts, ts, ts, self.TILE, self.TILE), self.source.path
            )
        self._prune()
        # mesma grade de tiles (L, ts, wanted) para o overlay tiled da predição
        if self.ov_source is not None:
            for tx, ty in wanted:
                key = ("ov", self._ov_gen, L, tx, ty)
                if key in self.ov_tiles:
                    self.ov_tiles.move_to_end(key)
                    continue
                if key in self.ov_requested:
                    continue
                self.ov_requested.add(key)
                self._reqOvTile.emit(
                    key, (tx * ts, ty * ts, ts, ts, self.TILE, self.TILE), self.ov_source.path
                )
            self._prune_ov()
        self._update_overlays()

    @pyqtSlot(object, object, object)
    def _on_tile(self, key, arr, win):
        self.requested.discard(key)
        if arr is None or key[0] != self._gen:
            return
        item = pg.ImageItem(self.colorize(arr), autoLevels=False)
        x0, y0, ww, hh = win
        item.setRect(QRectF(x0, y0, ww, hh))
        item.setZValue(-key[1])  # nível mais fino (L=0) em cima
        self.vb.addItem(item)
        self.tiles[key] = item
        self._prune()

    def _prune(self):
        while len(self.tiles) > self.MAX_TILES:
            _, it = self.tiles.popitem(last=False)
            self.vb.removeItem(it)

    @pyqtSlot(object, object, object)
    def _on_ov_tile(self, key, arr, win):
        self.ov_requested.discard(key)
        if arr is None or key[1] != self._ov_gen:
            return
        item = pg.ImageItem(self.ov_colorize(arr), autoLevels=False)
        x0, y0, ww, hh = win
        item.setRect(QRectF(x0, y0, ww, hh))
        item.setZValue(30 - key[2])   # acima da base (z=-L), abaixo dos itens do MapCanvas (44+)
        item.setOpacity(self._ov_alpha)
        self.vb.addItem(item)
        self.ov_tiles[key] = item
        self._prune_ov()

    def _prune_ov(self):
        while len(self.ov_tiles) > self.MAX_TILES:
            _, it = self.ov_tiles.popitem(last=False)
            self.vb.removeItem(it)

    # ---------- overlay tiled (predição) ----------
    def set_tiled_overlay(self, source, colorize, alpha=None):
        """Exibe `source` como camada TILED sobreposta (pirâmide + handle próprio),
        renderizada em qualquer zoom e atualizável ao vivo (refresh_tiled_overlay)."""
        self._ov_gen += 1
        for _, it in list(self.ov_tiles.items()):
            self.vb.removeItem(it)
        self.ov_tiles.clear()
        self.ov_requested.clear()
        self._invOvTile.emit()
        self.ov_source = source
        self.ov_colorize = colorize
        if alpha is not None:
            self._ov_alpha = alpha
        self._update()

    def clear_tiled_overlay(self):
        self._ov_gen += 1
        for _, it in list(self.ov_tiles.items()):
            self.vb.removeItem(it)
        self.ov_tiles.clear()
        self.ov_requested.clear()
        self.ov_source = None
        self.ov_colorize = None

    def set_tiled_overlay_alpha(self, alpha):
        self._ov_alpha = alpha
        for it in self.ov_tiles.values():
            it.setOpacity(alpha)

    def refresh_tiled_overlay(self):
        """Re-lê a fonte do overlay (o raster cresceu no disco — job de classify):
        invalida geração + handle, mantém tiles atuais até os novos chegarem."""
        if self.ov_source is None:
            return
        self._ov_gen += 1
        self.ov_requested.clear()
        self._invOvTile.emit()
        self._update()

    # ---------- overlays ----------
    def add_vector_layer(self, layer, pen=None, brush=None):
        ov = _VectorOverlay(self.vb, layer, pen, brush)
        self._overlays.append(ov)
        self._update_overlays()
        return ov

    def refresh_overlays(self):
        """Recalcula a geometria dos overlays no viewport atual (ex.: ao ligar/
        trocar o preenchimento, que precisa reconstruir o path)."""
        self._update_overlays()

    def _update_overlays(self):
        if not self._overlays:
            return
        r = self.vb.viewRect()
        for ov in self._overlays:
            ov.update_geom(r.x(), r.y(), r.x() + r.width(), r.y() + r.height())

    # ---------- API ----------
    def refresh(self):
        """Re-lê a MESMA fonte (o dado no disco mudou — ex.: raster de predição
        crescendo durante o job). Invalida geração + handle do worker, mas MANTÉM
        os tiles atuais na tela até os novos chegarem (sem piscar); o LRU recolhe
        os antigos conforme os novos entram."""
        self._gen += 1
        self.requested.clear()
        self._invTile.emit()
        self._update()

    def set_source(self, source: RasterSource, colorize):
        self._gen += 1
        for _, it in list(self.tiles.items()):
            self.vb.removeItem(it)
        self.tiles.clear()
        self.requested.clear()
        self._invTile.emit()  # mesmo path pode ter dado novo — força reabrir no worker
        self.source = source
        self.colorize = colorize
        self.factors = [1] + list(source.overviews)
        self.vb.setLimits(xMin=-source.width, xMax=2 * source.width,
                          yMin=-source.height, yMax=2 * source.height)  # zoom-out além da imagem
        self._update()

    def _on_click(self, ev):
        if ev.button() != Qt.MouseButton.LeftButton:
            return
        p = self.vb.mapSceneToView(ev.scenePos())
        col, row = int(p.x()), int(p.y())
        if 0 <= row < self.source.height and 0 <= col < self.source.width:
            self.pixelClicked.emit(row, col)

    def stop(self):
        """Encerra os tile workers (base + overlay). Chamar no teardown do TOP-LEVEL:
        o closeEvent daqui nunca dispara (a view vive num splitter, não é top-level)."""
        for th in (getattr(self, "_thread", None), getattr(self, "_ov_thread", None)):
            try:
                if th is not None:
                    th.quit()
                    th.wait(2000)
            except Exception:
                pass

    def closeEvent(self, ev):
        self.stop()
        super().closeEvent(ev)
