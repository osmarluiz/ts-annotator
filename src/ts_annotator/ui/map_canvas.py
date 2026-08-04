"""MapCanvas — the pyqtgraph items drawn over the TiledMapView and the map-side tools.

Owns the overlay items (annotation markers, proposed/hover highlights, the model
prediction overlay, the semi-transparent basemap overlay), the clickable grid tool
(cells = analysis scope for propose/classify) and the drag-to-draw rect ROI. All of
it is view-only: WHAT to show (marker data, prediction RGBA, which basemap overlay)
is decided by the window, which reaches the items directly (they are public).

`basemaps` is the window's label -> (RasterSource, colorizer) dict, shared by
reference (the dynamic prediction layer is swapped in/out of it by the window).
"""

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QObject, QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QPen
from PyQt6.QtWidgets import QGraphicsRectItem


class MapCanvas(QObject):
    def __init__(self, view, width, height, grid_cols, basemaps):
        super().__init__()
        self.view = view
        self.W = width
        self.H = height
        self.basemaps = basemaps
        self.overlay_state = {"label": None, "alpha": 0.6}
        self.draw = False                      # modo desenhar-retângulo ligado?
        self.cells = set()                     # células do grid selecionadas (escopo)
        self.cell_rects = {}
        self.cell_px = width / grid_cols

        vb = view.vb
        self.markers = pg.ScatterPlotItem(pen=pg.mkPen("w", width=1))
        self.markers.setZValue(60)
        vb.addItem(self.markers)
        self.prop_marker = pg.ScatterPlotItem()  # candidato proposto / ponto em revisão
        self.prop_marker.setZValue(65)
        vb.addItem(self.prop_marker)
        self.hover_marker = pg.ScatterPlotItem()  # realce do ponto sob o cursor
        self.hover_marker.setZValue(66)
        vb.addItem(self.hover_marker)
        self.cursor_marker = pg.ScatterPlotItem()  # crosshair do pixel clicado (foco
        self.cursor_marker.setZValue(67)           # atual — mostra ONDE está, antes de rotular)
        vb.addItem(self.cursor_marker)
        self.cluster_overlay = pg.ImageItem(axisOrder="row-major")  # mapa de clusters (raster)
        self.cluster_overlay.setZValue(46)
        self.cluster_overlay.setVisible(False)
        vb.addItem(self.cluster_overlay)

        # camada de predição (overlay do modelo, com transparência)
        self.pred_overlay = pg.ImageItem(axisOrder="row-major")
        self.pred_overlay.setZValue(45)
        self.pred_overlay.setVisible(False)
        vb.addItem(self.pred_overlay)

        # overlay de basemap (sobrepor um raster à base, com opacidade)
        self.overlay_item = pg.ImageItem(axisOrder="row-major")
        self.overlay_item.setZValue(44)
        self.overlay_item.setVisible(False)
        vb.addItem(self.overlay_item)

        # grid clicável (escopo p/ propor e regen)
        self.grid_lines = pg.PlotDataItem(pen=pg.mkPen("#ffffff", width=1, style=Qt.PenStyle.DashLine),
                                          connect="finite")
        self.grid_lines.setZValue(55)
        self.grid_lines.setVisible(False)
        vb.addItem(self.grid_lines)
        self.draw_grid_lines()

        # ROI do retângulo (escopo do propor) + desenhar por clique-arrasta
        self.rect_roi = pg.RectROI([width * 0.4, height * 0.4], [width * 0.2, height * 0.2],
                                   pen=pg.mkPen("#ffd166", width=2), movable=True)
        self.rect_roi.setZValue(70)
        self.rect_roi.setVisible(False)
        vb.addItem(self.rect_roi)
        self._orig_vb_drag = vb.mouseDragEvent
        vb.mouseDragEvent = self._vb_drag

    def set_cluster_raster(self, labels, rect, palette, alpha=0.75):
        """Mapa de clusters CONTÍGUO: labels[Hs,Ws] (-1=transparente) → rgba com a
        cor de cada cluster; rect=(col0,row0,w,h) posiciona/estica na grade cheia."""
        import numpy as np
        from PyQt6.QtGui import QColor
        Hs, Ws = labels.shape
        lut = np.zeros((max(int(labels.max()) + 1, 1), 3), np.ubyte)
        for i in range(lut.shape[0]):
            c = QColor(palette[i % len(palette)])
            lut[i] = [c.red(), c.green(), c.blue()]
        rgba = np.zeros((Hs, Ws, 4), np.ubyte)
        valid = labels >= 0
        rgba[valid, :3] = lut[labels[valid]]
        rgba[valid, 3] = int(255 * alpha)
        self.cluster_overlay.setImage(rgba, autoLevels=False)
        col0, row0, w, h = rect
        from PyQt6.QtCore import QRectF
        self.cluster_overlay.setRect(QRectF(col0, row0, w, h))
        self.cluster_overlay.setVisible(True)

    def clear_clusters(self):
        self.cluster_overlay.setVisible(False)

    # =====================================================================
    # Grid tool
    # =====================================================================
    def draw_grid_lines(self):
        xs, ys = [], []
        c = 0.0
        while c <= self.W + 1:
            xs += [c, c, np.nan]
            ys += [0, self.H, np.nan]
            c += self.cell_px
        r = 0.0
        while r <= self.H + 1:
            xs += [0, self.W, np.nan]
            ys += [r, r, np.nan]
            r += self.cell_px
        self.grid_lines.setData(xs, ys)

    def set_grid_visible(self, on):
        self.grid_lines.setVisible(on)
        for it in self.cell_rects.values():
            it.setVisible(on)

    def set_cols(self, n):
        for it in self.cell_rects.values():
            self.view.vb.removeItem(it)
        self.cell_rects.clear()
        self.cells.clear()
        self.cell_px = self.W / n
        self.draw_grid_lines()

    def toggle_cell(self, row, col):
        key = (int(row // self.cell_px), int(col // self.cell_px))
        if key in self.cells:
            self.cells.discard(key)
            it = self.cell_rects.pop(key, None)
            if it is not None:
                self.view.vb.removeItem(it)
        else:
            self.cells.add(key)
            cr, cc = key
            rect = QGraphicsRectItem(cc * self.cell_px, cr * self.cell_px, self.cell_px, self.cell_px)
            rect.setBrush(QBrush(QColor(0, 230, 118, 55)))
            rect.setPen(QPen(QColor("#00e676"), 2))
            rect.setZValue(54)
            self.view.vb.addItem(rect)
            self.cell_rects[key] = rect

    def cells_bounds(self):
        """(x0, y0, x1, y1) do bbox das células selecionadas; None se nenhuma."""
        if not self.cells:
            return None
        cells = sorted(self.cells)
        rs = [c[0] for c in cells]
        cs = [c[1] for c in cells]
        return (min(cs) * self.cell_px, min(rs) * self.cell_px,
                (max(cs) + 1) * self.cell_px, (max(rs) + 1) * self.cell_px)

    # =====================================================================
    # Rect ROI / drag-to-draw
    # =====================================================================
    def rect_bounds(self):
        p, s = self.rect_roi.pos(), self.rect_roi.size()
        return p.x(), p.y(), p.x() + s.x(), p.y() + s.y()

    def set_draw(self, on):
        self.draw = on
        try:
            self.view.glw.setCursor(Qt.CursorShape.CrossCursor if on else Qt.CursorShape.ArrowCursor)
        except Exception:
            pass

    def _vb_drag(self, ev, axis=None):
        if self.draw and ev.button() == Qt.MouseButton.LeftButton:
            ev.accept()
            p0 = self.view.vb.mapSceneToView(ev.buttonDownScenePos())
            p1 = self.view.vb.mapSceneToView(ev.scenePos())
            x0, y0 = min(p0.x(), p1.x()), min(p0.y(), p1.y())
            w, h = abs(p1.x() - p0.x()), abs(p1.y() - p0.y())
            if ev.isFinish():
                x0 = max(0.0, min(x0, self.W - 1)); y0 = max(0.0, min(y0, self.H - 1))
                w = max(1.0, min(w, self.W - x0)); h = max(1.0, min(h, self.H - y0))
            self.rect_roi.setVisible(True)
            self.rect_roi.blockSignals(True)
            self.rect_roi.setPos([x0, y0]); self.rect_roi.setSize([max(1.0, w), max(1.0, h)])
            self.rect_roi.blockSignals(False)
        else:
            self._orig_vb_drag(ev, axis)

    # =====================================================================
    # Basemap overlay (raster semi-transparente sobre a base)
    # =====================================================================
    def refresh_overlay(self, *_):
        lbl = self.overlay_state["label"]
        if not lbl or lbl not in self.basemaps:
            self.overlay_item.setVisible(False)
            return
        src, col = self.basemaps[lbl]
        r = self.view.vb.viewRect()
        col0 = int(max(0, r.x())); row0 = int(max(0, r.y()))
        w = int(min(src.width - col0, r.width())); h = int(min(src.height - row0, r.height()))
        if w <= 1 or h <= 1:
            self.overlay_item.setVisible(False)
            return
        step = max(1, int(np.ceil((h * w / 1_500_000) ** 0.5)))
        Ws = max(1, w // step); Hs = max(1, h // step)
        try:
            arr, win = src.read_view(col0, row0, w, h, Ws, Hs)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("overlay n/d: %s", e)
            self.overlay_item.setVisible(False)
            return
        if arr is None:
            self.overlay_item.setVisible(False)
            return
        self.overlay_item.setImage(col(arr), autoLevels=False)
        x0, y0, ww, hh = win
        self.overlay_item.setRect(QRectF(x0, y0, ww, hh))
        self.overlay_item.setOpacity(self.overlay_state["alpha"])
        self.overlay_item.setVisible(True)

    def set_overlay(self, name):
        self.overlay_state["label"] = None if name == "(nenhum)" else name
        self.refresh_overlay()

    def set_overlay_alpha(self, v):
        self.overlay_state["alpha"] = v / 100.0
        self.overlay_item.setOpacity(self.overlay_state["alpha"])
