"""Colorizers — turn raster blocks into RGBA for display (pure numpy, testable).

Decoupled from any widget: every map view receives a ``colorize(arr[bands, h, w])
-> rgba[h, w, 4]`` callable. Three ready-made factories cover the workspace's
inference rules (class LUT / continuous colormap / 3-band RGB).
"""
from __future__ import annotations

import numpy as np

# categorical palette for discovered clusters (card ↔ map share the index)
CLUSTER_PALETTE = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990", "#dcbeff",
]


# 11-class default palette (matches the RIDE preview maps)
CLASS_COLORS = [
    "#fee08b", "#74add1", "#313695", "#8c510a", "#4575b4", "#b2182b",
    "#d9ef8b", "#a6d96a", "#1a9850", "#fdae61", "#b8e186",
]


class ClassColorizer:
    """Callable LUT-based colorizer for a class raster, MUTABLE at runtime.

    - ``colors``: hex per class index (0..N-1). Editing a color (``set_color``)
      or toggling a class off (``set_hidden``) rebuilds the LUT in place, so the
      same object handed to a map view repaints on the next ``refresh()`` — no
      need to rewire the view. ``nodata`` and hidden classes -> transparent.
    """

    def __init__(self, colors, nodata=255, hidden=None):
        self.colors = list(colors)
        self.nodata = nodata
        self.hidden = set(hidden or ())   # class INDICES made transparent
        self._build()

    def _build(self):
        lut = np.zeros((256, 4), np.uint8)
        for i, hx in enumerate(self.colors):
            hx = hx.lstrip("#")
            a = 0 if i in self.hidden else 255
            lut[i] = [int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16), a]
        self._lut = lut

    def set_color(self, idx, hexstr):
        if 0 <= idx < len(self.colors):
            self.colors[idx] = hexstr
            self._build()

    def set_hidden(self, hidden):
        self.hidden = set(hidden or ())
        self._build()

    def __call__(self, arr):
        v = arr[0].astype(np.int32)
        rgba = self._lut[np.clip(v, 0, 255)].copy()
        if self.nodata is not None:
            rgba[v == self.nodata, 3] = 0
        return rgba


def make_class_colorizer(colors: list[str], nodata=255, hidden=None):
    """colors: lista de hex por classe (índice 0..N-1). nodata -> transparente.

    Retorna um ``ClassColorizer`` (callable, mas com estado editável — cores e
    classes ocultas podem mudar sem recriar a view)."""
    return ClassColorizer(colors, nodata=nodata, hidden=hidden)


def make_scalar_colorizer(vmin, vmax, cmap="viridis", nodata=None):
    import pyqtgraph as pg
    cm = pg.colormap.get(cmap, source="matplotlib")
    table = cm.getLookupTable(0.0, 1.0, 256)  # (256,3)

    def fn(arr):
        a = arr[0].astype(np.float32)
        t = np.clip((a - vmin) / (vmax - vmin + 1e-9), 0, 1)
        # NaN sobrevive ao clip e vira INT_MIN no cast → IndexError no lookup;
        # zera aqui (o pixel já sai transparente pela máscara `bad` abaixo)
        idx = (np.nan_to_num(t) * 255).astype(np.int32)
        rgb = table[idx]
        rgba = np.dstack([rgb, np.full(a.shape, 255, np.uint8)])
        bad = ~np.isfinite(a)
        if nodata is not None:
            bad |= a == nodata
        rgba[bad, 3] = 0
        return rgba

    return fn


def make_rgb_colorizer(nodata=0):
    def fn(arr):
        rgb = np.transpose(arr[:3], (1, 2, 0)).astype(np.uint8)
        rgba = np.dstack([rgb, np.full(rgb.shape[:2], 255, np.uint8)])
        rgba[(arr[:3].sum(0) == nodata * 3), 3] = 0
        return rgba

    return fn


def make_bands_rgb_colorizer(rgb_idx, lo, hi, nodata=None):
    """RGBA de um raster multibanda CRU (ex.: um mês da série): pega as bandas
    ``rgb_idx`` como (R,G,B) e estica cada uma de [lo,hi] → [0,255]. nodata (ou
    não-finito) → transparente. Basemap de FALLBACK (mês cru) quando o projeto
    ainda não tem visualizações prontas."""
    idx = list(rgb_idx)
    lo = np.asarray(lo, np.float32).reshape(3, 1, 1)
    hi = np.asarray(hi, np.float32).reshape(3, 1, 1)

    def fn(arr):
        a = arr[idx].astype(np.float32)                       # (3, h, w)
        bad = ~np.isfinite(a).all(0)
        if nodata is not None:
            bad |= (a == nodata).any(0)
        t = np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)
        rgb = (np.nan_to_num(t) * 255).astype(np.uint8)
        rgba = np.dstack([np.transpose(rgb, (1, 2, 0)),
                          np.full(a.shape[1:], 255, np.uint8)])
        rgba[bad, 3] = 0
        return rgba

    return fn
