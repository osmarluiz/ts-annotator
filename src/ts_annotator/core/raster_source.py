"""RasterSource — leitor COG por janela com seleção automática de overview.

Núcleo testável (sem GUI): dada uma janela em coords de pixel full-res e um
tamanho de saída (~tela), lê do overview apropriado (rasterio escolhe quando
out_shape < janela). Nunca carrega a imagem inteira.
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window


class RasterSource:
    def __init__(self, path: str, cache_size: int = 96):
        self.path = str(path)
        self._ds = rasterio.open(self.path)
        self.height = self._ds.height
        self.width = self._ds.width
        self.count = self._ds.count
        self.nodata = self._ds.nodata
        self.dtype = self._ds.dtypes[0]
        self.overviews = self._ds.overviews(1)
        self.transform = self._ds.transform
        self.crs = self._ds.crs
        self._cache: OrderedDict = OrderedDict()
        self._cache_size = cache_size
        self.cache_hits = 0
        self.cache_miss = 0

    def read_view(
        self,
        x0: float,
        y0: float,
        w: float,
        h: float,
        out_w: int,
        out_h: int,
        resampling: Resampling = Resampling.nearest,
    ):
        """Lê a janela [x0,x0+w] x [y0,y0+h] (coords de pixel full-res) num
        array ~(out_h, out_w). Retorna (arr[bands,oh,ow] float/orig, win_usada
        (x0,y0,ww,hh)) ou (None, None) se fora da imagem.
        """
        x0c = max(0, int(np.floor(x0)))
        y0c = max(0, int(np.floor(y0)))
        x1c = min(self.width, int(np.ceil(x0 + w)))
        y1c = min(self.height, int(np.ceil(y0 + h)))
        ww, hh = x1c - x0c, y1c - y0c
        if ww <= 0 or hh <= 0:
            return None, None
        ow = max(1, min(int(out_w), ww))
        oh = max(1, min(int(out_h), hh))
        key = (x0c, y0c, ww, hh, ow, oh, int(resampling))
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            self.cache_hits += 1
            return cached[0], cached[1]
        self.cache_miss += 1
        win = Window(x0c, y0c, ww, hh)
        arr = self._ds.read(
            window=win, out_shape=(self.count, oh, ow), resampling=resampling
        )
        win_tuple = (x0c, y0c, ww, hh)
        self._cache[key] = (arr, win_tuple)
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return arr, win_tuple

    def read_pixel(self, row: int, col: int) -> np.ndarray | None:
        """Valor (todas as bandas) de 1 pixel full-res."""
        if not (0 <= row < self.height and 0 <= col < self.width):
            return None
        return self._ds.read(window=Window(col, row, 1, 1)).reshape(self.count)

    def close(self):
        self._ds.close()

    def __del__(self):
        try:
            self._ds.close()
        except Exception:
            pass
