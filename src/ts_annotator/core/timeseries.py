"""TimeSeriesSource — lê a curva (12 meses x bandas) de um pixel, sob demanda.

Flexível: aceita uma lista de rasters mensais (ex.: os 12 .dat da RIDE, com
offset de linha por mês p/ alinhar grids) ou rasters genéricos. Retorna
reflectância (nodata -> NaN). É a fonte do clique->curva.
"""
from __future__ import annotations

import numpy as np
import rasterio
from rasterio.windows import Window


def parse_bands(raw):
    """``timeseries.bands`` do projeto -> ``(índices de leitura, nomes ou None)``.

    Nomes (strings) declaram o que cada canal do cubo É, na ordem do arquivo, e
    os índices de leitura viram 1..n. Inteiros são o formato original, índices
    de banda do raster; quatro deles continuam valendo o contrato óptico
    B,G,R,NIR de sempre, e qualquer outra contagem fica sem nomes. Misturar os
    dois é erro.
    """
    raw = list(raw)
    if raw and all(isinstance(b, str) for b in raw):
        return tuple(range(1, len(raw) + 1)), [str(b) for b in raw]
    if all(isinstance(b, int) for b in raw):
        names = ["B", "G", "R", "NIR"] if len(raw) == 4 else None
        return tuple(raw), names
    raise ValueError(
        f"timeseries.bands mistura nomes e índices: {raw!r} — use só strings ou só ints")


class TimeSeriesSource:
    def __init__(
        self,
        month_paths,
        bands=(1, 2, 3, 4),
        band_names=None,
        row_offsets=None,
        col_offsets=None,
        nodata=65535,
        scale=10000.0,
    ):
        self.paths = list(month_paths)
        self.bands = list(bands)
        self.band_names = list(band_names) if band_names is not None else None
        n = len(self.paths)
        self.row_offsets = list(row_offsets) if row_offsets is not None else [0] * n
        self.col_offsets = list(col_offsets) if col_offsets is not None else [0] * n
        self.nodata = nodata
        self.scale = scale
        self._ds = [rasterio.open(p) for p in self.paths]
        # leitura decimada só é BARATA se os rasters têm overviews; senão, um
        # read_block decimado lê em resolução cheia (péssimo p/ área grande).
        # TODOS os meses precisam ter — um único mês sem overview já força full-res.
        self.has_overviews = bool(self._ds) and all(ds.overviews(1) for ds in self._ds)

    def read_curve(self, row: int, col: int) -> np.ndarray:
        """(n_meses, n_bandas) em reflectância; mês fora/nodata -> NaN."""
        out = np.full((len(self._ds), len(self.bands)), np.nan)
        for mi, ds in enumerate(self._ds):
            r = int(row + self.row_offsets[mi])
            c = int(col + self.col_offsets[mi])
            if not (0 <= r < ds.height and 0 <= c < ds.width):
                continue
            v = ds.read(self.bands, window=Window(c, r, 1, 1)).astype(float).reshape(len(self.bands))
            v[v == self.nodata] = np.nan
            out[mi] = v / self.scale
        return out

    def safe_window(self):
        """(row_min, col_min, row_max, col_max) onde TODOS os meses leem in-bounds.

        É a interseção dos extents mensais (offsets aplicados). Janelas dentro dela
        usam o caminho rápido de read_block (overviews); fora, read_block recai em
        boundless — que lê em RESOLUÇÃO CHEIA e trava a UI em áreas grandes. Quem
        amostra áreas (ex.: SelectionEngine) deve clampar por aqui antes de ler.
        """
        if not self._ds:
            return 0, 0, 0, 0
        row_min = max(0, max(-o for o in self.row_offsets))
        col_min = max(0, max(-o for o in self.col_offsets))
        row_max = min(ds.height - o for ds, o in zip(self._ds, self.row_offsets))
        col_max = min(ds.width - o for ds, o in zip(self._ds, self.col_offsets))
        return row_min, col_min, max(row_min, row_max), max(col_min, col_max)

    def read_block(self, row0, col0, h, w, step=1):
        """Bloco (Hs, Ws, n_meses, n_bandas) em reflectância (nodata/fora -> NaN).

        Lê cada mês UMA vez (janela), decimando por `step` p/ áreas grandes.
        row0/col0 na grade de referência; aplica os offsets de cada mês.
        """
        Hs = max(1, h // step)
        Ws = max(1, w // step)
        nb = len(self.bands)
        out = np.full((Hs, Ws, len(self._ds), nb), np.nan, np.float32)

        def _read_month(mi):
            ds = self._ds[mi]
            r0 = int(row0 + self.row_offsets[mi])
            c0 = int(col0 + self.col_offsets[mi])
            win = Window(c0, r0, w, h)
            # Janela dentro dos limites: leitura NORMAL com out_shape usa os
            # overviews (rápido p/ preview decimado). boundless=True IGNORA
            # overviews e lê sempre em resolução cheia — era o que congelava a UI
            # num rubber-band grande. Só recai no boundless quando a janela cruza
            # a borda (offset por mês pode empurrar pra fora).
            in_bounds = r0 >= 0 and c0 >= 0 and r0 + h <= ds.height and c0 + w <= ds.width
            if in_bounds:
                arr = ds.read(self.bands, window=win, out_shape=(nb, Hs, Ws)).astype(np.float32)
            else:
                arr = ds.read(self.bands, window=win, out_shape=(nb, Hs, Ws),
                              boundless=True, fill_value=self.nodata).astype(np.float32)
            arr[arr == self.nodata] = np.nan
            out[:, :, mi, :] = np.moveaxis(arr, 0, -1) / self.scale

        # Blocos grandes (classify de área/tile): lê os meses EM PARALELO — cada mês
        # é um dataset próprio (GDAL é thread-safe entre datasets distintos) e escreve
        # numa fatia disjunta de `out`. É o gargalo de I/O que o wall-to-wall atacou.
        # Leituras pequenas (curva 1x1, preview) ficam sequenciais: o overhead de
        # thread pool não compensa.
        if Hs * Ws >= 128 * 128 and len(self._ds) > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=min(len(self._ds), 8)) as ex:
                list(ex.map(_read_month, range(len(self._ds))))
        else:
            for mi in range(len(self._ds)):
                _read_month(mi)
        return out

    def close(self):
        for ds in self._ds:
            try:
                ds.close()
            except Exception:
                pass
