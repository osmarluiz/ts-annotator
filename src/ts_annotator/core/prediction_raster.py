"""PredictionRaster — raster de classificação persistido POR MODELO + sidecar de proveniência.

Um GeoTIFF uint8 por modelo (``pred_<modelo>.tif``) no tamanho cheio da imagem, tiled
512, esparso (tiles não escritos não ocupam disco), nodata 255, com colormap das
classes. O valor do pixel é o ÍNDICE na lista ``labs`` do modelo (a ordem do treino).

O sidecar ``pred_<modelo>.json`` grava a proveniência (arquivo do modelo + mtime,
classes → índice, cores) e o progresso do job de lote (``done_tiles``) — é ele que
torna o job retomável e que detecta modelo re-treinado (mtime mudou → predições
velhas não podem misturar com novas → o raster é resetado).
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window


class PredictionRaster:
    NODATA = 255

    def __init__(self, pred_dir, model_path, tile=1024):
        from ts_annotator.core.version_store import model_stem
        stem = model_stem(model_path)   # versão (it_vN) p/ versionados; nome do .pt p/ soltos
        self.dir = pred_dir
        self.model_path = model_path
        self.tile = tile
        self.tif = os.path.join(pred_dir, f"pred_{stem}.tif")
        self.meta_path = os.path.join(pred_dir, f"pred_{stem}.json")

    # ------------------------------------------------------------- sidecar
    def load_meta(self):
        try:
            with open(self.meta_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _write_meta(self, meta):
        tmp = self.meta_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)
        # Windows: os.replace falha (PermissionError) se a UI estiver LENDO o sidecar
        # neste exato instante (load_meta em _refresh_pred_basemap). Janela de µs a
        # cada ~25 tiles do job — sem retry, mataria o lote inteiro por azar de timing.
        for attempt in range(5):
            try:
                os.replace(tmp, self.meta_path)
                return
            except PermissionError:
                time.sleep(0.05 * (attempt + 1))
        os.replace(tmp, self.meta_path)   # última tentativa: se falhar, deixa subir

    def save_progress(self, done_tiles, complete=False):
        meta = self.load_meta() or {}
        meta["done_tiles"] = sorted(done_tiles)
        meta["complete"] = bool(complete)
        self._write_meta(meta)

    def model_stale(self, meta) -> bool:
        """O sidecar pertence a OUTRA versão do modelo?

        Identidade preferida: sha1 do conteúdo do .pt (robusta a cópia/migração —
        mtime muda, o dado não). Sidecars antigos (sem sha1) caem no mtime, como
        antes — assim um raster de horas de job não é invalidado pelo upgrade.
        """
        from ts_annotator.core.version_store import file_sha1
        if meta.get("model_sha1"):
            return meta["model_sha1"] != file_sha1(self.model_path)
        return meta.get("model_mtime") != os.stat(self.model_path).st_mtime

    # -------------------------------------------------- criação / validação
    def ensure(self, width, height, transform, crs, labs, colors):
        """Garante raster + sidecar compatíveis com o modelo ATUAL.

        Se já existem e batem (mesmas classes, mesma identidade do .pt, mesma
        grade), mantém — é o que permite retomar o job de lote. Senão, (re)cria
        zerado. Retorna 'kept', 'created' ou 'reset'.
        """
        st = os.stat(self.model_path)
        meta = self.load_meta()
        if (meta and os.path.exists(self.tif)
                and meta.get("labs") == list(labs)
                and not self.model_stale(meta)
                and meta.get("width") == int(width)
                and meta.get("height") == int(height)):
            return "kept"
        state = "reset" if (meta or os.path.exists(self.tif)) else "created"
        os.makedirs(self.dir, exist_ok=True)
        prof = dict(driver="GTiff", width=int(width), height=int(height), count=1,
                    dtype="uint8", crs=crs, transform=transform, nodata=self.NODATA,
                    tiled=True, blockxsize=512, blockysize=512, compress="deflate",
                    BIGTIFF="IF_SAFER", SPARSE_OK="TRUE")
        with rasterio.open(self.tif, "w", **prof) as ds:
            cmap = {}
            for i, lb in enumerate(labs):
                h = str(colors.get(lb, "#888888")).lstrip("#")
                cmap[i] = (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), 255)
            cmap[self.NODATA] = (0, 0, 0, 0)
            ds.write_colormap(1, cmap)
        from ts_annotator.core.version_store import file_sha1, model_stem
        self._write_meta({
            "model": model_stem(self.model_path),
            "model_file": self.model_path,
            "model_sha1": file_sha1(self.model_path),
            "model_mtime": st.st_mtime,
            "model_size": st.st_size,
            "labs": list(labs),
            "colors": {lb: colors.get(lb, "#888888") for lb in labs},
            "nodata": self.NODATA,
            "width": int(width), "height": int(height),
            "tile": int(self.tile),
            "created": datetime.now().isoformat(timespec="seconds"),
            "done_tiles": [],
            "complete": False,
        })
        return state

    # -------------------------------------------------------------- escrita
    def write_block(self, arr, row0, col0):
        """Grava um bloco uint8 (valores = índice de classe; 255 = sem predição)."""
        arr = np.asarray(arr, np.uint8)
        with rasterio.open(self.tif, "r+") as ds:
            ds.write(arr, 1, window=Window(int(col0), int(row0), arr.shape[1], arr.shape[0]))

    def open_writer(self):
        """Handle r+ de longa duração p/ o job de lote (o worker fecha ao terminar)."""
        return rasterio.open(self.tif, "r+")

    def build_overviews(self, factors=(2, 4, 8, 16, 32)):
        with rasterio.open(self.tif, "r+") as ds:
            ds.build_overviews(list(factors), Resampling.nearest)
