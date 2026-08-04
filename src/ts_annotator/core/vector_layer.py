"""VectorLayer — camada vetorial (shapefile/geojson/gpkg) como overlay no mapa.

Transforma geometrias geo->pixel (coords do MapView) e indexa (STRtree do
shapely 2.0, sem precisar de rtree). `query_outlines(viewport)` devolve arrays
(x,y) com NaN-separadores prontos p/ um pyqtgraph PlotDataItem.
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
from shapely import STRtree
from shapely.geometry import box


class VectorLayer:
    def __init__(self, path: str, transform, crs):
        gdf = gpd.read_file(path)
        if crs is not None:
            gdf = gdf.to_crs(crs)
        inv = ~transform  # affine inversa: (col,row) = inv * (x,y)
        self.polys_px: list[tuple[np.ndarray, np.ndarray]] = []
        boxes = []
        for geom in gdf.geometry:
            if geom is None or geom.is_empty:
                continue
            parts = geom.geoms if geom.geom_type.startswith("Multi") else [geom]
            for p in parts:
                ring = getattr(p, "exterior", None)
                if ring is None:
                    continue
                xs, ys = np.asarray(ring.coords.xy[0]), np.asarray(ring.coords.xy[1])
                cols, rows = inv * (xs, ys)  # affine vetorizado
                cols = np.asarray(cols); rows = np.asarray(rows)
                self.polys_px.append((cols, rows))
                boxes.append(box(cols.min(), rows.min(), cols.max(), rows.max()))
        self.tree = STRtree(boxes) if boxes else None
        self.n = len(self.polys_px)

    def query_outlines(self, x0, y0, x1, y1):
        if self.tree is None:
            return np.array([]), np.array([])
        idx = self.tree.query(box(x0, y0, x1, y1))
        if len(idx) == 0:
            return np.array([]), np.array([])
        xs, ys = [], []
        nan = np.array([np.nan])
        for i in idx:
            c, r = self.polys_px[i]
            xs.append(c); xs.append(nan)
            ys.append(r); ys.append(nan)
        return np.concatenate(xs), np.concatenate(ys)

    def query_polys(self, x0, y0, x1, y1):
        """Lista de (cols, rows) dos polígonos que tocam o viewport — p/ preencher
        o interior (QPainterPath). Só é chamado quando a camada tem cor de fill."""
        if self.tree is None:
            return []
        idx = self.tree.query(box(x0, y0, x1, y1))
        return [self.polys_px[i] for i in idx]
