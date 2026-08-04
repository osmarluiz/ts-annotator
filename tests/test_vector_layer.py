"""Tests for core.vector_layer.VectorLayer — geo->pixel outlines + spatial query."""
import numpy as np
import pytest

gpd = pytest.importorskip("geopandas")
from affine import Affine                       # noqa: E402  (rasterio dep)
from shapely.geometry import Polygon            # noqa: E402

from ts_annotator.core.vector_layer import VectorLayer   # noqa: E402


def _write(tmp_path, polys):
    gdf = gpd.GeoDataFrame(geometry=polys, crs="EPSG:32723")
    p = tmp_path / "v.gpkg"
    gdf.to_file(p, driver="GPKG")
    return str(p)


def test_vector_layer_outlines_and_query(tmp_path):
    polys = [Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
             Polygon([(100, 100), (110, 100), (110, 110), (100, 110)])]
    path = _write(tmp_path, polys)
    tr = Affine(1, 0, 0, 0, -1, 200)             # x=col, y=200-row (pixel<->geo)
    vl = VectorLayer(path, tr, crs=None)         # crs=None: usa as coords como estão
    assert vl.n == 2

    xs, ys = vl.query_outlines(-50, -50, 300, 300)   # cobre os dois polígonos
    assert len(xs) > 0 and len(xs) == len(ys)
    assert np.isnan(xs).any()                    # separadores NaN p/ o PlotDataItem

    xs2, ys2 = vl.query_outlines(1000, 1000, 1100, 1100)   # vazio
    assert len(xs2) == 0 and len(ys2) == 0


def test_vector_layer_empty(tmp_path):
    path = _write(tmp_path, [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])])
    vl = VectorLayer(path, Affine(1, 0, 0, 0, -1, 10), crs=None)
    assert vl.n == 1
    # tree existe; consulta fora não quebra
    xs, ys = vl.query_outlines(500, 500, 600, 600)
    assert len(xs) == 0
