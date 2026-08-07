"""Testes do TimeSeriesSource — lê curva no pixel, nodata->NaN, fora->NaN."""
import numpy as np
import pytest
import rasterio

from ts_annotator.core.timeseries import TimeSeriesSource, parse_bands


def test_parse_bands_names_give_indices_in_order():
    """Nomes declarados: índices de leitura são 1..n na ordem do arquivo."""
    idx, names = parse_bands(["VV", "VH"])
    assert idx == (1, 2) and names == ["VV", "VH"]


def test_parse_bands_legacy_ints_keep_the_optical_contract():
    """Quatro índices (o padrão de sempre) continuam valendo B,G,R,NIR."""
    idx, names = parse_bands([1, 2, 3, 4])
    assert idx == (1, 2, 3, 4) and names == ["B", "G", "R", "NIR"]


def test_parse_bands_ints_not_four_have_no_names():
    idx, names = parse_bands([2, 3])
    assert idx == (2, 3) and names is None


def test_parse_bands_mixed_is_an_error():
    with pytest.raises(ValueError, match="bands"):
        parse_bands([1, "R"])


def _make_month(path, red, nir, w=10, h=10):
    data = np.zeros((4, h, w), dtype="uint16")
    data[2] = red
    data[3] = nir
    with rasterio.open(
        path, "w", driver="GTiff", height=h, width=w, count=4, dtype="uint16"
    ) as d:
        d.write(data)


def test_read_curve_reflectance(tmp_path):
    paths = []
    for m in range(3):
        p = tmp_path / f"m{m}.tif"
        _make_month(p, red=2000 + m * 100, nir=6000)
        paths.append(str(p))
    ts = TimeSeriesSource(paths, bands=(1, 2, 3, 4), nodata=65535, scale=10000)
    curve = ts.read_curve(5, 5)
    assert curve.shape == (3, 4)
    assert abs(curve[0, 2] - 0.2) < 1e-3   # red mês0 = 2000/10000
    assert abs(curve[0, 3] - 0.6) < 1e-3   # nir = 6000/10000
    assert abs(curve[2, 2] - 0.22) < 1e-3  # red mês2 = 2200/10000
    ts.close()


def test_nodata_becomes_nan(tmp_path):
    p = tmp_path / "m.tif"
    _make_month(p, red=65535, nir=6000)
    ts = TimeSeriesSource([str(p)], nodata=65535, scale=10000)
    curve = ts.read_curve(5, 5)
    assert np.isnan(curve[0, 2])         # red nodata -> NaN
    assert abs(curve[0, 3] - 0.6) < 1e-3
    ts.close()


def test_out_of_bounds_is_nan(tmp_path):
    p = tmp_path / "m.tif"
    _make_month(p, red=2000, nir=6000)
    ts = TimeSeriesSource([str(p)], nodata=65535, scale=10000)
    assert np.isnan(ts.read_curve(999, 999)).all()
    ts.close()


def _make_row_gradient(path, h=12, w=10):
    """red = valor que varia por LINHA — permite detectar um offset de linha na leitura."""
    data = np.zeros((4, h, w), dtype="uint16")
    for r in range(h):
        data[2, r, :] = 1000 + r * 10
    data[3] = 6000
    with rasterio.open(path, "w", driver="GTiff", height=h, width=w, count=4, dtype="uint16") as d:
        d.write(data)


def test_row_offsets_correct_misregistration(tmp_path):
    """row_offsets desloca a leitura por mês (ex.: correção do shift de mai-set)."""
    p0, p1 = tmp_path / "a.tif", tmp_path / "b.tif"
    _make_row_gradient(p0)
    _make_row_gradient(p1)
    # mês1 tem +3 de offset: ler (row=2) deve pegar a linha 5 dele
    ts = TimeSeriesSource([str(p0), str(p1)], row_offsets=[0, 3], nodata=65535, scale=10000)
    c = ts.read_curve(2, 5)
    assert abs(c[0, 2] - 0.102) < 1e-6      # mês0 offset 0 → linha 2 → 1020/10000
    assert abs(c[1, 2] - 0.105) < 1e-6      # mês1 offset 3 → linha 5 → 1050/10000
    # read_block honra o mesmo offset
    blk = ts.read_block(2, 5, 1, 1, step=1)
    assert abs(blk[0, 0, 1, 2] - 0.105) < 1e-6
    ts.close()
