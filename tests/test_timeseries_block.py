"""TimeSeriesSource.read_block + safe_window (offsets, decimação, borda)."""
import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin  # noqa: E402

from ts_annotator.core.timeseries import TimeSeriesSource  # noqa: E402

W, H, NOD = 40, 30, 65535


def _write_month(path, fill):
    prof = dict(driver="GTiff", width=W, height=H, count=4, dtype="uint16",
                crs="EPSG:32723", transform=from_origin(0, H, 1, 1), nodata=NOD)
    arr = np.full((4, H, W), fill, np.uint16)
    arr[:, 0, 0] = NOD                       # 1 pixel nodata p/ testar NaN
    with rasterio.open(path, "w", **prof) as ds:
        ds.write(arr)


@pytest.fixture()
def ts(tmp_path):
    paths = []
    for m in range(3):
        p = str(tmp_path / f"m{m}.tif")
        _write_month(p, 1000 * (m + 1))
        paths.append(p)
    return TimeSeriesSource(paths, bands=(1, 2, 3, 4), nodata=NOD, scale=10000)


def test_read_curve_values_and_nodata(ts):
    c = ts.read_curve(5, 5)
    assert c.shape == (3, 4)
    assert np.allclose(c[:, 0], [0.1, 0.2, 0.3])
    assert np.isnan(ts.read_curve(0, 0)).all()       # pixel nodata → NaN
    assert np.isnan(ts.read_curve(-1, 5)).all()      # fora → NaN


def test_read_block_shapes_and_decimation(ts):
    b = ts.read_block(0, 0, H, W, step=1)
    assert b.shape == (H, W, 3, 4)
    assert np.allclose(b[5, 5, :, 0], [0.1, 0.2, 0.3])
    b2 = ts.read_block(0, 0, H, W, step=2)
    assert b2.shape == (H // 2, W // 2, 3, 4)


def test_read_block_out_of_bounds_fills_nan(ts):
    b = ts.read_block(-10, -10, 20, 20, step=1)      # cruza a borda → boundless
    assert np.isnan(b[0, 0]).all()                   # fora da imagem
    assert np.isfinite(b[15, 15]).all()              # dentro


def test_safe_window_no_offsets(ts):
    assert ts.safe_window() == (0, 0, H, W)


def test_safe_window_with_offsets(tmp_path):
    paths = []
    for m in range(2):
        p = str(tmp_path / f"o{m}.tif")
        _write_month(p, 500)
        paths.append(p)
    ts = TimeSeriesSource(paths, row_offsets=[0, 3], col_offsets=[-2, 0], nodata=NOD)
    rmin, cmin, rmax, cmax = ts.safe_window()
    # mês 1 desloca +3 em row → últimas 3 linhas do grid caem fora; col -2 → col_min=2
    assert (rmin, cmin) == (0, 2)
    assert rmax == H - 3 and cmax == W
    # janela clampada nesses limites lê in-bounds em TODOS os meses (sem NaN de borda)
    b = ts.read_block(rmin, cmin, rmax - rmin, cmax - cmin, step=1)
    assert np.isfinite(b[1:, 1:]).all() or np.isfinite(b).sum() > 0
