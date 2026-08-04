"""Testes headless do núcleo (sem GUI): RasterSource (leitura/cache) + colorizers."""
import numpy as np
import pytest
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin

from ts_annotator.core.raster_source import RasterSource


@pytest.fixture
def tiny_cog(tmp_path):
    p = tmp_path / "tiny.tif"
    rng = np.random.default_rng(0)
    data = (rng.random((3, 512, 512)) * 255).astype("uint8")
    with rasterio.open(
        p, "w", driver="GTiff", height=512, width=512, count=3, dtype="uint8",
        tiled=True, blockxsize=256, blockysize=256,
        transform=from_origin(0, 512, 1, 1), crs="EPSG:32723",
    ) as d:
        d.write(data)
        d.build_overviews([2, 4], Resampling.nearest)
    return str(p)


def test_metadata(tiny_cog):
    s = RasterSource(tiny_cog)
    assert (s.width, s.height) == (512, 512)
    assert s.count == 3
    assert s.overviews == [2, 4]
    s.close()


def test_read_view_shape(tiny_cog):
    s = RasterSource(tiny_cog)
    arr, win = s.read_view(0, 0, 512, 512, 128, 128)
    assert arr.shape == (3, 128, 128)
    assert win == (0, 0, 512, 512)
    s.close()


def test_read_view_clip_bounds(tiny_cog):
    s = RasterSource(tiny_cog)
    arr, win = s.read_view(-100, -100, 200, 200, 100, 100)
    assert win[0] == 0 and win[1] == 0  # clipado à imagem
    s.close()


def test_read_view_outside(tiny_cog):
    s = RasterSource(tiny_cog)
    arr, win = s.read_view(10000, 10000, 50, 50, 50, 50)
    assert arr is None and win is None
    s.close()


def test_cache(tiny_cog):
    s = RasterSource(tiny_cog)
    s.read_view(0, 0, 512, 512, 128, 128)
    s.read_view(0, 0, 512, 512, 128, 128)  # hit
    s.read_view(0, 0, 256, 256, 64, 64)    # miss
    assert s.cache_hits == 1
    assert s.cache_miss == 2
    s.close()


def test_read_pixel(tiny_cog):
    s = RasterSource(tiny_cog)
    px = s.read_pixel(10, 10)
    assert px.shape == (3,)
    assert s.read_pixel(9999, 9999) is None
    s.close()


def test_class_colorizer():
    from ts_annotator.ui.map_view import make_class_colorizer

    cz = make_class_colorizer(["#ff0000", "#00ff00"], nodata=255)
    arr = np.array([[[0, 1, 255]]])  # (bands=1, h=1, w=3)
    rgba = cz(arr)
    assert rgba.shape == (1, 3, 4)
    assert tuple(rgba[0, 0, :3]) == (255, 0, 0)  # classe 0 = vermelho
    assert rgba[0, 2, 3] == 0  # nodata transparente
