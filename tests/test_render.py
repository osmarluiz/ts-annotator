"""Tests for render.py — the raster -> RGBA colorizers (pure numpy, GUI-free)."""
import numpy as np

from ts_annotator.render import (
    ClassColorizer,
    make_class_colorizer,
    make_rgb_colorizer,
    make_scalar_colorizer,
)


def test_class_colorizer_lut_and_nodata():
    c = make_class_colorizer(["#ff0000", "#00ff00", "#0000ff"], nodata=255)
    arr = np.array([[[0, 1, 2, 255]]])          # (bands=1, h=1, w=4)
    rgba = c(arr)[0]
    assert list(rgba[0]) == [255, 0, 0, 255]
    assert list(rgba[1]) == [0, 255, 0, 255]
    assert list(rgba[2]) == [0, 0, 255, 255]
    assert rgba[3][3] == 0                       # nodata -> transparent


def test_class_colorizer_hidden_and_recolor():
    c = make_class_colorizer(["#ff0000", "#00ff00"], nodata=255)
    assert isinstance(c, ClassColorizer)
    arr = np.array([[[0, 1]]])
    c.set_hidden({1})                            # esconde a classe 1
    r = c(arr)[0]
    assert r[0][3] == 255 and r[1][3] == 0
    c.set_hidden(set())                          # reexibe
    c.set_color(1, "#ffff00")                    # edita a cor da classe 1
    r = c(arr)[0]
    assert list(r[1]) == [255, 255, 0, 255]


def test_rgb_colorizer_nodata():
    c = make_rgb_colorizer(nodata=0)
    # (3 bandas, h=1, w=2): pixel0 = (10,20,30); pixel1 = (0,0,0) -> nodata
    arr = np.array([[[10, 0]], [[20, 0]], [[30, 0]]], dtype=np.uint8)
    rgba = c(arr)
    assert rgba.shape == (1, 2, 4)
    assert list(rgba[0, 0, :3]) == [10, 20, 30] and rgba[0, 0, 3] == 255
    assert rgba[0, 1, 3] == 0                    # soma 0 -> transparente


def test_scalar_colorizer_range_and_nan():
    c = make_scalar_colorizer(0.0, 1.0, cmap="viridis", nodata=-1.0)
    arr = np.array([[[0.0, 0.5, 1.0, -1.0]]], dtype=np.float32)
    rgba = c(arr)[0]
    assert rgba.shape == (4, 4)
    assert rgba[0][3] == 255 and rgba[2][3] == 255   # 0.0 e 1.0 opacos
    assert rgba[3][3] == 0                            # nodata transparente
    nanarr = np.array([[[np.nan, 0.5]]], dtype=np.float32)
    assert c(nanarr)[0][0][3] == 0                    # NaN transparente
