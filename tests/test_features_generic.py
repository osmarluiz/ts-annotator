"""Extratores genéricos: servem a qualquer curva, e o padrão não muda de comportamento.

O caso que motiva o módulo está em test_sar_kills_phenology_axes: uma série de
retroespalhamento em dB passa pelo extrator fenológico sem erro nenhum e sai com
dois eixos constantes, porque os limiares são absolutos e a curva é negativa.
"""
import numpy as np
import pytest

from ts_annotator.core.features import (
    CurveFeatures,
    FeatureExtractor,
    PhenologyFeatures,
    ShapeFeatures,
    make_extractor,
)

NDVI_1C = [0.2, 0.3, 0.5, 0.7, 0.8, 0.7, 0.5, 0.3, 0.2, 0.2, 0.2, 0.2]
NDVI_2C = [0.2, 0.4, 0.7, 0.8, 0.5, 0.2, 0.4, 0.7, 0.6, 0.3, 0.2, 0.2]
# retroespalhamento Sentinel-1 VV, dB, tudo negativo
SAR_1C = [-12.0, -11.2, -9.8, -8.1, -7.4, -8.0, -9.5, -11.0, -12.2, -12.4, -12.1, -12.3]
SAR_2C = [-12.0, -9.5, -7.8, -9.9, -12.1, -9.2, -7.5, -9.8, -12.0, -12.3, -12.2, -12.1]


def test_default_is_phenology_and_alias_holds():
    """Projeto sem `descriptors:` recebe o extrator de sempre."""
    assert isinstance(make_extractor(), PhenologyFeatures)
    assert isinstance(make_extractor("phenology"), PhenologyFeatures)
    assert FeatureExtractor is PhenologyFeatures


def test_unknown_name_names_the_options():
    with pytest.raises(ValueError, match="shape"):
        make_extractor("naoexiste")


def test_sar_kills_phenology_axes():
    """A razão de o módulo existir: em dB, n_cycles e n_green viram constantes."""
    fx = PhenologyFeatures()
    a, b = fx.extract_dict(SAR_1C), fx.extract_dict(SAR_2C)
    assert a["n_cycles"] == b["n_cycles"] == 1     # nenhum pico passa de 0.40
    assert a["n_green"] == b["n_green"] == 0       # nada acima do limiar de verdor

    sh = ShapeFeatures()
    fa, fb = sh.extract_dict(SAR_1C), sh.extract_dict(SAR_2C)
    assert fa["n_extrema"] == 1 and fb["n_extrema"] == 2


@pytest.mark.parametrize("kind", ["shape", "curve"])
@pytest.mark.parametrize("curve", [NDVI_1C, SAR_1C])
def test_generic_extractors_are_finite_and_fixed_length(kind, curve):
    fx = make_extractor(kind)
    f = fx.extract(np.array(curve, float))
    assert f.ndim == 1 and len(f) > 0
    assert np.isfinite(f).all()
    assert len(f) == len(fx.extract(np.array(curve, float)[::-1]))


@pytest.mark.parametrize("kind", ["phenology", "shape", "curve"])
def test_accepts_one_band_and_four_band(kind):
    fx = make_extractor(kind)
    cube = np.zeros((12, 4))
    cube[:, 2] = 0.2
    cube[:, 3] = np.linspace(0.3, 0.8, 12)
    assert np.isfinite(fx.extract(cube)).all()
    assert np.isfinite(fx.extract(np.array(NDVI_1C, float))).all()


def test_shape_scales_with_channels():
    sh = ShapeFeatures()
    assert len(sh.extract(np.array(NDVI_1C, float))) == len(sh.BASE)
    assert len(sh.extract(np.zeros((12, 4)))) == 4 * len(sh.BASE)


def test_curve_is_the_curve():
    cv = CurveFeatures()
    assert np.allclose(cv.extract(np.array(NDVI_1C, float)), NDVI_1C)
    assert len(cv.extract(np.zeros((12, 4)))) == 48


def test_shape_is_scale_free():
    """Deslocar e escalar a curva não muda a forma que os descritores medem."""
    sh = ShapeFeatures()
    a = sh.extract_dict(np.array(NDVI_2C, float))
    b = sh.extract_dict(np.array(NDVI_2C, float) * 10.0 - 30.0)
    for k in ("argmax", "argmin", "n_extrema", "top_prominence", "acf1"):
        assert abs(a[k] - b[k]) < 1e-6, k


def test_nan_never_escapes():
    for kind in ("phenology", "shape", "curve"):
        f = make_extractor(kind).extract(np.array([np.nan] * 12, float))
        assert np.isfinite(f).all(), kind


def test_headline_only_where_it_means_something():
    ph = PhenologyFeatures()
    assert ph.headline(ph.extract(np.array(NDVI_2C, float))) == 2
    for kind in ("shape", "curve"):
        fx = make_extractor(kind)
        assert fx.headline(fx.extract(np.array(NDVI_2C, float))) is None
