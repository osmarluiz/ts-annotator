"""Testes do FeatureExtractor — foco na contagem de ciclo por VALE (não altura)."""
import numpy as np

from ts_annotator.core.features import FeatureExtractor, to_ndvi

fx = FeatureExtractor()


def _n(nd):
    return fx.extract_dict(np.array(nd, float))["n_cycles"]


def test_to_ndvi_from_bgrn():
    c = np.zeros((12, 4))
    c[:, 2] = 0.2  # red
    c[:, 3] = 0.6  # nir
    nd = to_ndvi(c)
    assert nd.shape == (12,)
    assert abs(nd[0] - (0.6 - 0.2) / (0.6 + 0.2)) < 1e-3


def test_1c_clear():
    assert _n([0.2, 0.3, 0.5, 0.7, 0.8, 0.7, 0.5, 0.3, 0.2, 0.2, 0.2, 0.2]) == 1


def test_2c_clear():
    assert _n([0.2, 0.4, 0.7, 0.8, 0.5, 0.2, 0.4, 0.7, 0.6, 0.3, 0.2, 0.2]) == 2


def test_weak_safrinha_is_2c():
    # 2o pico 0.45 (< 0.50 de ALTURA) MAS com VALE claro -> deve contar como 2c
    assert _n([0.2, 0.4, 0.8, 0.7, 0.4, 0.25, 0.45, 0.42, 0.3, 0.2, 0.2, 0.2]) == 2


def test_persistent_canopy_is_1c():
    # dossel persistente (sem vale) -> 1c
    assert _n([0.70, 0.72, 0.75, 0.78, 0.76, 0.74, 0.70, 0.68, 0.70, 0.72, 0.70, 0.68]) == 1


def test_slow_decline_is_1c():
    assert _n([0.2, 0.5, 0.8, 0.75, 0.65, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25]) == 1


def test_dry_and_amplitude():
    f = fx.extract_dict([0.2, 0.3, 0.5, 0.7, 0.8, 0.7, 0.5, 0.3, 0.2, 0.2, 0.2, 0.2])
    assert 0.0 <= f["dry_ndvi"] <= 0.3
    assert f["amplitude"] > 0.5
