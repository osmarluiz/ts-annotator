"""Testes do SimilarityEngine — classe óbvia pontua alto; outlier = novidade."""
import numpy as np

from ts_annotator.core.features import FeatureExtractor
from ts_annotator.core.similarity import SimilarityEngine

fx = FeatureExtractor()


def _c1():
    return np.array([0.2, 0.3, 0.5, 0.7, 0.8, 0.7, 0.5, 0.3, 0.2, 0.2, 0.2, 0.2])


def _c2():
    return np.array([0.2, 0.4, 0.7, 0.8, 0.5, 0.2, 0.4, 0.7, 0.6, 0.3, 0.2, 0.2])


def _c3():
    return np.array([0.5, 0.7, 0.6, 0.4, 0.7, 0.6, 0.4, 0.7, 0.6, 0.5, 0.6, 0.5])


def _items():
    rng = np.random.default_rng(0)
    items = []
    for _ in range(6):
        items.append((fx.extract(_c1() + rng.normal(0, 0.02, 12)), "1c"))
        items.append((fx.extract(_c2() + rng.normal(0, 0.02, 12)), "2c"))
        items.append((fx.extract(_c3() + rng.normal(0, 0.02, 12)), "3c"))
    return items


def test_obvious_class_scores_highest():
    eng = SimilarityEngine(k=3)
    eng.load(_items())
    s = eng.score(fx.extract(_c2()))
    assert max(s, key=s.get) == "2c"


def test_novelty_on_outlier():
    eng = SimilarityEngine(k=3, novelty_thr=0.5)
    eng.load(_items())
    weird = fx.extract(np.full(12, 0.05))  # quase sem vegetação
    assert eng.is_novel(eng.score(weird))


def test_nearest_returns_exemplar_id():
    items = [(fx.extract(_c2()), "2c", 42)] + _items()
    eng = SimilarityEngine()
    eng.load(items)
    n = eng.nearest(fx.extract(_c2()))
    assert "2c" in n and isinstance(n["2c"], tuple)
