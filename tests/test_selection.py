"""Testes da SelectionEngine — propor por classe / incerteza, com TS fake."""
import numpy as np

from ts_annotator.core.features import FeatureExtractor
from ts_annotator.core.selection import SelectionEngine
from ts_annotator.core.similarity import SimilarityEngine

C1 = np.array([0.2, 0.3, 0.5, 0.7, 0.8, 0.7, 0.5, 0.3, 0.2, 0.2, 0.2, 0.2])
C2 = np.array([0.2, 0.4, 0.7, 0.8, 0.5, 0.2, 0.4, 0.7, 0.6, 0.3, 0.2, 0.2])
fx = FeatureExtractor()


def _curve(nd):
    nir = 0.4
    red = nir * (1 - nd) / (1 + nd)
    c = np.zeros((12, 4))
    c[:, 2] = red
    c[:, 3] = nir
    return c


class FakeTS:
    def read_curve(self, row, col):
        return _curve(C2 if row % 2 == 0 else C1)  # linha par = 2c, ímpar = 1c


def _sim():
    items = []
    for _ in range(5):
        items.append((fx.extract(_curve(C1)), "1c"))
        items.append((fx.extract(_curve(C2)), "2c"))
    s = SimilarityEngine(k=3)
    s.load(items)
    return s


def test_propose_class_returns_target():
    eng = SelectionEngine(FakeTS(), fx, _sim())
    cand = eng.propose(0, 0, 100, 100, metric="class", target="2c", n=40)
    assert cand is not None
    assert max(cand["scores"], key=cand["scores"].get) == "2c"
    assert cand["row"] % 2 == 0


def test_propose_uncertain_runs():
    eng = SelectionEngine(FakeTS(), fx, _sim())
    cand = eng.propose(0, 0, 100, 100, metric="margin", order="asc", n=40)
    assert cand is not None and "scores" in cand


def test_exclude_skips_points():
    eng = SelectionEngine(FakeTS(), fx, _sim())
    c1 = eng.propose(0, 0, 100, 100, metric="class", target="2c", n=40, seed=1)
    c2 = eng.propose(0, 0, 100, 100, metric="class", target="2c", n=40, seed=1,
                     exclude={(c1["row"], c1["col"])})
    assert (c2["row"], c2["col"]) != (c1["row"], c1["col"])
