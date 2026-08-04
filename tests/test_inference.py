"""Tests for core.inference.classify_block — the shared read->mask->predict kernel."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")

from ts_annotator.core import trainer                      # noqa: E402
from ts_annotator.core.inference import classify_block     # noqa: E402


class _FakeTS:
    """Fonte de série temporal falsa: read_block devolve um bloco fixo."""
    def __init__(self, block):
        self.block = block

    def read_block(self, row0, col0, h, w, step):
        return self.block


def _tiny_model(rng):
    """Treina um IT minúsculo, 2 classes separáveis (curvas N,4,12)."""
    n = 60
    cur = rng.random((n, 4, 12)).astype("float32")
    y = np.array([0] * 30 + [1] * 30)
    cur[y == 1] += 0.6                                      # sinal separável
    X = trainer.build_X(cur)
    net, mu, sd = trainer.fit(X, y, 2, lr=1e-2, epochs=25)
    return net, mu, sd


def test_classify_block_masks_incomplete_and_classifies():
    rng = np.random.default_rng(0)
    net, mu, sd = _tiny_model(rng)
    block = rng.random((3, 3, 12, 4)).astype("float32")    # (Hs,Ws,meses,bandas)
    block[0, 0, 5, 2] = np.nan                             # pixel (0,0): curva incompleta
    calls = {}
    ci, valid, (hs, ws) = classify_block(
        _FakeTS(block), net, mu, sd, 0, 0, 3, 3,
        after_read=lambda h, w: calls.setdefault("read", (h, w)),
        before_infer=lambda n: calls.setdefault("infer", n))
    assert (hs, ws) == (3, 3)
    assert valid.shape == (9,) and valid.sum() == 8        # 1 NaN excluído
    assert not valid[0]                                    # o (0,0) é o inválido
    assert ci.shape == (8,) and set(ci.tolist()) <= {0, 1}
    assert calls["read"] == (3, 3) and calls["infer"] == 8  # callbacks disparam


def test_classify_block_all_invalid_returns_empty():
    block = np.full((2, 2, 12, 4), np.nan, "float32")
    ci, valid, dims = classify_block(_FakeTS(block), None, None, None, 0, 0, 2, 2)
    assert ci.shape == (0,) and not valid.any() and dims == (2, 2)
