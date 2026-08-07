"""Testes do trainer — build_X (sem GPU) + cleanlab flagando erros."""
import numpy as np
import pytest

pytest.importorskip("torch")            # trainer importa torch no módulo
pytest.importorskip("cleanlab")         # label_scores importa cleanlab lazy
from ts_annotator.core.trainer import build_X, fit, label_scores, spatial_cv  # noqa: E402


def test_build_X_adds_ndvi_channel():
    cur4 = np.random.rand(7, 4, 12).astype("float32")
    X = build_X(cur4)
    assert X.shape == (7, 5, 12)  # 4 bandas + NDVI


def test_build_X_without_red_nir_passes_channels_through():
    """Projeto SAR: bandas declaradas sem red/NIR -> canais entram como estão."""
    sar = np.random.rand(5, 2, 12).astype("float32")
    X = build_X(sar, bands=["VV", "VH"])
    assert X.shape == (5, 2, 12)
    assert np.allclose(X, sar)


def test_build_X_derives_ndvi_from_declared_band_names():
    """red/NIR achados pelo NOME das bandas declaradas, não pela posição 2/3."""
    cur = np.random.rand(3, 4, 12).astype("float32")
    X = build_X(cur, bands=["NIR", "Red", "G", "B"])
    assert X.shape == (3, 5, 12)
    nir, red = cur[:, 0], cur[:, 1]
    esperado = np.clip((nir - red) / (nir + red + 1e-6), -1, 1)
    assert np.allclose(X[:, 4], esperado, atol=1e-6)


def test_fit_progress_reports_epoch_and_loss():
    X = build_X(np.random.rand(40, 4, 12).astype("float32"))
    y = np.array([0, 1] * 20)
    seen = []
    fit(X, y, 2, epochs=5, progress=lambda e, E, loss: seen.append((e, E, loss)))
    assert len(seen) == 5 and seen[-1][0] == 5 and seen[-1][1] == 5
    assert all(isinstance(s[2], float) for s in seen)   # loss por época


def test_spatial_cv_callbacks_and_confusion():
    X = build_X(np.random.rand(60, 4, 12).astype("float32"))
    y = np.array([0, 1, 2] * 20)
    groups = np.repeat(np.arange(12), 5)   # 12 blocos espaciais
    folds_done, epochs_seen = [], []
    res = spatial_cv(X, y, groups, 3, epochs=3, k=3,
                     progress=lambda f, k, bacc: folds_done.append((f, k, bacc)),
                     epoch_cb=lambda f, k, e, E, loss: epochs_seen.append((f, e)))
    assert len(folds_done) == 3 and all(0 <= b <= 1 for _f, _k, b in folds_done)
    assert epochs_seen and epochs_seen[0] == (1, 1)      # fold 1, época 1
    assert np.asarray(res["confusion"]).shape == (3, 3)  # matriz por classe


def test_label_scores_flags_injected_errors():
    y = np.array([0, 1, 2] * 30)
    probs = np.zeros((len(y), 3), "float32")
    for i, c in enumerate(y):
        probs[i, c] = 0.8
        probs[i] += 0.1
    probs /= probs.sum(1, keepdims=True)
    y_bad = y.copy()
    y_bad[::15] = (y_bad[::15] + 1) % 3  # injeta erros
    scores, issues = label_scores(y_bad, probs)
    assert len(scores) == len(y)
    assert len(issues) > 0  # cleanlab detecta os erros
