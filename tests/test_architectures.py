"""Catálogo de arquiteturas + treino/checkpoint com a arquitetura declarada.

O caso que motiva: o combo da aba de treino promete "+ arquiteturas no futuro",
e o futuro é o tsai como catálogo opcional. Sem ele instalado sobra `it`, e o
checkpoint grava a arquitetura para load_model reconstruir a rede certa.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ts_annotator.core import architectures, trainer  # noqa: E402


def test_available_lists_builtin_first():
    assert architectures.available()[0] == "it"


def test_build_default_is_the_builtin_it():
    net = architectures.build(None, 5, 3)
    assert isinstance(net, trainer.IT)
    net2 = architectures.build("it", 2, 4)
    assert net2.bk[0].b.in_channels == 2


def test_build_unknown_names_the_options():
    with pytest.raises(ValueError, match="it"):
        architectures.build("nao_existe", 5, 3)


def test_fit_with_arch_and_checkpoint_roundtrip(tmp_path):
    """fit aceita arch, o checkpoint grava a arquitetura e load_model reconstrói."""
    X = np.random.rand(40, 3, 12).astype("float32")
    y = np.array([0, 1] * 20)
    net, mu, sd = trainer.fit(X, y, 2, epochs=2, arch="it")
    p = str(tmp_path / "model.pt")
    trainer.save_model(net, mu, sd, ["a", "b"], p, arch="it")
    net2, mu2, sd2, labs = trainer.load_model(p)
    assert labs == ["a", "b"]
    pr = trainer.predict_proba(net2, mu2, sd2, X[:5])
    assert pr.shape == (5, 2)


def test_estimators_are_in_the_catalogue():
    """Curva achatada com interface sklearn: catálogo além das redes torch."""
    names = architectures.available()
    assert "random_forest" in names and "hist_gradient_boosting" in names
    assert ("xgboost" in names) == architectures.has_xgboost()
    assert architectures.kind("random_forest") == "estimator"
    assert architectures.kind("it") == "torch"


def test_fit_predict_and_checkpoint_roundtrip_with_estimator(tmp_path):
    """fit/predict_proba/save/load funcionam com um estimador sklearn."""
    X = np.random.rand(40, 3, 12).astype("float32")
    y = np.array([0] * 20 + [1] * 20)
    X[20:] += 0.8
    net, mu, sd = trainer.fit(X, y, 2, arch="random_forest")
    pr = trainer.predict_proba(net, mu, sd, X)
    assert pr.shape == (40, 2)
    assert (pr.argmax(1) == y).mean() > 0.9
    p = str(tmp_path / "model.pt")
    trainer.save_model(net, mu, sd, ["a", "b"], p)
    net2, mu2, sd2, labs = trainer.load_model(p)
    assert labs == ["a", "b"]
    assert trainer.predict_proba(net2, mu2, sd2, X[:7]).shape == (7, 2)


def test_spatial_cv_with_estimator_produces_metrics():
    """A validação cruzada espacial e o cleanlab valem para qualquer família."""
    X = np.random.rand(60, 2, 12).astype("float32")
    y = np.array([0, 1, 2] * 20)
    X[y == 1] += 0.5
    X[y == 2] -= 0.5
    groups = np.repeat(np.arange(12), 5)
    res = trainer.spatial_cv(X, y, groups, 3, k=3, arch="random_forest")
    assert 0.0 <= res["bacc"] <= 1.0
    assert np.asarray(res["confusion"]).shape == (3, 3)


def test_xgboost_absent_is_a_named_error():
    if architectures.has_xgboost():
        pytest.skip("xgboost instalado neste ambiente")
    with pytest.raises(ValueError, match="xgboost"):
        architectures.build_estimator("xgboost")


def test_old_checkpoint_without_arch_loads_as_it(tmp_path):
    """Checkpoint antigo (sem a chave arch) segue carregando como IT."""
    net = trainer.IT(5, 2)
    mu = np.zeros((1, 5, 1), "float32")
    sd = np.ones((1, 5, 1), "float32")
    p = str(tmp_path / "model.pt")
    torch.save({"state": net.state_dict(), "mu": mu, "sd": sd,
                "labs": ["x", "y"], "ic": 5}, p)
    net2, _mu, _sd, labs = trainer.load_model(p)
    assert isinstance(net2, trainer.IT) and labs == ["x", "y"]
