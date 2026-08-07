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
