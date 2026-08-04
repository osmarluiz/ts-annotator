"""Testes das métricas de seleção (cada uma)."""
from ts_annotator.core import metrics as M


def test_confidence_and_margin():
    peaked = {"a": 0.9, "b": 0.05, "c": 0.05}
    flat = {"a": 0.34, "b": 0.33, "c": 0.33}
    assert M.confidence(peaked) > M.confidence(flat)
    assert M.margin(peaked) > M.margin(flat)


def test_entropy():
    peaked = {"a": 0.9, "b": 0.05, "c": 0.05}
    flat = {"a": 0.34, "b": 0.33, "c": 0.33}
    assert M.entropy(peaked) < M.entropy(flat)


def test_similarity_and_novelty():
    s = {"a": 0.8, "b": 0.2}
    assert abs(M.similarity(s) - 0.8) < 1e-6
    assert abs(M.novelty(s) - 0.2) < 1e-6


def test_disagreement():
    sim = {"a": 0.8, "b": 0.1, "c": 0.1}
    agree = {"a": 0.85, "b": 0.10, "c": 0.05}
    disagree = {"a": 0.05, "b": 0.10, "c": 0.85}
    assert M.disagreement(sim, sim) < 1e-6
    assert M.disagreement(sim, agree) < 0.2
    assert M.disagreement(sim, disagree) > 0.6


def test_compute_dispatch():
    sim = {"a": 0.7, "b": 0.3}
    pred = {"a": 0.2, "b": 0.8}
    assert M.compute(sim, pred, "disagreement") > 0
    assert abs(M.compute(sim, pred, "confidence") - 0.8) < 1e-6
    assert M.compute(sim, pred, "novelty") > 0
