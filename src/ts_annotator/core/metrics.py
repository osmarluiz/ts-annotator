"""Métricas de seleção (active learning) a partir de sim/pred por classe.

sim  = dict classe->similaridade (kNN)
pred = dict classe->probabilidade (modelo)
Todas retornam um escalar; a UI escolhe a ordem (maior->menor ou menor->maior).
"""
from __future__ import annotations

import numpy as np


def _dist(d, classes):
    v = np.array([d.get(c, 0.0) for c in classes], float)
    s = v.sum()
    return v / s if s > 0 else v


def confidence(pred):
    """Confiança = maior probabilidade do modelo (baixa = incerto)."""
    return float(max(pred.values())) if pred else 0.0


def margin(pred):
    """Margem top1-top2 da predição (baixa = ambíguo)."""
    v = sorted(pred.values(), reverse=True)
    return float(v[0] - v[1]) if len(v) > 1 else (float(v[0]) if v else 0.0)


def entropy(pred):
    """Entropia da predição (alta = incerto)."""
    if not pred:
        return 0.0
    p = _dist(pred, sorted(pred))
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def similarity(sim):
    """Maior similaridade kNN (baixa = novidade/outlier)."""
    return float(max(sim.values())) if sim else 0.0


def novelty(sim):
    """1 - similaridade máxima (alta = não parece com nada)."""
    return 1.0 - similarity(sim)


def disagreement(sim, pred):
    """Discordância sim x pred = TV distance entre as distribuições (0=concordam, 1=discordam)."""
    if not sim or not pred:
        return 0.0
    classes = sorted(set(sim) | set(pred))
    s = _dist(sim, classes)
    p = _dist(pred, classes)
    return float(0.5 * np.abs(s - p).sum())


def compute(sim, pred, name):
    """Calcula a métrica `name` a partir de sim/pred (pred pode ser None)."""
    pred = pred or {}
    sim = sim or {}
    if name == "disagreement":
        return disagreement(sim, pred)
    if name == "confidence":
        return confidence(pred)
    if name == "margin":
        return margin(pred)
    if name == "entropy":
        return entropy(pred)
    if name == "similarity":
        return similarity(sim)
    if name == "novelty":
        return novelty(sim)
    return 0.0
