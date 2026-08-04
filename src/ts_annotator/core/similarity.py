"""SimilarityEngine — similaridade por classe (k-NN nas features, % absoluto, novidade).

k-NN (não centróide) nas FEATURES interpretáveis (padronizadas p/ distância
comparável — isso é nas features, não na curva). Retorna similaridade ABSOLUTA
por classe (0-1) — permite detectar NOVIDADE (tudo baixo = não parece com nada).
"""
from __future__ import annotations

import numpy as np


class SimilarityEngine:
    def __init__(self, k: int = 3, novelty_thr: float = 0.45):
        self.k = k
        self.novelty_thr = novelty_thr
        self._mean = None
        self._std = None
        self._classes: dict = {}   # classe -> matriz (N,F) padronizada
        self._ids: dict = {}       # classe -> lista de ids (exemplares)

    def load(self, items) -> None:
        """items: lista de (features[F], classe[, id])."""
        self._classes, self._ids = {}, {}
        if not items:
            self._mean = self._std = None   # não deixar z-space obsoleto p/ diversidade
            return
        feats = np.array([np.asarray(it[0], float) for it in items])
        self._mean = feats.mean(0)
        self._std = feats.std(0) + 1e-6
        for it in items:
            z = (np.asarray(it[0], float) - self._mean) / self._std
            c = it[1]
            pid = it[2] if len(it) > 2 else None
            self._classes.setdefault(c, []).append(z)
            self._ids.setdefault(c, []).append(pid)
        self._classes = {c: np.array(v) for c, v in self._classes.items()}

    def _kdist(self, M, z):
        d = np.linalg.norm(M - z, axis=1)
        return float(np.sort(d)[: self.k].mean())

    def score(self, features, exclude=None) -> dict:
        """{classe: similaridade 0-1} (absoluta, média dos top-k).

        ``exclude``: id de um exemplar a IGNORAR (leave-one-out) — pontuar um ponto
        já rotulado SEM ele mesmo, senão o auto-match (distância 0) infla o número.
        """
        if not self._classes:
            return {}
        z = (np.asarray(features, float) - self._mean) / self._std
        out = {}
        for c, M in self._classes.items():
            if exclude is not None:
                ids = self._ids.get(c)
                if ids is not None and exclude in ids:
                    keep = [i for i, pid in enumerate(ids) if pid != exclude]
                    if not keep:
                        continue          # era o único exemplar da classe → sem base p/ comparar
                    M = M[keep]
            out[c] = 1.0 / (1.0 + self._kdist(M, z))
        return out

    def nearest(self, features) -> dict:
        """{classe: (id_exemplar, similaridade)} — o exemplar mais próximo de cada classe."""
        if not self._classes:
            return {}
        z = (np.asarray(features, float) - self._mean) / self._std
        out = {}
        for c, M in self._classes.items():
            d = np.linalg.norm(M - z, axis=1)
            i = int(np.argmin(d))
            out[c] = (self._ids[c][i], 1.0 / (1.0 + float(d[i])))
        return out

    def ranked(self, features):
        return sorted(self.score(features).items(), key=lambda kv: -kv[1])

    def is_novel(self, scores) -> bool:
        return (not scores) or (max(scores.values()) < self.novelty_thr)
