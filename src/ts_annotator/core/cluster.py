"""ClusterEngine — user-driven hierarchical discovery of recurring curve patterns.

The opposite of active learning: instead of the rare/uncertain point, it finds
the DOMINANT curve shapes — what repeats. Samples valid pixels once (reusing the
SelectionEngine's decimated block sampler → overview fast path, safe-window
clamp, cell `inside` filter), then builds a DIVISIVE tree the user drives
top-down: split the area into k groups, pick an interesting one, split THAT into
k again, deeper only where structure is worth it. Each split is a local k-means
on that node's pixels; nesting is guaranteed by construction (a grandchild lives
inside its child). A cluster is represented by its MEDOID — the real sampled
pixel nearest the node centre (a curve you can jump to and label).

Domain-agnostic: clusters whatever the active FeatureExtractor produces.
"""
from __future__ import annotations

import numpy as np


class ClusterNode:
    __slots__ = ("idx", "medoid", "size", "frac", "children", "parent", "path")

    def __init__(self, idx, medoid, frac, path, parent=None):
        self.idx = idx                 # índices na amostra (np.array)
        self.medoid = medoid           # candidato REAL representativo {row,col,curve,features}
        self.size = int(len(idx))
        self.frac = float(frac)        # prevalência na ÁREA toda (0-1)
        self.children = None           # None = não dividido ainda
        self.parent = parent
        self.path = path               # "raiz", "raiz.1", "raiz.1.3", …


class ClusterEngine:
    def __init__(self, sel):
        self.sel = sel
        self._Z = None
        self._cands = None
        self.root = None

    def discover(self, x0, y0, x1, y1, k=6, step=10, seed=0, inside=None, cap=80000):
        """Amostra a área numa GRADE de passo `step` px (densidade explícita) e
        divide a RAIZ em `k` grupos. Retorna a raiz; a amostragem fica em
        self.info (n_valid, eff_step, capped…). Divisões seguintes: split()."""
        cands, self.info = self.sel.grid_sample(x0, y0, x1, y1, step, inside=inside, cap=cap)
        if not cands:
            self.root = None
            return None
        feats = np.array([c["features"] for c in cands], float)
        self._mu = feats.mean(0)
        self._sd = feats.std(0) + 1e-6
        self._Z = (feats - self._mu) / self._sd
        self._cands = cands
        self._seed = seed
        allidx = np.arange(len(cands))
        self.root = ClusterNode(allidx, self._medoid(allidx), 1.0, "raiz")
        self.split(self.root, k)
        return self.root

    def split(self, node, k):
        """Divide um nó em até `k` sub-grupos (k-means local nos pixels do nó).
        Idempotente: re-dividir substitui os filhos. Retorna os filhos."""
        if self._Z is None:
            return []
        from sklearn.cluster import KMeans
        idx = node.idx
        k = max(1, min(int(k), len(idx)))
        if k == 1 or len(idx) < 2:
            node.children = []
            return []
        Z = self._Z[idx]
        lab = KMeans(n_clusters=k, n_init=4, random_state=self._seed).fit(Z).labels_
        children = []
        for g in np.unique(lab):
            sub = idx[lab == g]
            ch = ClusterNode(sub, self._medoid(sub), len(sub) / len(self._cands),
                             f"{node.path}.{len(children) + 1}", parent=node)
            children.append(ch)
        children.sort(key=lambda c: c.size, reverse=True)
        node.children = children
        return children

    def _child_centroids(self, node):
        """Centróides dos filhos no espaço padronizado Z (k, F)."""
        return np.array([self._Z[ch.idx].mean(0) for ch in node.children])

    def segment_area(self, x0, y0, x1, y1, node, step, inside=None, cap=200000):
        """MAPA de clusters CONTÍGUO da área: lê a grade de passo `step` e atribui
        CADA célula ao filho de `node` mais próximo (centróide em features). Retorna
        (labels[Hs,Ws] int -1=inválido, rect (col0,row0,w,h)) p/ pintar como raster.
        Diferente de assignments (que só devolve a amostra em pontos)."""
        if node is None or not node.children:
            return None, None
        cents = self._child_centroids(node)
        sel = self.sel
        col0 = int(min(x0, x1)); row0 = int(min(y0, y1))
        col1 = int(max(x0, x1)); row1 = int(max(y0, y1))
        if hasattr(sel.ts, "safe_window"):
            rmin, cmin, rmax, cmax = sel.ts.safe_window()
            row0 = max(row0, rmin); col0 = max(col0, cmin)
            row1 = min(row1, rmax); col1 = min(col1, cmax)
        w = col1 - col0; h = row1 - row0
        if w < 1 or h < 1:
            return None, None
        step = max(1, int(step))
        while (h // step) * (w // step) > cap and step < max(h, w):
            step += max(1, step // 2)
        cube = sel.ts.read_block(row0, col0, h, w, step)     # (Hs, Ws, M, B)
        Hs, Ws = cube.shape[:2]
        labels = np.full((Hs, Ws), -1, np.int32)
        for iy in range(Hs):
            for ix in range(Ws):
                curve = cube[iy, ix]
                if not np.isfinite(curve).all():
                    continue
                if inside is not None:
                    ry = h / Hs; rx = w / Ws
                    if not inside(row0 + int((iy + 0.5) * ry), col0 + int((ix + 0.5) * rx)):
                        continue
                z = (sel.fx.extract(curve) - self._mu) / self._sd
                labels[iy, ix] = int(np.argmin(np.linalg.norm(cents - z, axis=1)))
        return labels, (col0, row0, w, h)

    def _medoid(self, idx):
        center = self._Z[idx].mean(0)
        best = idx[int(np.argmin(np.linalg.norm(self._Z[idx] - center, axis=1)))]
        return self._cands[int(best)]

    def similar_to(self, x0, y0, x1, y1, target_features, k=10, n=3000, seed=0,
                   inside=None, exclude=None, min_px=None):
        """Busca-por-exemplo: os `k` pixels da área MAIS parecidos com `target_features`
        (features de um protótipo) → cai na lista de sugestões do 'propor'."""
        cands = self.sel.candidates(x0, y0, x1, y1, n=n, seed=seed, inside=inside)
        if exclude is not None:
            cands = [c for c in cands if (c["row"], c["col"]) not in exclude]
        if not cands:
            return []
        feats = np.array([c["features"] for c in cands], float)
        mu, sd = feats.mean(0), feats.std(0) + 1e-6
        zt = (np.asarray(target_features, float) - mu) / sd
        for c in cands:
            z = (np.asarray(c["features"], float) - mu) / sd
            c["_m"] = -float(np.linalg.norm(z - zt))
            c["pred"] = None
        cands.sort(key=lambda c: c["_m"], reverse=True)
        area = abs((x1 - x0) * (y1 - y0))
        return self.sel._greedy_diverse(cands, k, area, min_px=min_px, div_thr=0.0)
