"""AnnotationStore — os pontos rotulados do usuário (JSON auto-contido, sem duplicata/conflito).

Garante integridade: um LOCAL = um rótulo (dedup por proximidade em pixels;
rotular de novo ATUALIZA, não duplica nem conflita). Suporta find_at (flag de
"já rotulado") e query por viewport (marcadores no mapa). Escrita atômica.
"""
from __future__ import annotations

import json
import os
import tempfile

import numpy as np


class AnnotationStore:
    def __init__(self, path, transform=None, crs=None, period="default", tol_px=3):
        self.path = path
        self.transform = transform   # affine pixel<->geo (opcional)
        self.crs = crs
        self.period = period
        self.tol = tol_px            # tolerância de dedup (pixels)
        self.points: list = []
        self._rc = np.empty((0, 2))
        self.load()

    # ---------- io ----------
    def load(self):
        if os.path.exists(self.path):
            with open(self.path) as f:
                self.points = json.load(f).get("points", [])
        self._reindex()

    def save(self):
        payload = {
            "format_version": "1.0",
            "crs": str(self.crs) if self.crs else None,
            "period": self.period,
            "points": self.points,
        }
        d = os.path.dirname(self.path) or "."
        fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, self.path)  # atômico

    # ---------- índice ----------
    def _reindex(self):
        self._rc = (
            np.array([[p["row"], p["col"]] for p in self.points], float)
            if self.points else np.empty((0, 2))
        )

    def find_at(self, row, col, tol=None):
        """Ponto existente dentro da tolerância (tol em pixels; default self.tol), ou None."""
        if len(self._rc) == 0:
            return None
        t = self.tol if tol is None else tol
        d = np.hypot(self._rc[:, 0] - row, self._rc[:, 1] - col)
        i = int(np.argmin(d))
        return self.points[i] if d[i] <= t else None

    def query(self, x0, y0, x1, y1):
        """Pontos no viewport (coords de pixel: x=col, y=row) — p/ marcadores."""
        if len(self._rc) == 0:
            return []
        m = (
            (self._rc[:, 1] >= x0) & (self._rc[:, 1] <= x1)
            & (self._rc[:, 0] >= y0) & (self._rc[:, 0] <= y1)
        )
        return [self.points[i] for i in np.nonzero(m)[0]]

    # ---------- escrita ----------
    def add_or_update(self, row, col, cls, curve=None, source="manual", confidence=None):
        """Adiciona OU atualiza (se já existe um ponto no local). Sem duplicata/conflito."""
        existing = self.find_at(row, col)
        pt = {"row": int(row), "col": int(col), "period": self.period, "class": cls, "source": source}
        if self.transform is not None:
            x, y = self.transform * (col + 0.5, row + 0.5)
            pt["x"], pt["y"] = float(x), float(y)
        if curve is not None:
            pt["curve"] = np.round(np.asarray(curve, float), 4).tolist()
        if confidence is not None:
            pt["confidence"] = confidence
        if existing is not None:
            pt["id"] = existing["id"]
            self.points[self.points.index(existing)] = pt
            updated = True
        else:
            pt["id"] = (max(p["id"] for p in self.points) + 1) if self.points else 1
            self.points.append(pt)
            updated = False
        self._reindex()
        self.save()
        return pt, updated

    def bulk_add(self, pts):
        """Adiciona muitos pontos de uma vez (semear dataset) — 1 reindex + 1 save."""
        self.points.extend(pts)
        self._reindex()
        self.save()

    def restore(self, pt):
        """Recoloca um ponto EXATAMENTE como era (base do undo): substitui o do
        mesmo local ou re-insere — preserva id e metadados (_clean/_pred/...)."""
        existing = self.find_at(pt["row"], pt["col"])
        if existing is not None:
            self.points[self.points.index(existing)] = pt
        else:
            self.points.append(pt)
        self._reindex()
        self.save()

    def remove_at(self, row, col):
        existing = self.find_at(row, col)
        if existing is not None:
            self.points.remove(existing)
            self._reindex()
            self.save()
        return existing is not None

    # ---------- operações de CLASSE (renomear/mesclar/remover) ----------
    def relabel(self, old, new):
        """Reatribui TODOS os pontos da classe `old` para `new` (renomear/mesclar).
        Retorna quantos pontos mudaram. Um único save (barato)."""
        n = 0
        for p in self.points:
            if p.get("class") == old:
                p["class"] = new
                n += 1
        if n:
            self.save()
        return n

    def class_counts(self):
        """{classe: nº de pontos} no dataset (para o gerenciador de classes)."""
        from collections import Counter
        return dict(Counter(p.get("class") for p in self.points))

    def __len__(self):
        return len(self.points)
