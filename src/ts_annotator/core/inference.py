"""Shared inference kernel — read a time-series block, run the model, return classes.

The SAME kernel serves the synchronous area classify (controller) and the batch
tile job (ClassifyAllWorker): read_block → mask incomplete curves → build_X →
predict_proba → argmax. Month/band counts come from the cube's own shape (no
12×4 hardcode — a project with a different series length works or fails loudly
in build_X, never by silent misreshape).
"""
from __future__ import annotations

import numpy as np


def classify_block(ts, net, mu, sd, row0, col0, h, w, step=1, batch=8192,
                   after_read=None, before_infer=None):
    """Classifica um bloco (h, w) lido com decimação ``step``.

    Retorna ``(ci, valid, (Hs, Ws))``: ``valid`` é a máscara achatada de pixels
    com curva completa; ``ci`` são os índices de classe SÓ dos válidos (ordem de
    ``np.flatnonzero(valid)``). Callbacks opcionais p/ UI: ``after_read(Hs, Ws)``
    após a leitura (o gargalo), ``before_infer(n_valid)`` antes da GPU.
    """
    from ts_annotator.core import trainer

    cube = ts.read_block(row0, col0, h, w, step)          # (Hs, Ws, meses, bandas)
    Hs, Ws = cube.shape[:2]
    if after_read is not None:
        after_read(Hs, Ws)
    flat = cube.reshape(Hs * Ws, cube.shape[2], cube.shape[3])
    valid = np.isfinite(flat).all((1, 2))
    if not valid.any():
        return np.zeros(0, np.int64), valid, (Hs, Ws)
    if before_infer is not None:
        before_infer(int(valid.sum()))
    X = trainer.build_X(np.transpose(flat[valid], (0, 2, 1)),
                        bands=getattr(ts, "band_names", None))
    proba = trainer.predict_proba(net, mu, sd, X, batch=batch)
    return proba.argmax(1), valid, (Hs, Ws)
