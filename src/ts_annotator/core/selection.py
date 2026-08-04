"""SelectionEngine — sugestão ativa de pontos: propõe candidatos por estratégia.

Amostra candidatos numa região, lê as curvas, pontua, e escolhe o melhor segundo:
- "class": mais parecido com a classe-alvo (exemplo limpo p/ reforçar X).
- "uncertain": menor margem top1-top2 (ambíguo, vale rotular).
- "novelty": menor similaridade máxima (não parece com nada conhecido).
"""
from __future__ import annotations

import numpy as np


def _margin(scores):
    v = sorted(scores.values(), reverse=True)
    return (v[0] - v[1]) if len(v) > 1 else (v[0] if v else 0.0)


class SelectionEngine:
    def __init__(self, ts, fx, sim):
        self.ts = ts
        self.fx = fx
        self.sim = sim

    def _eval(self, row, col):
        curve = self.ts.read_curve(row, col)
        # exige curva COMPLETA (mesmo critério do classificador): pixel com mês
        # nodata/nuvem gera feature NaN, que envenena similaridade e a ordenação.
        if not np.isfinite(curve).all():
            return None
        f = self.fx.extract(curve)
        return {"row": int(row), "col": int(col), "curve": curve,
                "features": f, "scores": self.sim.score(f)}

    def candidates(self, x0, y0, x1, y1, n=80, seed=0, progress=None, inside=None):
        """Amostra ~n candidatos válidos na região e os avalia (features + similaridade).

        Caminho rápido (TS com read_block): lê UM bloco decimado da região via
        overview e amostra n células válidas dele — evita as n×12 leituras de 1px
        (que levavam ~19s sobre a imagem toda). Sobre área grande o candidato fica
        numa grade grossa; a curva exata (full-res) é relida ao SELECIONAR a sugestão.
        Caminho lento (TS só com read_curve): amostra pixels e lê um a um (fallback,
        usado nos testes/fontes simples).

        inside(row, col) -> bool opcional: restringe a região a um sub-conjunto do
        bbox (ex.: união de células selecionadas, não o retângulo que as contém)."""
        if hasattr(self.ts, "read_block"):
            return self._candidates_block(x0, y0, x1, y1, n, seed, progress, inside)
        return self._candidates_pointwise(x0, y0, x1, y1, n, seed, progress, inside)

    def _candidates_pointwise(self, x0, y0, x1, y1, n, seed, progress, inside=None):
        rng = np.random.default_rng(seed)
        rows = rng.integers(int(min(y0, y1)), int(max(y0, y1)) + 1, n)
        cols = rng.integers(int(min(x0, x1)), int(max(x0, x1)) + 1, n)
        out = []
        for i, (r, c) in enumerate(zip(rows, cols)):
            if inside is None or inside(int(r), int(c)):
                e = self._eval(int(r), int(c))
                if e is not None:
                    out.append(e)
            if progress is not None:
                progress(i + 1, n)
        return out

    # teto de pixels FULL-RES que um bloco decimado pode tocar quando NÃO há
    # overview (sem overview, decimar ainda lê tudo em resolução cheia). Acima
    # disso, cai no ponto-a-ponto — ler o bloco cheio da imagem toda travaria.
    _BLOCK_NATIVE_CAP = 8_000_000

    def grid_sample(self, x0, y0, x1, y1, step, inside=None, cap=80000):
        """Amostra a área numa GRADE de passo `step` px (1 pixel a cada `step`),
        devolvendo TODAS as células válidas (curva completa) — não um n fixo.

        Densidade explícita p/ a descoberta de padrões: step 1 = todo pixel;
        step 20 = 1 a cada 20. Retorna (candidatos, info) onde info tem
        n_grid (células da grade), n_valid, capped (bool), eff_step. Estoura o
        teto `cap` → subamostra aleatório (o k-means não escala além disso).
        """
        col0 = int(min(x0, x1)); row0 = int(min(y0, y1))
        col1 = int(max(x0, x1)); row1 = int(max(y0, y1))
        if hasattr(self.ts, "safe_window"):
            rmin, cmin, rmax, cmax = self.ts.safe_window()
            row0 = max(row0, rmin); col0 = max(col0, cmin)
            row1 = min(row1, rmax); col1 = min(col1, cmax)
        w = col1 - col0; h = row1 - row0
        if w < 1 or h < 1:
            return [], {"n_grid": 0, "n_valid": 0, "capped": False, "eff_step": step}
        step = max(1, int(step))
        # teto: a grade não pode passar de ~cap células (memória/k-means). Se
        # passar, aumenta o step efetivo — o usuário vê no info.
        n_grid = (h // step) * (w // step)
        eff_step = step
        while n_grid > cap and eff_step < max(h, w):
            eff_step += max(1, eff_step // 2)
            n_grid = (h // eff_step) * (w // eff_step)
        cube = self.ts.read_block(row0, col0, h, w, eff_step)
        valid = np.isfinite(cube).all(axis=(2, 3))
        ij = np.argwhere(valid)
        ry = h / cube.shape[0]; rx = w / cube.shape[1]
        if inside is not None and len(ij):
            keep = [p for p in ij
                    if inside(row0 + int((p[0] + 0.5) * ry), col0 + int((p[1] + 0.5) * rx))]
            ij = np.array(keep) if keep else np.empty((0, 2), int)
        n_valid = len(ij)
        capped = n_valid > cap
        if capped:
            ij = ij[np.random.default_rng(0).choice(n_valid, cap, replace=False)]
        out = []
        for p in ij:
            iy, ix = int(p[0]), int(p[1])
            curve = cube[iy, ix]
            f = self.fx.extract(curve)
            out.append({"row": row0 + int((iy + 0.5) * ry), "col": col0 + int((ix + 0.5) * rx),
                        "curve": curve, "features": f, "scores": self.sim.score(f)})
        return out, {"n_grid": int((h // eff_step) * (w // eff_step)), "n_valid": int(n_valid),
                     "capped": bool(capped), "eff_step": int(eff_step),
                     "area_px": int(w) * int(h)}

    def _candidates_block(self, x0, y0, x1, y1, n, seed, progress, inside=None):
        rng = np.random.default_rng(seed)
        col0 = int(min(x0, x1)); row0 = int(min(y0, y1))
        col1 = int(max(x0, x1)); row1 = int(max(y0, y1))
        # clampa aos limites onde TODOS os meses leem in-bounds: janela que cruza a
        # borda (viewport com folga do fit, retângulo arrastado pra fora) faria
        # read_block cair no boundless = leitura FULL-RES da área inteira (freeze).
        # Candidatos fora da imagem são inúteis de qualquer forma.
        if hasattr(self.ts, "safe_window"):
            rmin, cmin, rmax, cmax = self.ts.safe_window()
            row0 = max(row0, rmin); col0 = max(col0, cmin)
            row1 = min(row1, rmax); col1 = min(col1, cmax)
        w = col1 - col0; h = row1 - row0
        if w < 1 or h < 1:
            return []
        # sem overview + área grande: o bloco decimado leria full-res (bilhões de
        # px). Melhor amostrar ponto-a-ponto (n×12 leituras de 1px).
        if not getattr(self.ts, "has_overviews", False) and w * h > self._BLOCK_NATIVE_CAP:
            return self._candidates_pointwise(x0, y0, x1, y1, n, seed, progress, inside)
        # decimação p/ um bloco com folga p/ amostrar n células (rápido via overview);
        # área pequena -> step 1 (full-res, preciso); área grande -> grade grossa.
        pool = max(n * 40, 20000)
        step = max(1, int((w * h / pool) ** 0.5))
        cube = self.ts.read_block(row0, col0, h, w, step)      # (Hs, Ws, meses, bandas)
        if progress is not None:
            progress(1, 2)                                     # leitura (o gargalo) feita
        valid = np.isfinite(cube).all(axis=(2, 3))             # células com curva completa
        ij = np.argwhere(valid)
        # razão EFETIVA da decimação: Hs = h//step faz cada célula cobrir h/Hs ≥ step
        # linhas; reportar via `step` acumula drift de até ~1 célula. O centro da
        # célula reamostrada é o pixel que melhor representa a curva devolvida.
        ry = h / cube.shape[0]; rx = w / cube.shape[1]
        if inside is not None and len(ij):
            keep = [p for p in ij
                    if inside(row0 + int((p[0] + 0.5) * ry), col0 + int((p[1] + 0.5) * rx))]
            ij = np.array(keep) if keep else np.empty((0, 2), int)
        if len(ij) == 0:
            if progress is not None:
                progress(2, 2)
            return []
        take = min(n, len(ij))
        pick = rng.choice(len(ij), size=take, replace=False)
        out = []
        for pi in pick:
            iy, ix = int(ij[pi][0]), int(ij[pi][1])
            curve = cube[iy, ix]                               # (meses, bandas)
            f = self.fx.extract(curve)
            out.append({"row": row0 + int((iy + 0.5) * ry), "col": col0 + int((ix + 0.5) * rx),
                        "curve": curve, "features": f, "scores": self.sim.score(f)})
        if progress is not None:
            progress(2, 2)
        return out

    def rank(self, x0, y0, x1, y1, metric="disagreement", order="desc",
             target=None, n=80, seed=0, exclude=None, predict_fn=None,
             progress=None, inside=None):
        """Amostra candidatos, calcula sim+pred, devolve a lista ORDENADA pela métrica/ordem.

        metric="class": mais parecido com a classe-alvo; os demais via metrics.compute
        (confidence/margin/entropy/disagreement — todos sobre o vetor de probabilidades).
        """
        from ts_annotator.core import metrics as M
        cands = self.candidates(x0, y0, x1, y1, n=n, seed=seed, progress=progress, inside=inside)
        if exclude is not None:
            cands = [c for c in cands if (c["row"], c["col"]) not in exclude]
        for c in cands:
            c["pred"] = predict_fn(c["curve"]) if predict_fn else None
            if metric == "class":
                src = c["pred"] or c["scores"]
                c["_m"] = src.get(target, -1.0)
            else:
                c["_m"] = M.compute(c["scores"], c["pred"], metric)
        cands.sort(key=lambda c: c["_m"], reverse=(order == "desc"))
        return cands

    def propose(self, x0, y0, x1, y1, metric="disagreement", order="desc",
                target=None, n=80, seed=0, exclude=None, predict_fn=None):
        """Melhor candidato único (= rank()[0])."""
        r = self.rank(x0, y0, x1, y1, metric=metric, order=order, target=target,
                      n=n, seed=seed, exclude=exclude, predict_fn=predict_fn)
        return r[0] if r else None

    def _greedy_diverse(self, ranked, k, area, min_px=None, div_thr=0.5):
        """Escolhe k candidatos gulosamente na ordem da métrica, rejeitando quem está
        perto demais dos já escolhidos — no ESPAÇO (mata pixels vizinhos/idênticos) e
        no espaço de FEATURES (mata curvas quase iguais). Um segundo passe preenche
        até k ignorando as restrições, então NUNCA devolve menos que o top-k puro.

        min_px: separação mínima em pixels; default adaptativo ~metade do espaçamento
        médio do pool (espalha sem sufocar áreas pequenas). div_thr: distância mínima
        em features padronizadas (0 desliga; requer referência de padronização).
        """
        if not ranked or k <= 0:
            return []
        if min_px is None:
            min_px = max(3.0, 0.5 * (area / max(len(ranked), 1)) ** 0.5)
        min_px2 = min_px * min_px
        mean, std = self.sim._mean, self.sim._std
        use_feat = mean is not None and div_thr > 0

        def _z(c):
            return (np.asarray(c["features"], float) - mean) / std

        chosen, zsel = [], []
        for c in ranked:
            if len(chosen) >= k:
                break
            ok = True
            for o in chosen:
                if (c["row"] - o["row"]) ** 2 + (c["col"] - o["col"]) ** 2 < min_px2:
                    ok = False
                    break
            if ok and use_feat:
                zc = _z(c)
                if any(np.linalg.norm(zc - zo) < div_thr for zo in zsel):
                    ok = False
            if ok:
                chosen.append(c)
                if use_feat:
                    zsel.append(_z(c))
        if len(chosen) < k:  # fallback: completa na ordem da métrica, sem duplicar coord
            picked = {(c["row"], c["col"]) for c in chosen}
            for c in ranked:
                if len(chosen) >= k:
                    break
                rc = (c["row"], c["col"])
                if rc not in picked:
                    chosen.append(c)
                    picked.add(rc)
        return chosen[:k]

    def similar_to(self, x0, y0, x1, y1, target_features, k=10, n=3000, seed=0,
                   inside=None, exclude=None, min_px=None):
        """Busca-por-exemplo: os `k` pixels da área MAIS parecidos com `target_features`
        (features de um protótipo, ex.: a curva em foco) → cai na fila do 'propor'."""
        cands = self.candidates(x0, y0, x1, y1, n=n, seed=seed, inside=inside)
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
        return self._greedy_diverse(cands, k, area, min_px=min_px, div_thr=0.0)

    def propose_many(self, x0, y0, x1, y1, k=10, metric="disagreement", order="desc",
                     target=None, seed=0, exclude=None, predict_fn=None,
                     progress=None, min_px=None, div_thr=0.5, inside=None):
        """Top-k candidatos DIVERSOS: amostra um pool ~8x maior, ordena pela métrica e
        seleciona gulosamente com diversidade espacial + de features entre os picks
        (não só vs. o dataset já rotulado). Ver _greedy_diverse."""
        n = max(80, k * 8)
        ranked = self.rank(x0, y0, x1, y1, metric=metric, order=order, target=target,
                           n=n, seed=seed, exclude=exclude, predict_fn=predict_fn,
                           progress=progress, inside=inside)
        area = abs((x1 - x0) * (y1 - y0))
        return self._greedy_diverse(ranked, k, area, min_px=min_px, div_thr=div_thr)
