"""Trainer — IT (InceptionTime-style temporal CNN) na GPU + spatial-CV + cleanlab.

Reaproveita a arquitetura oficial do pipeline (IM/IT). Treina nos canais que o
projeto declara — as bandas do cubo, mais NDVI derivado quando red/NIR estão
entre elas (o legado B,G,R,NIR vira 5 canais, um projeto SAR VV/VH treina nos
2). spatial_cv faz GroupKFold por grupo espacial ->
predições out-of-fold -> métricas honestas; cleanlab ranqueia prováveis erros
de rótulo a partir dessas probas OOF.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

DEV = "cuda" if torch.cuda.is_available() else "cpu"


class _IM(nn.Module):
    def __init__(self, ic, nf=32, ks=(9, 19, 39), bn=32):
        super().__init__()
        self.b = nn.Conv1d(ic, bn, 1, bias=False)
        self.cv = nn.ModuleList([nn.Conv1d(bn, nf, k, padding=k // 2, bias=False) for k in ks])
        self.mp = nn.MaxPool1d(3, 1, 1)
        self.cp = nn.Conv1d(ic, nf, 1, bias=False)
        self.bn = nn.BatchNorm1d(nf * 4)
        self.r = nn.ReLU()

    def forward(self, x):
        xb = self.b(x)
        o = [c(xb) for c in self.cv]
        o.append(self.cp(self.mp(x)))
        return self.r(self.bn(torch.cat(o, 1)))


class IT(nn.Module):
    def __init__(self, ic, nc, nf=32, dp=6):
        super().__init__()
        self.bk = nn.ModuleList([_IM(ic if i == 0 else nf * 4, nf) for i in range(dp)])
        self.g = nn.AdaptiveAvgPool1d(1)
        self.dr = nn.Dropout(0.2)
        self.fc = nn.Linear(nf * 4, nc)

    def forward(self, x):
        for b in self.bk:
            x = b(x)
        return self.fc(self.dr(self.g(x).squeeze(-1)))


def build_X(cur, bands=None):
    """(N,canais,meses) -> X de treino, com NDVI anexado quando há red/NIR.

    ``bands`` são os NOMES declarados no projeto: com red e NIR entre eles, o
    NDVI entra como canal extra, calculado das posições nomeadas; sem o par
    (ex.: SAR VV/VH), os canais do cubo entram como estão. ``bands=None`` é o
    contrato legado B,G,R,NIR, que segue anexando NDVI de 2/3.
    """
    from ts_annotator.core.features import NIR_IDX, RED_IDX, ndvi_pair
    c = np.asarray(cur, "float32")
    pair = (RED_IDX, NIR_IDX) if bands is None else ndvi_pair(bands)
    if pair is None:
        return c
    r, n = pair
    nd = np.clip((c[:, n] - c[:, r]) / (c[:, n] + c[:, r] + 1e-6), -1, 1)
    return np.concatenate([c, nd[:, None]], 1).astype("float32")


def fit(Xt, yt, nc, lr=1e-3, epochs=200, mu=None, sd=None, progress=None):
    """progress(epoch, epochs, loss) — chamado A CADA época com a loss média
    (acumulada no device, 1 sync/época; custo ~zero) p/ a curva de treino."""
    if mu is None:
        mu = Xt.mean((0, 2), keepdims=True)
        sd = Xt.std((0, 2), keepdims=True) + 1e-6
    cw = torch.tensor([len(yt) / (nc * max((yt == i).sum(), 1)) for i in range(nc)],
                      dtype=torch.float32, device=DEV)
    net = IT(Xt.shape[1], nc).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr, weight_decay=1e-2)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    lf = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.05)
    xt = torch.tensor((Xt - mu) / sd, device=DEV)
    yt_t = torch.tensor(yt, dtype=torch.long, device=DEV)
    for e in range(epochs):
        net.train()
        pm = torch.randperm(len(yt_t), device=DEV)
        tot = torch.zeros((), device=DEV)
        nb = 0
        for b in range(0, len(yt_t), 256):
            i = pm[b:b + 256]
            opt.zero_grad()
            loss = lf(net(xt[i]), yt_t[i])
            loss.backward()
            opt.step()
            tot += loss.detach()
            nb += 1
        sch.step()
        if progress:
            progress(e + 1, epochs, float((tot / max(nb, 1)).item()))
    net.eval()
    return net, mu, sd


def predict_proba(net, mu, sd, X, batch=8192):
    # fp16/autocast na GPU: ~2x mais rápido na inferência (tensor cores), argmax
    # idêntico. Igual ao kernel otimizado do wall-to-wall. CPU cai no fp32 normal.
    out = []
    use_amp = (DEV == "cuda")
    with torch.no_grad():
        for b in range(0, len(X), batch):
            xb = torch.tensor((X[b:b + batch] - mu) / sd, device=DEV)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = net(xb)
            out.append(torch.softmax(logits.float(), 1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, net.fc.out_features), "float32")


def spatial_blocks(coords, folds=5, n_blocks=None, seed=0):
    """Blocos espaciais (k-means sobre row/col) que viram os grupos da CV.

    ``n_blocks=None`` reproduz a fórmula automática de sempre. Passar um número
    é o que permite blocos maiores ou menores que o automático.
    """
    from sklearn.cluster import KMeans
    coords = np.asarray(coords, float)
    n = len(coords)
    if n_blocks in (None, 0):
        n_blocks = int(min(n // 4, max(6 * folds, 30)))
    n_blocks = max(2, min(int(n_blocks), n))
    return KMeans(n_blocks, n_init=3, random_state=seed).fit_predict(coords)


def _nearest_dist(a, b):
    """Distância de cada linha de `a` ao ponto mais próximo de `b`."""
    if len(a) == 0 or len(b) == 0:
        return np.full(len(a), np.inf)
    from scipy.spatial import cKDTree
    return cKDTree(np.asarray(b, float)).query(np.asarray(a, float), k=1)[0]


def spatial_cv(X, y, groups, nc, lr=1e-3, epochs=200, k=5, progress=None, epoch_cb=None,
               coords=None, buffer_px=0.0):
    """spatial-CV ESTRATIFICADA (StratifiedGroupKFold) -> OOF + métricas.

    progress(fold, k, fold_bacc) ao fim de cada fold (métrica incremental);
    epoch_cb(fold, k, epoch, epochs, loss) durante o fit de cada fold (barra +
    curva de loss ao vivo — o CV é o grosso do tempo e antes era mudo).

    `coords` (N,2) em row/col habilita duas coisas. Com `buffer_px>0`, cada dobra
    DESCARTA do treino os pontos a menos dessa distância de qualquer ponto de
    validação — é o que GARANTE a separação, que blocos sozinhos não fazem
    (células vizinhas se tocam). E, com ou sem buffer, a separação obtida é
    medida e volta no resultado, então o número não depende de script externo.

    `stratified=False` no retorno significa que StratifiedGroupKFold não coube e
    o GroupKFold entrou: as métricas continuam válidas, mas sem garantia de
    classe em toda dobra. Antes isso acontecia calado.
    """
    y = np.asarray(y)
    g = np.asarray([str(x) for x in groups])
    ng = len(set(g.tolist()))
    k = max(2, min(k, ng))
    stratified = True
    try:
        folds = list(StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=0).split(X, y, g))
    except Exception:
        stratified = False
        folds = list(GroupKFold(k).split(X, y, g))   # fallback se estratificação não couber
    C = None if coords is None else np.asarray(coords, float)
    oof = np.zeros((len(y), nc), "float32")
    seps, dropped = [], 0
    for fi, (tr, va) in enumerate(folds):
        if C is not None and buffer_px > 0:
            keep = _nearest_dist(C[tr], C[va]) >= buffer_px
            dropped += int((~keep).sum())
            tr = tr[keep]
            # buffer maior que o espalhamento dos pontos esvazia o treino. Falhar
            # aqui, dizendo o número, e' melhor que treinar com o que sobrou.
            if len(tr) < nc:
                raise ValueError(
                    f"buffer de {buffer_px:g} px deixou {len(tr)} ponto(s) de treino na dobra "
                    f"{fi + 1} de {k} (mínimo {nc}, um por classe). Os pontos estão mais "
                    f"próximos entre si que o buffer pedido: reduza o buffer ou colete "
                    f"pontos mais espalhados."
                )
        if C is not None:
            d = _nearest_dist(C[va], C[tr])
            seps.append(float(d.min()) if len(d) else np.inf)
        _ec = (lambda e, E, loss, _f=fi: epoch_cb(_f + 1, k, e, E, loss)) if epoch_cb else None
        net, mu, sd = fit(X[tr], y[tr], nc, lr, epochs, progress=_ec)
        oof[va] = predict_proba(net, mu, sd, X[va])
        if progress:
            fb = float(balanced_accuracy_score(y[va], oof[va].argmax(1)))
            progress(fi + 1, k, fb)
    pred = oof.argmax(1)
    return {
        "oof_proba": oof, "pred": pred, "n_groups": ng, "k": k,
        "stratified": stratified,
        "buffer_px": float(buffer_px),
        "dropped_by_buffer": int(dropped),
        "min_separation_px": (float(min(seps)) if seps else None),
        "bacc": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "f1_per_class": f1_score(y, pred, average=None),
        "confusion": confusion_matrix(y, pred),
    }


def save_model(net, mu, sd, labs, path):
    """Salva pesos + normalização + classes (modelo re-aplicável)."""
    torch.save({"state": net.state_dict(), "mu": mu, "sd": sd,
                "labs": list(labs), "ic": net.bk[0].b.in_channels}, path)


def load_model(path):
    """Carrega -> (net, mu, sd, labs) pronto p/ predict_proba."""
    ck = torch.load(path, map_location=DEV, weights_only=False)
    labs = ck["labs"]
    net = IT(ck["ic"], len(labs)).to(DEV)
    net.load_state_dict(ck["state"])
    net.eval()
    return net, ck["mu"], ck["sd"], labs


def label_scores(y, oof_proba):
    """cleanlab: score de qualidade por ponto (baixo=suspeito) + índices piores-primeiro."""
    from cleanlab.filter import find_label_issues
    from cleanlab.rank import get_label_quality_scores
    y = np.asarray(y)
    scores = get_label_quality_scores(y, oof_proba)
    issues = find_label_issues(y, oof_proba, return_indices_ranked_by="self_confidence", n_jobs=1)
    return scores, issues
