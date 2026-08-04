"""Trainer — IT (InceptionTime-style temporal CNN) na GPU + spatial-CV + cleanlab.

Reaproveita a arquitetura oficial do pipeline (IM/IT). Treina nas 5 channels
(4 bandas + NDVI) x 12 meses. spatial_cv faz GroupKFold por grupo espacial ->
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


def build_X(cur4):
    """(N,4,meses) reflectância -> (N,5,meses) com NDVI como 5º canal."""
    from ts_annotator.core.features import NIR_IDX, RED_IDX
    c = np.asarray(cur4, "float32")
    nd = np.clip((c[:, NIR_IDX] - c[:, RED_IDX]) / (c[:, NIR_IDX] + c[:, RED_IDX] + 1e-6), -1, 1)
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


def spatial_cv(X, y, groups, nc, lr=1e-3, epochs=200, k=5, progress=None, epoch_cb=None):
    """spatial-CV ESTRATIFICADA (StratifiedGroupKFold) -> OOF + métricas.

    progress(fold, k, fold_bacc) ao fim de cada fold (métrica incremental);
    epoch_cb(fold, k, epoch, epochs, loss) durante o fit de cada fold (barra +
    curva de loss ao vivo — o CV é o grosso do tempo e antes era mudo).
    """
    y = np.asarray(y)
    g = np.asarray([str(x) for x in groups])
    ng = len(set(g.tolist()))
    k = max(2, min(k, ng))
    try:
        folds = list(StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=0).split(X, y, g))
    except Exception:
        folds = list(GroupKFold(k).split(X, y, g))   # fallback se estratificação não couber
    oof = np.zeros((len(y), nc), "float32")
    for fi, (tr, va) in enumerate(folds):
        _ec = (lambda e, E, loss, _f=fi: epoch_cb(_f + 1, k, e, E, loss)) if epoch_cb else None
        net, mu, sd = fit(X[tr], y[tr], nc, lr, epochs, progress=_ec)
        oof[va] = predict_proba(net, mu, sd, X[va])
        if progress:
            fb = float(balanced_accuracy_score(y[va], oof[va].argmax(1)))
            progress(fi + 1, k, fb)
    pred = oof.argmax(1)
    return {
        "oof_proba": oof, "pred": pred, "n_groups": ng, "k": k,
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
