"""VersionStore — modelos versionados auto-descritos (proveniência da cadeia
anotações → modelo → predição; ver docs/WORKSPACE.md).

Cada treino vira ``models/<prefix>_vN/`` com ``model.pt`` (pesos + mu/sd + classes,
via trainer.save_model — re-aplicável) e ``meta.yaml`` (hiperparâmetros, métricas do
spatial-CV, nº de pontos, hash das anotações usadas, sha1 do próprio .pt, data).
Nada é sobrescrito: comparar versões = ler os meta.yaml.

A IDENTIDADE de um modelo é o sha1 do arquivo .pt (``file_sha1``) — robusta a
cópia/migração de máquina, diferente do mtime. O PredictionRaster usa isso pra
decidir se um raster de predição pertence ao modelo ativo.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import re
from datetime import datetime

import yaml

_SHA_CACHE: dict = {}   # (path, mtime, size) -> sha1 (evita re-hash a cada refresh)


def file_sha1(path) -> str:
    """sha1 do CONTEÚDO do arquivo (identidade de modelo; cacheado por mtime+size)."""
    st = os.stat(path)
    key = (os.path.abspath(path), st.st_mtime, st.st_size)
    hit = _SHA_CACHE.get(key)
    if hit is not None:
        return hit
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    _SHA_CACHE[key] = h.hexdigest()
    return _SHA_CACHE[key]


def anno_hash(points) -> str:
    """Hash do ESTADO da rotulagem usada no treino: (id,row,col,classe) ordenados.

    Curvas ficam de fora (pesadas e derivadas da coord); mudar um rótulo, adicionar
    ou remover um ponto muda o hash — é o elo anotações→modelo do meta.yaml.
    """
    items = sorted((int(p["id"]), int(p["row"]), int(p["col"]), str(p["class"])) for p in points)
    return hashlib.sha1(json.dumps(items).encode()).hexdigest()


def model_stem(model_path) -> str:
    """Nome-base p/ artefatos por-modelo (ex.: pred_<stem>.tif).

    Modelos versionados são sempre ``<versão>/model.pt`` — o stem útil é o nome da
    VERSÃO (senão toda versão colidiria em "model"). Arquivos .pt soltos (legado)
    continuam usando o nome do arquivo.

    Aceita os dois separadores de propósito. Um projeto é uma pasta que se copia
    entre máquinas, e o sidecar da predição guarda o caminho como ele foi escrito,
    então um projeto criado no Windows tem de ser lido no Linux sem perder a
    identidade da versão. ``os.path.basename`` no POSIX não corta em ``\\``.
    """
    parts = str(model_path).replace("\\", "/").rstrip("/").split("/")
    stem = os.path.splitext(parts[-1])[0]
    if stem == "model" and len(parts) > 1:
        return parts[-2]
    return stem


class VersionStore:
    def __init__(self, model_dir, prefix="it"):
        self.dir = model_dir
        self.prefix = prefix
        self._rx = re.compile(rf"^{re.escape(prefix)}_v(\d+)$")

    def list_versions(self):
        """[{name, num, path (model.pt), meta (dict|None)}] em ordem de versão."""
        out = []
        for d in glob.glob(os.path.join(self.dir, f"{self.prefix}_v*")):
            m = self._rx.match(os.path.basename(d))
            mp = os.path.join(d, "model.pt")
            if not m or not os.path.exists(mp):
                continue
            meta = None
            try:
                with open(os.path.join(d, "meta.yaml"), encoding="utf-8") as f:
                    meta = yaml.safe_load(f)
            except Exception:
                pass
            if not isinstance(meta, dict):
                meta = None   # yaml truncado que parseia como escalar não pode virar .get()
            out.append({"name": os.path.basename(d), "num": int(m.group(1)),
                        "path": mp, "meta": meta})
        return sorted(out, key=lambda v: v["num"])

    def latest(self):
        """Caminho do model.pt da versão mais nova (None se não há versões)."""
        vs = self.list_versions()
        return vs[-1]["path"] if vs else None

    def next_name(self):
        vs = self.list_versions()
        return f"{self.prefix}_v{(vs[-1]['num'] + 1) if vs else 1}"

    def save_version(self, net, mu, sd, labs, meta=None):
        """Salva pesos + meta.yaml numa versão NOVA. Retorna (nome, caminho do .pt)."""
        from ts_annotator.core import trainer
        name = self.next_name()
        vdir = os.path.join(self.dir, name)
        os.makedirs(vdir, exist_ok=True)
        mpath = os.path.join(vdir, "model.pt")
        trainer.save_model(net, mu, sd, labs, mpath)
        doc = {"format": 1, "name": name, "type": self.prefix,
               "created": datetime.now().isoformat(timespec="seconds"),
               "labs": [str(x) for x in labs],
               "model_sha1": file_sha1(mpath)}
        doc.update(meta or {})
        ypath = os.path.join(vdir, "meta.yaml")
        tmp = ypath + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            yaml.safe_dump(doc, f, allow_unicode=True, sort_keys=False)
        os.replace(tmp, ypath)   # atômico: crash no meio não deixa meta truncado
        return name, mpath
