"""Criação de projeto SEM GUI — escreve ``project.yaml`` + pastas de convenção.

Isolado do PyQt de propósito: o :class:`~ts_annotator.ui.new_project.NewProjectDialog`
só coleta campos e chama :func:`init_project`; toda a lógica (validação da série,
serialização do yaml, scaffold de pastas) mora aqui e é testável headless.

O projeto resultante ABRE direto pelo :func:`~ts_annotator.app.workspace.load_workspace`
mesmo sem basemaps prontos: o loader cai no fundo cru da série (mês central) —
ver ``_fallback_basemap``. Gerar as visualizações bonitas é a Fatia 2.
"""
from __future__ import annotations

import os
import re

import yaml

_IMG_EXTS = (".tif", ".tiff")
# pastas de convenção do workspace (algumas o loader recria; criar aqui deixa o
# projeto legível no explorador e pronto pra Fatia 2)
_SCAFFOLD = ("visualizations", "layers", "annotations", "models", "predictions")


class ProjectInitError(ValueError):
    """Parâmetros inválidos para criar o projeto — a mensagem diz o quê."""


def find_images(folder, exts=_IMG_EXTS):
    """Lista os rasters de ``folder`` (não recursivo), ordenados por nome —
    ``ride_YYYYMM.tif`` sai em ordem cronológica. Retorna caminhos absolutos."""
    if not os.path.isdir(folder):
        raise ProjectInitError(f"pasta de imagens não existe: {folder}")
    out = [os.path.join(folder, f) for f in os.listdir(folder)
           if os.path.splitext(f)[1].lower() in exts]
    return sorted(out, key=_natural_key)


def _natural_key(p):
    """Ordenação natural pelo basename (números como números, não texto)."""
    s = os.path.basename(p).lower()
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def _rgb_index_from_1based(spec, n_bands):
    """"3,2,1" (números de banda 1-based p/ R,G,B) → índices 0-based na leitura.

    Retorna None se ``spec`` vazio (o loader escolhe o default do cubo)."""
    if not spec:
        return None
    parts = [t.strip() for t in str(spec).replace(";", ",").split(",") if t.strip()]
    if len(parts) != 3:
        raise ProjectInitError("RGB precisa de exatamente 3 números de banda (ex.: 3,2,1)")
    try:
        idx = [int(t) - 1 for t in parts]
    except ValueError:
        raise ProjectInitError(f"RGB inválido: {spec!r} (use números, ex.: 3,2,1)")
    if any(i < 0 or i >= n_bands for i in idx):
        raise ProjectInitError(f"RGB fora do intervalo 1..{n_bands}: {spec!r}")
    return idx


def _parse_bands(spec):
    """"1,2,3,4" → (1,2,3,4). Vazio → default (1,2,3,4)."""
    if not spec:
        return (1, 2, 3, 4)
    try:
        b = tuple(int(t.strip()) for t in str(spec).replace(";", ",").split(",") if t.strip())
    except ValueError:
        raise ProjectInitError(f"bandas inválidas: {spec!r} (use números, ex.: 1,2,3,4)")
    if not b:
        raise ProjectInitError("a lista de bandas ficou vazia")
    return b


def build_config(images, *, name, period="", bands=(1, 2, 3, 4), nodata=65535,
                 scale=10000, rgb_index=None, classes=None, project_dir=None):
    """Monta o dict do ``project.yaml`` (puro, sem tocar disco).

    ``images`` já ordenadas. Caminhos DENTRO de ``project_dir`` viram relativos
    (projeto portátil); fora, ficam absolutos (o loader aceita ambos)."""
    if not images:
        raise ProjectInitError("selecione ao menos uma imagem da série")
    paths = []
    for p in images:
        ap = os.path.abspath(p)
        if project_dir:
            try:
                rel = os.path.relpath(ap, os.path.abspath(project_dir))
                ap = rel if not rel.startswith("..") and not os.path.isabs(rel) else ap
            except ValueError:
                pass   # drives diferentes no Windows — mantém absoluto
        paths.append(ap.replace("\\", "/"))
    cfg = {
        "name": name or (os.path.basename(project_dir) if project_dir else "projeto"),
        "timeseries": {
            "paths": paths,
            "bands": list(bands),
            "nodata": int(nodata),
            "scale": float(scale),
        },
        "classes": dict(classes) if classes else {"nao_sei": "#7a7a7a"},
    }
    if period:
        cfg["period"] = str(period)
    if rgb_index is not None:
        cfg["rgb_index"] = list(rgb_index)
    return cfg


def init_project(project_dir, images, *, name=None, period="", bands="1,2,3,4",
                 nodata=65535, scale=10000, rgb="", classes=None, overwrite=False):
    """Cria o projeto: valida, escreve ``project.yaml`` e faz o scaffold de pastas.

    ``bands``/``rgb`` aceitam string ("1,2,3,4" / "3,2,1" 1-based) ou já parseados.
    Retorna o caminho absoluto do projeto. Levanta :class:`ProjectInitError`."""
    project_dir = os.path.abspath(project_dir)
    yml = os.path.join(project_dir, "project.yaml")
    if os.path.exists(yml) and not overwrite:
        raise ProjectInitError(f"já existe um project.yaml em {project_dir} (use overwrite)")
    imgs = list(images)
    if not imgs:
        raise ProjectInitError("selecione ao menos uma imagem da série")
    missing = [p for p in imgs if not os.path.exists(p)]
    if missing:
        raise ProjectInitError(f"imagens não encontradas: {missing[:3]}")

    band_tuple = bands if isinstance(bands, (tuple, list)) else _parse_bands(bands)
    rgb_index = rgb if (rgb is None or isinstance(rgb, (tuple, list))) \
        else _rgb_index_from_1based(rgb, len(band_tuple))

    cfg = build_config(imgs, name=name, period=period, bands=band_tuple, nodata=nodata,
                       scale=scale, rgb_index=rgb_index, classes=classes,
                       project_dir=project_dir)

    os.makedirs(project_dir, exist_ok=True)
    for sub in _SCAFFOLD:
        os.makedirs(os.path.join(project_dir, sub), exist_ok=True)
    with open(yml, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return project_dir
