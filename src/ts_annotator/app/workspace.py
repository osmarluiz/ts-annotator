"""WorkspaceLoader — monta um AppContext a partir de uma PASTA de projeto.

Convention-over-configuration (ver docs/WORKSPACE.md): o software DESCOBRE o
conteúdo do projeto pelas pastas; o ``project.yaml`` dá o mínimo que não dá pra
inferir (fonte da série temporal, classes) e overrides opcionais de estilo.

::

    <projeto>/
      project.yaml        # obrigatório: timeseries + classes; resto opcional
      visualizations/     # basemaps (COGs) — cada .tif vira uma opção no combo
      layers/             # overlays vetoriais (gpkg/geojson/shp) — listados
      annotations/        # datasets de pontos (*.json) — selecionáveis no topo
      models/             # .pt treinados (criada se faltar)
      predictions/        # rasters de predição por-modelo (criada se faltar)

Colorização inferida por visualização: >=3 bandas → RGB; 1 banda com colormap
embutido → LUT do arquivo; 1 banda contínua → colormap com vmin/vmax por
percentil (2–98) de uma amostra decimada. Override por entrada em
``visualizations:`` do yaml (type: rgb | scalar | class).

``load_workspace(dir)`` devolve o mesmo :class:`AppContext` que um launcher
escrito à mão (ex.: examples/run_annotation_demo.py) montaria — a janela não
sabe a diferença. Requer um ``QApplication`` vivo (constrói widgets).
"""
from __future__ import annotations

import glob
import logging
import os

import numpy as np
import yaml

log = logging.getLogger(__name__)

from ts_annotator.app.session import AppContext
from ts_annotator.core.annotation_store import AnnotationStore
from ts_annotator.core.features import make_extractor
from ts_annotator.core.raster_source import RasterSource
from ts_annotator.core.selection import SelectionEngine
from ts_annotator.core.similarity import SimilarityEngine
from ts_annotator.core.timeseries import TimeSeriesSource, parse_bands

_VECTOR_EXTS = (".gpkg", ".geojson", ".json", ".shp")
_FALLBACK_COLORS = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4",
                    "#46f0f0", "#f032e6", "#bcf60c", "#fabebe", "#008080", "#e6beff"]


class WorkspaceError(RuntimeError):
    """Projeto malformado — mensagem diz o que falta/onde."""


def _read_yaml(project_dir):
    p = os.path.join(project_dir, "project.yaml")
    if not os.path.exists(p):
        raise WorkspaceError(f"project.yaml não encontrado em {project_dir}")
    with open(p, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise WorkspaceError("project.yaml não é um mapeamento YAML")
    return cfg


def _class_colors(cfg):
    """classes: nome -> "#hex" ou {color: "#hex", super: nome}. Ordem do yaml = ordem dos índices.

    Filha SEM `color` cujo super tem cor declarada ganha um tom suave derivado da
    família (family_shades) — cor explícita sempre vence."""
    from ts_annotator.core.colors import family_shades

    raw, sup_of = {}, {}
    for name, v in (cfg.get("classes") or {}).items():
        raw[str(name)] = (v.get("color") if isinstance(v, dict) else v) or None
        s = v.get("super") if isinstance(v, dict) else None
        sup_of[str(name)] = str(s) if s else None
    sup_colors = _super_colors(cfg)
    fams = {}
    for name, c in raw.items():
        s = sup_of[name]
        if c is None and s in sup_colors:
            fams.setdefault(s, []).append(name)
    for s, members in fams.items():
        for name, shade in zip(members, family_shades(sup_colors[s], len(members))):
            raw[name] = shade
    return {name: str(c) if c else _FALLBACK_COLORS[i % len(_FALLBACK_COLORS)]
            for i, (name, c) in enumerate(raw.items())}


def _class_supers(cfg):
    """classe -> superclasse (default: a própria classe, se `super` não declarado)."""
    out = {}
    for name, v in (cfg.get("classes") or {}).items():
        s = v.get("super") if isinstance(v, dict) else None
        out[str(name)] = str(s) if s else str(name)
    return out


def _super_colors(cfg):
    """superclasse -> "#hex" (usada quando o treino colapsa aquele super)."""
    return {str(k): str(v) for k, v in (cfg.get("superclasses") or {}).items()}


def _build_timeseries(project_dir, cfg):
    tcfg = cfg.get("timeseries")
    if not tcfg or not tcfg.get("paths"):
        raise WorkspaceError("project.yaml precisa de timeseries.paths (lista ou glob)")
    paths = tcfg["paths"]
    if isinstance(paths, str):
        paths = sorted(glob.glob(os.path.join(project_dir, paths)))
    else:
        paths = [os.path.join(project_dir, p) for p in paths]
    if not paths:
        raise WorkspaceError(f"timeseries.paths não bateu com nenhum arquivo: {tcfg['paths']}")
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise WorkspaceError(f"timeseries: arquivos ausentes: {missing[:3]}")
    if len(paths) != 12:
        log.warning("%d rasters na série — o padrão do tool é 12 meses; confira season/months",
                    len(paths))
    # offsets de linha/coluna POR MÊS: corrigem desregistragem entre meses lida por
    # (row,col) que NÃO está no georreferenciamento (ex.: RIDE 2023, mai-set +8 linhas).
    def _offsets(key):
        v = tcfg.get(key)
        if v is None:
            return None
        if len(v) != len(paths):
            raise WorkspaceError(
                f"timeseries.{key} tem {len(v)} valores mas há {len(paths)} meses")
        return [int(x) for x in v]
    # bands aceita NOMES ("VV", "R", "NIR"...) ou índices de raster; os nomes
    # dizem o que cada canal É e decidem se há NDVI derivado no treino.
    try:
        band_idx, band_names = parse_bands(tcfg.get("bands", (1, 2, 3, 4)))
    except ValueError as e:
        raise WorkspaceError(str(e))
    return TimeSeriesSource(
        paths,
        bands=band_idx,
        band_names=band_names,
        row_offsets=_offsets("row_offsets"),
        col_offsets=_offsets("col_offsets"),
        nodata=tcfg.get("nodata", 65535),
        scale=float(tcfg.get("scale", 10000)),
    )


def _scalar_range(src, override):
    """vmin/vmax do override, ou percentis 2–98 de uma amostra decimada."""
    if "vmin" in override and "vmax" in override:
        return float(override["vmin"]), float(override["vmax"])
    arr, _ = src.read_view(0, 0, src.width, src.height, 256, 256)
    a = arr[0].astype(float)
    bad = ~np.isfinite(a)
    nod = override.get("nodata", src.nodata)
    if nod is not None:
        bad |= a == nod
    vals = a[~bad]
    if vals.size < 16:
        return 0.0, 255.0
    return float(np.percentile(vals, 2)), float(np.percentile(vals, 98))


def _colorizer_for(src, override, cls_colors):
    """Infere o colorizer de uma visualização (ou aplica o override do yaml)."""
    from ts_annotator.render import (
        make_class_colorizer,
        make_rgb_colorizer,
        make_scalar_colorizer,
    )
    kind = override.get("type")
    if kind is None:
        if src.count >= 3:
            kind = "rgb"
        else:
            try:
                import rasterio
                with rasterio.open(src.path) as ds:
                    cmap = ds.colormap(1)
            except Exception:
                cmap = None
            kind = "filemap" if cmap else "scalar"
    if kind == "rgb":
        return make_rgb_colorizer(nodata=override.get("nodata", 0))
    if kind == "bands_rgb":
        # raster CRU (ex.: um mês da série): escolhe bandas rgb_index e estica
        from ts_annotator.render import make_bands_rgb_colorizer
        ri = tuple(int(i) for i in override.get("rgb_index", (2, 1, 0)))
        lo = override.get("lo", [0, 0, 0])
        hi = override.get("hi", [3000, 3000, 3000])
        return make_bands_rgb_colorizer(ri, lo, hi, nodata=override.get("nodata", 65535))
    if kind == "class":
        colors = override.get("colors") or list(cls_colors.values())
        nod = override.get("nodata", src.nodata)
        return make_class_colorizer(colors, nodata=255 if nod is None else int(nod))
    if kind == "filemap":
        import rasterio
        with rasterio.open(src.path) as ds:
            cmap = ds.colormap(1)
        colors = ["#{:02x}{:02x}{:02x}".format(*cmap.get(i, (136, 136, 136, 255))[:3])
                  for i in range(max(cmap) + 1)]
        nod = override.get("nodata", src.nodata)
        return make_class_colorizer(colors, nodata=None if nod is None else int(nod))
    vmin, vmax = _scalar_range(src, override)
    return make_scalar_colorizer(vmin, vmax, cmap=override.get("cmap", "viridis"),
                                 nodata=override.get("nodata", src.nodata))


def _discover_basemaps(project_dir, cfg, cls_colors):
    """visualizations/*.tif → {label: (RasterSource, colorizer)} + grupos."""
    vdir = os.path.join(project_dir, cfg.get("visualizations_dir", "visualizations"))
    overrides = cfg.get("visualizations") or {}
    basemaps = {}
    labels = []
    for p in sorted(glob.glob(os.path.join(vdir, "*.tif")) + glob.glob(os.path.join(vdir, "*.tiff"))):
        stem = os.path.splitext(os.path.basename(p))[0]
        ov = overrides.get(stem) or {}
        label = ov.get("label", stem)
        try:
            src = RasterSource(p)
            basemaps[label] = (src, _colorizer_for(src, ov, cls_colors))
            labels.append(label)
        except Exception as e:
            log.warning("visualização %s n/d: %s", stem, e)
    # entradas com `path:` explícito (fora de visualizations_dir, ex.: os VRTs
    # mensais do cubo) — cada uma vira uma opção de OBSERVAÇÃO própria
    for stem, ov in overrides.items():
        if not isinstance(ov, dict) or "path" not in ov:
            continue
        label = ov.get("label", stem)
        if label in basemaps:
            continue
        p = ov["path"]
        if not os.path.isabs(p):
            p = os.path.join(project_dir, p)
        try:
            src = RasterSource(p)
            basemaps[label] = (src, _colorizer_for(src, ov, cls_colors))
            labels.append(label)
        except Exception as e:
            log.warning("visualização %s (path) n/d: %s", stem, e)
    groups_cfg = cfg.get("groups")
    if groups_cfg:
        groups = [(g, [l for l in ls if l in basemaps]) for g, ls in groups_cfg.items()]
        groups = [(g, ls) for g, ls in groups if ls]
    else:
        groups = [("Visualizações", labels)] if labels else []
    return basemaps, groups


def _discover_layers(project_dir, cfg):
    ldir = os.path.join(project_dir, cfg.get("layers_dir", "layers"))
    out = {}
    if os.path.isdir(ldir):
        for p in sorted(os.listdir(ldir)):
            if os.path.splitext(p)[1].lower() in _VECTOR_EXTS:
                out[os.path.splitext(p)[0]] = os.path.join(ldir, p)
    return out


def _discover_datasets(project_dir, cfg, transform, crs):
    """annotations/*.json → {nome: AnnotationStore}; cria points.json se vazio."""
    adir = os.path.join(project_dir, cfg.get("annotations_dir", "annotations"))
    os.makedirs(adir, exist_ok=True)
    period = str(cfg.get("period", "default"))
    datasets = {}
    for p in sorted(glob.glob(os.path.join(adir, "*.json"))):
        name = os.path.splitext(os.path.basename(p))[0]
        datasets[name] = AnnotationStore(p, transform=transform, crs=crs, period=period)
    if not datasets:
        datasets["points"] = AnnotationStore(os.path.join(adir, "points.json"),
                                             transform=transform, crs=crs, period=period)
    default = cfg.get("dataset_default")
    if default not in datasets:
        default = "points" if "points" in datasets else next(iter(datasets))
    return datasets, default


def _load_reference(project_dir, cfg, fx):
    """reference: {npz, curve_key, label_key} → itens (features, classe, i) p/ similaridade."""
    rcfg = cfg.get("reference")
    if not rcfg:
        return []
    path = os.path.join(project_dir, rcfg["npz"]) if not os.path.isabs(rcfg["npz"]) else rcfg["npz"]
    d = np.load(path, allow_pickle=True)
    cur = np.asarray(d[rcfg.get("curve_key", "curve")])
    lab = d[rcfg.get("label_key", "label")].astype(str)
    if cur.ndim != 3:
        raise WorkspaceError(f"reference: curvas com shape {cur.shape}, esperado (N, meses, bandas)")
    if cur.shape[1] == 4 and cur.shape[2] != 4:
        cur = np.transpose(cur, (0, 2, 1))   # (N,4,12) -> (N,12,4)
    return [(fx.extract(cur[i]), lab[i], i) for i in range(len(lab))]


def _fallback_basemap(ts, cfg):
    """Sem visualizações prontas: renderiza um MÊS cru da série como basemap RGB
    (bandas visíveis, stretch por percentil) — o projeto abre sem precisar gerar
    basemaps antes. Retorna (label, (RasterSource, colorizer))."""
    from ts_annotator.render import make_bands_rgb_colorizer
    tcfg = cfg.get("timeseries") or {}
    bands = tuple(tcfg.get("bands", (1, 2, 3, 4)))
    rgb_idx = cfg.get("rgb_index")
    if rgb_idx is None:
        rgb_idx = (2, 1, 0) if len(bands) >= 3 else (0, 0, 0)   # cubo B,G,R,NIR → R,G,B
    rgb_idx = tuple(int(i) for i in rgb_idx)
    nod = tcfg.get("nodata", 65535)
    src = RasterSource(ts.paths[len(ts.paths) // 2])            # mês central
    arr, _ = src.read_view(0, 0, src.width, src.height, 256, 256)   # amostra decimada
    lo, hi = [], []
    for c in rgb_idx:
        v = arr[c].astype(float)
        v = v[np.isfinite(v) & (v != nod)]
        if v.size < 16:
            lo.append(0.0); hi.append(1.0)
        else:
            lo.append(float(np.percentile(v, 2)))
            hi.append(float(np.percentile(v, 98)))
    return "Série — RGB do mês central", (src, make_bands_rgb_colorizer(rgb_idx, lo, hi, nodata=nod))


def load_workspace(project_dir) -> AppContext:
    """Monta o AppContext do projeto em ``project_dir`` (precisa de QApplication vivo)."""
    from ts_annotator.ui.similarity_panel import CurveView, SimilarityPanel
    from ts_annotator.ui.tiled_map_view import TiledMapView

    project_dir = os.path.abspath(project_dir)
    cfg = _read_yaml(project_dir)

    # calendário do projeto: rótulos dos meses (eixo da curva) + índices dos meses
    # SECOS (âncora das features fenológicas — dry_ndvi). Sem season:, valem os
    # defaults do ano agrícola Out–Set do Cerrado (contrato original do tool).
    scfg_season = cfg.get("season") or {}
    season_months = scfg_season.get("months")
    dry = scfg_season.get("dry_months")
    # descriptors: escolhe o extrator. Ausente = phenology, que e' o que todo
    # projeto recebia antes de haver escolha; dry_months so' faz sentido nele.
    kind = (cfg.get("descriptors") or "phenology").strip().lower()
    fx = (make_extractor(kind, dry=tuple(dry)) if (kind == "phenology" and dry)
          else make_extractor(kind))

    ts = _build_timeseries(project_dir, cfg)
    # grade de referência (dims/transform/crs) = 1º raster da série; as
    # visualizações DEVEM estar na mesma grade (contrato do workspace)
    grid_src = RasterSource(ts.paths[0])
    TR = grid_src.transform

    cls_colors = _class_colors(cfg)
    class_super = _class_supers(cfg)
    super_colors = _super_colors(cfg)
    basemaps, basemap_groups = _discover_basemaps(project_dir, cfg, cls_colors)

    init_label = cfg.get("init")
    if init_label not in basemaps:
        init_label = next(iter(basemaps), None)
    if init_label is None:
        # projeto sem visualizações prontas → basemap cru da série (Fatia 1: o
        # projeto abre direto das imagens; os basemaps bonitos vêm depois)
        label, bm = _fallback_basemap(ts, cfg)
        basemaps[label] = bm
        basemap_groups = (basemap_groups or []) + [("Observação", [label])]
        init_label = label
        log.info("sem visualizações — usando '%s' (mês cru) como fundo inicial", label)

    view = TiledMapView(*basemaps[init_label])
    view.vb.setLimits(xMin=-grid_src.width, xMax=2 * grid_src.width,
                      yMin=-grid_src.height, yMax=2 * grid_src.height)

    items = _load_reference(project_dir, cfg, fx)
    scfg = cfg.get("similarity") or {}
    # engine começa VAZIA — a similaridade é reconstruída em runtime a partir dos
    # pontos ROTULADOS do dataset ativo (controller._ensure_similarity). A
    # referência só semeia a lista de classes abaixo.
    eng = SimilarityEngine(k=int(scfg.get("k", 3)), novelty_thr=float(scfg.get("novelty_thr", 0.45)))
    sel = SelectionEngine(ts, fx, eng)

    # classes = yaml + o que a referência trouxer a mais
    classes = dict(cls_colors)
    for _f, c, _i in items:
        classes.setdefault(str(c), _FALLBACK_COLORS[len(classes) % len(_FALLBACK_COLORS)])

    datasets, dataset_default = _discover_datasets(project_dir, cfg, TR, grid_src.crs)
    vlayer_paths = _discover_layers(project_dir, cfg)

    model_dir = os.path.join(project_dir, "models")
    pred_dir = os.path.join(project_dir, "predictions")
    os.makedirs(model_dir, exist_ok=True)

    name = cfg.get("name", os.path.basename(project_dir))
    log.info("workspace '%s': %d visualização(ões), %d layer(s), datasets %s, referência %d pts",
             name, len(basemaps), len(vlayer_paths), list(datasets), len(items))

    return AppContext(
        view=view, curve_view=CurveView(months=season_months), panel=SimilarityPanel(classes),
        datasets=datasets, dataset_default=dataset_default, ts=ts, fx=fx,
        eng=eng, sel=sel, class_src=grid_src, transform=TR,
        basemaps=basemaps, basemap_groups=basemap_groups, vlayer_paths=vlayer_paths,
        classes=classes, cls_colors=dict(classes),
        class_super=class_super, super_colors=super_colors,
        model_dir=model_dir, model_path=os.path.join(model_dir, "it_latest.pt"),
        pred_dir=pred_dir, init_label=init_label,
        # colunas de domínio na tabela (opcional; genérico não assume "ciclos"):
        # review_features: [[label, feature_idx], ...] no project.yaml
        review_features=[tuple(x) for x in cfg.get("review_features", [])] or None,
    )
