"""Demo — anotador map-native (RIDE/AOI). Thin launcher.

Wires up the project-specific data (rasters, time-series, class list, paths) and
hands an ``AppContext`` to ``AnnotatorWindow`` (all UI/interaction lives there).
The reference set (dataset_balanced.npz) is used only to seed the CLASS LIST (its
'cultivo' label re-split into 1c/2c/3c by the valley FEATURE); SIMILARITY is built
at runtime from the user's own LABELED points (controller._ensure_similarity), not
from the reference. User points go to the AnnotationStore (JSON, shown on the map).
"""
import glob
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(message)s")

import numpy as np
import rasterio
from PyQt6.QtWidgets import QApplication

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ts_annotator.core.annotation_store import AnnotationStore  # noqa: E402
from ts_annotator.core.features import FeatureExtractor  # noqa: E402
from ts_annotator.core.raster_source import RasterSource  # noqa: E402
from ts_annotator.core.selection import SelectionEngine  # noqa: E402
from ts_annotator.core.similarity import SimilarityEngine  # noqa: E402
from ts_annotator.core.timeseries import TimeSeriesSource  # noqa: E402
from ts_annotator.render import (  # noqa: E402
    CLASS_COLORS,
    make_class_colorizer,
    make_rgb_colorizer,
    make_scalar_colorizer,
)
from ts_annotator.ui.similarity_panel import CurveView, SimilarityPanel  # noqa: E402
from ts_annotator.ui.tiled_map_view import TiledMapView  # noqa: E402
from ts_annotator.app.theme import apply_theme  # noqa: E402
from ts_annotator.app.session import AppContext  # noqa: E402
from ts_annotator.ui.annotator_window import AnnotatorWindow  # noqa: E402

SITS = r"D:\PROJECTS\sits"
COG = SITS + r"\experiments\jstars\ride_pipeline\outputs\cog"
RDIR = SITS + r"\data\RIDE_DF_UTM_out2022_set2023"
REF = SITS + r"\experiments\jstars\ride_pipeline\skill_loop\dataset_balanced.npz"
# ---- recorte AOI (crop pronto em E:) — o anotador trabalha SÓ nessa área ----
AOI_DIR = r"E:\sits_aoi_crop"
AOI_COL0, AOI_ROW0 = 2305, 4608   # origem do crop na grade cheia (remapear pontos antigos/store)
ORDER = [202210, 202211, 202212, 202301, 202302, 202303, 202304, 202305, 202306, 202307, 202308, 202309]
OUT_JSON = os.path.join(os.path.dirname(__file__), "points.json")
ANTIGOS_JSON = os.path.join(os.path.dirname(__file__), "points_antigos.json")
CELLS_JSON = os.path.join(os.path.dirname(__file__), "cells.json")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "it_latest.pt")
# rasters de predição por-modelo (grandes) ficam junto do dado, em E:
PRED_DIR = os.path.join(AOI_DIR, "predictions")

CLS_COLORS = {
    "1_ciclo": "#fee08b", "2_ciclos": "#74add1", "3_ciclos": "#313695",
    "savana": "#b8e186", "afloramento": "#b8860b", "mata_galeria": "#1a9850",
    "agua": "#3b78c2", "cafe": "#8c510a", "campo1": "#a6d96a",
    "campo2": "#66bd63", "pasto": "#fdae61", "mata_seca": "#9e9e9e",
}
CYC = {1: "1_ciclo", 2: "2_ciclos", 3: "3_ciclos"}


def load_reference(fx):
    """dataset_balanced (limpo) -> curvas + labels, com cultivo re-dividido em 1c/2c/3c pela feature."""
    d = np.load(REF, allow_pickle=True)
    cur = np.transpose(d["cur4"], (0, 2, 1))  # (N,4,12) -> (N,12,4)
    lab = d["label"].astype(str)
    out = []
    for i in range(len(lab)):
        f = fx.extract(cur[i])
        cls = CYC[int(f[0])] if lab[i] == "cultivo" else lab[i]
        out.append((f, cls, i))
    return out


def load_antigos_store(transform, crs):
    """The old reliable DF points as a SELECTABLE working dataset (editable/trainable).

    Built once from ref_points.npz (already remapped to the crop grid, curves embedded)
    and cached as points_antigos.json; loaded fast on later runs. Returns None if the
    npz is unavailable.
    """
    store = AnnotationStore(ANTIGOS_JSON, transform=transform, crs=crs, period="2022_2023", tol_px=3)
    if len(store) > 0:
        print(f"dataset 'antigos': {len(store)} pontos", flush=True)
        return store
    try:
        rp = np.load(os.path.join(os.path.dirname(__file__), "ref_points.npz"), allow_pickle=True)
    except Exception as e:
        print("ref_points n/d:", e, flush=True)
        return None
    rows = np.asarray(rp["row"]) - AOI_ROW0
    cols = np.asarray(rp["col"]) - AOI_COL0
    labs, curves = rp["label"], rp["curve"]
    pts = []
    for i in range(len(rows)):
        r, c = int(rows[i]), int(cols[i])
        pt = {"row": r, "col": c, "period": "2022_2023", "class": str(labs[i]),
              "source": "reference", "id": i + 1}
        if transform is not None:
            x, y = transform * (c + 0.5, r + 0.5)
            pt["x"], pt["y"] = float(x), float(y)
        cur = curves[i]
        if cur is not None:
            pt["curve"] = np.round(np.asarray(cur, float), 4).tolist()
        pts.append(pt)
    store.bulk_add(pts)
    print(f"dataset 'antigos': {len(store)} pontos (construído de ref_points.npz)", flush=True)
    return store


def build_context():
    """Wire the RIDE/AOI project data into an AppContext for the window."""
    fx = FeatureExtractor()

    class_src = RasterSource(AOI_DIR + r"\products\basemap_class_aoi.tif")
    TR = class_src.transform

    files = [AOI_DIR + rf"\cube\ride_{m}_aoi.tif" for m in ORDER]
    offs = [round((rasterio.open(f).transform.f - TR.f) / TR.a) for f in files]  # 0 (mesma grade)
    ts = TimeSeriesSource(files, bands=(1, 2, 3, 4), row_offsets=offs, nodata=65535, scale=10000)

    # ---- basemaps AGRUPADOS: DADO (observação) x RESULTADO (produto) ----
    # separa o sinal do palpite do modelo -> evita viés de rotular olhando o resultado.
    _defs = [
        ("Dado (observação)", "input", [
            ("NDVI temporal", "ndvitemporal", lambda: make_rgb_colorizer(nodata=0)),
            ("MNF",           "mnf",          lambda: make_rgb_colorizer(nodata=0)),
            ("Fenológico",    "pheno",        lambda: make_rgb_colorizer(nodata=0)),
        ]),
        ("Resultado (produto)", "products", [
            ("Classificação", "class",     lambda: make_class_colorizer(CLASS_COLORS, 255)),
            ("Intensidade",   "intensity", lambda: make_scalar_colorizer(30, 180, "RdYlGn", nodata=255)),
            ("Regen",         "regen",     lambda: make_rgb_colorizer(nodata=0)),
        ]),
    ]
    basemaps = {}
    basemap_groups = []
    for gname, sub, layers in _defs:
        present = []
        for label, nm, colf in layers:
            if nm == "class":
                src = class_src
            else:
                p = AOI_DIR + rf"\{sub}\basemap_{nm}_aoi.tif"
                if not os.path.exists(p):
                    print(f"{nm} n/d: {p}", flush=True); continue
                try:
                    src = RasterSource(p)
                except Exception as e:
                    print(f"{nm} n/d:", e, flush=True); continue
            basemaps[label] = (src, colf())
            present.append(label)
        if present:
            basemap_groups.append((gname, present))

    # abre numa camada de OBSERVAÇÃO (não no mapa classificado) — anti-viés
    init_label = basemap_groups[0][1][0] if basemap_groups else "Classificação"
    view = TiledMapView(*basemaps[init_label])
    view.vb.setLimits(xMin=-class_src.width, xMax=2 * class_src.width,
                      yMin=-class_src.height, yMax=2 * class_src.height)
    vlayer_paths = {os.path.splitext(os.path.basename(p))[0]: p
                    for p in glob.glob(SITS + r"\data\external\*.gpkg")}

    items = load_reference(fx)
    # a engine começa VAZIA — o controller a reconstrói a partir dos pontos
    # rotulados do dataset ativo (similaridade = só os teus rótulos). A referência
    # serve apenas p/ derivar a lista de classes/cores.
    eng = SimilarityEngine(k=3, novelty_thr=0.45)
    sel = SelectionEngine(ts, fx, eng)
    classes = {c: CLS_COLORS.get(c, "#cccccc") for c in sorted({it[1] for it in items})}
    classes["mata_seca"] = CLS_COLORS["mata_seca"]  # nova classe p/ rotular à mão (cinza, floresta decídua)
    print(f"classes (da referência): {list(classes)}", flush=True)

    panel = SimilarityPanel(classes)
    curve_view = CurveView()

    # datasets selecionáveis: começamos pelos ANTIGOS (referência DF, editáveis)
    work_store = AnnotationStore(OUT_JSON, transform=TR, crs=class_src.crs, period="2022_2023", tol_px=3)
    print(f"dataset 'trabalho': {len(work_store)} pontos", flush=True)
    datasets = {}
    antigos_store = load_antigos_store(TR, class_src.crs)
    if antigos_store is not None:
        datasets["antigos (referência DF)"] = antigos_store
    datasets["trabalho (points.json)"] = work_store
    # dataset UNIFICADO p/ revisão: todas as classes, cobertura COMPORTAMENTAL + ESPACIAL
    # (diversidade de conteúdo e geográfica), nomes campo1<->mata_galeria corrigidos.
    # Rótulo tentativo do modelo — confirmar/corrigir. mata_seca entra depois (labels do usuário).
    uni = os.path.join(os.path.dirname(__file__), "dataset_unificado.json")
    if os.path.exists(uni):
        datasets["treino (curado)"] = AnnotationStore(
            uni, transform=TR, crs=class_src.crs, period="2022_2023", tol_px=3)
        print(f"dataset 'treino curado': {len(datasets['treino (curado)'])} pts", flush=True)
    dataset_default = next(iter(datasets))  # antigos, se houver; senão trabalho

    return AppContext(
        view=view, curve_view=curve_view, panel=panel,
        datasets=datasets, dataset_default=dataset_default, ts=ts, fx=fx,
        eng=eng, sel=sel, class_src=class_src, transform=TR,
        basemaps=basemaps, basemap_groups=basemap_groups, vlayer_paths=vlayer_paths,
        classes=classes, cls_colors=CLS_COLORS,
        model_dir=MODEL_DIR, model_path=MODEL_PATH, pred_dir=PRED_DIR, init_label=init_label,
        # colunas de domínio na tabela de revisão (RIDE/agrícola): fenologia da
        # curva — índices de FeatureExtractor.NAMES (n_cycles=0, amplitude=2)
        review_features=[("ciclos", 0), ("amplit.", 2)],
    )


def main():
    app = QApplication([])
    apply_theme(app)
    ctx = build_context()
    win = AnnotatorWindow(ctx)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
