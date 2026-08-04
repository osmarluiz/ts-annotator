# Architecture

> **Detailed per-file/class/method spec:** see [`SPECS/`](SPECS/README.md) — the
> "where do I edit X" reference (índice, fluxos ponta-a-ponta, modelo de threads,
> convenções de coordenada, e uma tabela onde-editar). This file is the high-level
> map; `SPECS/` is the exhaustive one.

TSA is layered so the domain logic runs and is tested without a GUI, the UI stays
thin, and project specifics never leak into the reusable app.

```
src/ts_annotator/
  core/                 # domain — NO PyQt, 100% unit-tested
      raster_source.py      lazy COG windowed reader + LRU cache
      timeseries.py         click -> pixel curve (12 mo × 4 bands), nodata-aware
      features.py           curve -> interpretable phenology features (cycle-by-valley, no z-norm)
      similarity.py         per-class k-NN similarity, % absolute, novelty
      selection.py          active-learning candidate proposer (metric × order × scope)
      metrics.py            AL scalars (confidence/margin/entropy/similarity/novelty/disagreement)
      trainer.py            InceptionTime + StratifiedGroupKFold spatial-CV + cleanlab
      annotation_store.py   labeled-points JSON = map layer + training dataset
      prediction_raster.py  per-model pred_<model>.tif + JSON sidecar (resumable batch)
      vector_layer.py       vector overlays -> pixel outlines (STRtree)

  ui/                   # presentation (PyQt6 + pyqtgraph)
      tiled_map_view.py     TiledMapView — slippy tiled-pyramid gigapixel map
      map_view.py           colorizers (class/scalar/rgb) + legacy CogMapView
      similarity_panel.py   CurveView (the curve) + SimilarityPanel (class cards)
      map_toolbar.py        MapToolbar — floating dataset/base/overlay/model bar
      review_panel.py       ReviewPanel — cleanlab/discrepancy ranked list + filters
      classify_panel.py     ClassifyPanel — area scope + batch-job progress UI
      annotator_window.py   AnnotatorWindow — assembles nav | (map/curve) | cards
                            + floating toolbar; owns the interaction logic

  app/                  # application layer (project-independent)
      theme.py              dark theme (qdarktheme + accent QSS)
      workers.py            TrainWorker + ClassifyAllWorker (off the UI thread)
      config.py             GOALS (active-learning goal presets)
      session.py            AppContext — the services+data bundle handed to the window

examples/               # thin launchers (project-specific wiring)
      run_annotation_demo.py  RIDE/AOI data + build_context() -> AnnotatorWindow
```

## The boundary

The **launcher** knows the project (raster paths, class names/colors, the reference
set, the time-series source). It builds an **`AppContext`** (`app/session.py`) and hands
it to **`AnnotatorWindow`**, which is generic: it operates only on the context, never on
hardcoded paths. This is the seam that a future `WorkspaceLoader` (see
`WORKSPACE.md`) slots into — replacing the hand-written `build_context()` with folder
discovery + `project.yaml`.

```
build_context()  ──►  AppContext  ──►  AnnotatorWindow(ctx)
 (project data)       (services)        (generic UI + logic)
```

## History

`AnnotatorWindow` was extracted from a single 1359-line `main()` in
`run_annotation_demo.py`: the closures became methods, the shared `state`/`tstate`/…
dicts became instance attributes. Behaviour is unchanged (verified by launching the
real app and a headless smoke test over the click/propose/review/navigate paths).

## Roadmap (next)

The widget split is DONE (phase 3, jul/2026): `MapCanvas` (pyqtgraph overlays +
grid/draw-rect tools + basemap overlay), `MapToolbar` (the floating bar), and
`AnnotatePanel` / `ReviewPanel` / `TrainPanel` / `ClassifyPanel` are all extracted
as dumb views emitting signals. The controller is DONE too: the cross-cutting
logic (on_click/on_label/navigate, propose, review, model management, train/
classify orchestration, worker threads) lives in `AnnotatorController`
(`app/controller.py`), which owns the shared state dicts and falls back to the
window via `__getattr__` for widget access; `AnnotatorWindow` (~290 lines) is
assembly + signal wiring only, plus widget/method aliases for backward
compatibility. The `WorkspaceLoader` is DONE (`app/workspace.py`: project.yaml +
folder discovery -> AppContext, see `WORKSPACE.md`) along with the `tsa <project>`
CLI entry-point (`app/main.py`). The hand-written launcher in `examples/` remains
as the RIDE-specific wiring (custom reference re-split). Model/prediction
provenance is DONE too (`core/version_store.py`): every training run saves a new
`models/it_vN/` with `meta.yaml` (params, spatial-CV metrics, annotation-state
hash, model sha1); model identity everywhere is the .pt content sha1. Next:
JOSS packaging (tests/CI/README/example dataset).
