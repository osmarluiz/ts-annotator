"""AppContext — the bundle of services + project data the UI window operates on.

The launcher (e.g. examples/run_annotation_demo.py) wires up the project-specific
data sources (rasters, time-series, reference set, classes, paths) and hands this
context to :class:`ts_annotator.ui.annotator_window.AnnotatorWindow`, which owns all
the UI and interaction logic. This keeps project specifics out of the reusable app.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AppContext:
    # core services / data
    view: Any                 # TiledMapView (already constructed on the initial basemap)
    curve_view: Any           # CurveView
    panel: Any                # SimilarityPanel
    datasets: dict            # name -> AnnotationStore (selectable working datasets)
    dataset_default: str      # which dataset is active on start
    ts: Any                   # TimeSeriesSource (click -> curve)
    fx: Any                   # FeatureExtractor
    eng: Any                  # SimilarityEngine (loaded on the reference set)
    sel: Any                  # SelectionEngine
    class_src: Any            # RasterSource of the class basemap (defines image size/transform/crs)
    transform: Any            # affine transform (geo <-> pixel)

    # basemaps + layers
    basemaps: dict            # label -> (RasterSource, colorizer)
    basemap_groups: list      # [(group_name, [labels...]), ...] for the grouped "base:" combo
    vlayer_paths: dict        # name -> path of scannable vector overlays

    # classes / colors
    classes: dict             # class name -> color (drives the SimilarityPanel)
    cls_colors: dict          # full class -> hex color map (markers, predictions)

    # model persistence
    model_dir: str
    model_path: str
    pred_dir: str             # where per-model prediction rasters (pred_<model>.tif) live

    # initial basemap label (which observation layer to open on)
    init_label: str

    # optional per-domain extra columns in the review table: [(label, feature_idx)]
    # — the launcher/project decides (e.g. RIDE shows ciclos+amplitude); the
    # generic tool shows none, since "cycles" is not universal to time series.
    review_features: Any = None

    # hierarquia de classes (superclasses) — opcional; default = cada classe é seu próprio super
    class_super: dict = field(default_factory=dict)   # classe fina -> superclasse
    super_colors: dict = field(default_factory=dict)  # superclasse -> hex
