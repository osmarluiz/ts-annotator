"""DEPRECATED shim — colorizers moved to :mod:`ts_annotator.render`.

The old ``CogMapView`` widget (single-image reload viewer) was superseded by
:class:`ts_annotator.ui.tiled_map_view.TiledMapView` and has been removed.
This module re-exports the colorizer factories for backward compatibility;
import from ``ts_annotator.render`` in new code.
"""
from ts_annotator.render import (  # noqa: F401
    CLASS_COLORS,
    make_class_colorizer,
    make_rgb_colorizer,
    make_scalar_colorizer,
)
