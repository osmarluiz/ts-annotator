"""Helpers headless do WorkspaceLoader (sem Qt): yaml, classes, timeseries, descoberta."""
import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin  # noqa: E402

from ts_annotator.app.workspace import (  # noqa: E402
    WorkspaceError,
    _build_timeseries,
    _class_colors,
    _discover_layers,
    _read_yaml,
)


def test_read_yaml_missing_raises(tmp_path):
    with pytest.raises(WorkspaceError, match="project.yaml"):
        _read_yaml(str(tmp_path))


def test_class_colors_both_forms():
    cfg = {"classes": {"a": "#ff0000", "b": {"color": "#00ff00"}, "c": None}}
    out = _class_colors(cfg)
    assert out["a"] == "#ff0000" and out["b"] == "#00ff00"
    assert out["c"].startswith("#")            # fallback da paleta


def test_build_timeseries_glob_and_errors(tmp_path):
    prof = dict(driver="GTiff", width=8, height=8, count=4, dtype="uint16",
                crs="EPSG:32723", transform=from_origin(0, 8, 1, 1), nodata=65535)
    for m in range(2):
        with rasterio.open(str(tmp_path / f"c{m}.tif"), "w", **prof) as ds:
            ds.write(np.zeros((4, 8, 8), np.uint16))
    ts = _build_timeseries(str(tmp_path), {"timeseries": {"paths": "c*.tif"}})
    assert len(ts.paths) == 2
    with pytest.raises(WorkspaceError, match="timeseries"):
        _build_timeseries(str(tmp_path), {})
    with pytest.raises(WorkspaceError, match="não bateu"):
        _build_timeseries(str(tmp_path), {"timeseries": {"paths": "nada*.tif"}})


def test_discover_layers(tmp_path):
    (tmp_path / "layers").mkdir()
    (tmp_path / "layers" / "a.gpkg").write_bytes(b"x")
    (tmp_path / "layers" / "b.geojson").write_bytes(b"x")
    (tmp_path / "layers" / "ignore.txt").write_bytes(b"x")
    out = _discover_layers(str(tmp_path), {})
    assert set(out) == {"a", "b"}
