"""PredictionRaster: identidade de modelo (sha1 vs mtime legado), resume, escrita."""
import os
import time

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")

from ts_annotator.core.prediction_raster import PredictionRaster  # noqa: E402

W, H = 64, 48
TR = rasterio.transform.from_origin(0, H, 1, 1)
CRS = "EPSG:32723"
LABS = ["a", "b"]
COLORS = {"a": "#ff0000", "b": "#00ff00"}


@pytest.fixture()
def model(tmp_path):
    p = tmp_path / "m.pt"
    p.write_bytes(b"fake-model-v1")
    return str(p)


def _ensure(pr):
    return pr.ensure(W, H, TR, CRS, LABS, COLORS)


def test_ensure_created_then_kept(tmp_path, model):
    pr = PredictionRaster(str(tmp_path / "pred"), model)
    assert _ensure(pr) == "created"
    meta = pr.load_meta()
    assert meta["labs"] == LABS and meta["model_sha1"] and meta["done_tiles"] == []
    assert _ensure(pr) == "kept"


def test_mtime_change_does_not_reset(tmp_path, model):
    """Identidade é o sha1 do CONTEÚDO: copiar/migrar o modelo não invalida o raster."""
    pr = PredictionRaster(str(tmp_path / "pred"), model)
    _ensure(pr)
    os.utime(model, (time.time() + 999, time.time() + 999))
    assert _ensure(pr) == "kept"


def test_content_change_resets(tmp_path, model):
    pr = PredictionRaster(str(tmp_path / "pred"), model)
    _ensure(pr)
    pr.write_block(np.zeros((8, 8), np.uint8), 0, 0)
    with open(model, "wb") as f:
        f.write(b"fake-model-v2-retrained!")   # re-treinado → conteúdo muda
    assert _ensure(pr) == "reset"
    assert pr.load_meta()["done_tiles"] == []


def test_legacy_sidecar_falls_back_to_mtime(tmp_path, model):
    """Sidecar antigo (sem model_sha1) segue válido pelo mtime — upgrade não reseta."""
    pr = PredictionRaster(str(tmp_path / "pred"), model)
    _ensure(pr)
    meta = pr.load_meta()
    meta.pop("model_sha1")
    pr._write_meta(meta)
    assert pr.model_stale(pr.load_meta()) is False
    os.utime(model, (time.time() + 999, time.time() + 999))
    assert pr.model_stale(pr.load_meta()) is True   # mtime é o critério legado


def test_write_block_and_progress(tmp_path, model):
    pr = PredictionRaster(str(tmp_path / "pred"), model)
    _ensure(pr)
    blk = np.full((16, 16), 1, np.uint8)
    pr.write_block(blk, 8, 8)
    with rasterio.open(pr.tif) as ds:
        arr = ds.read(1)
    assert (arr[8:24, 8:24] == 1).all()
    assert (arr[0:8, :] == PredictionRaster.NODATA).all()
    pr.save_progress({"0_0", "0_1"}, complete=False)
    m = pr.load_meta()
    assert sorted(m["done_tiles"]) == ["0_0", "0_1"] and m["complete"] is False


def test_model_stem_versioned_vs_loose(tmp_path):
    from ts_annotator.core.version_store import model_stem
    assert model_stem(r"C:\x\models\it_v3\model.pt") == "it_v3"
    assert model_stem(r"C:\x\models\it_latest.pt") == "it_latest"
