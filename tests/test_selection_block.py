"""SelectionEngine — caminho por BLOCO: clamp ao safe_window e filtro `inside` (união)."""
import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin  # noqa: E402

from ts_annotator.core.features import FeatureExtractor  # noqa: E402
from ts_annotator.core.selection import SelectionEngine  # noqa: E402
from ts_annotator.core.similarity import SimilarityEngine  # noqa: E402
from ts_annotator.core.timeseries import TimeSeriesSource  # noqa: E402

W, H, NOD = 60, 50, 65535


@pytest.fixture()
def sel(tmp_path):
    paths = []
    rng = np.random.default_rng(0)
    for m in range(12):
        p = str(tmp_path / f"m{m:02d}.tif")
        prof = dict(driver="GTiff", width=W, height=H, count=4, dtype="uint16",
                    crs="EPSG:32723", transform=from_origin(0, H, 1, 1), nodata=NOD)
        arr = rng.integers(500, 5000, (4, H, W)).astype(np.uint16)
        with rasterio.open(p, "w", **prof) as ds:
            ds.write(arr)
        paths.append(p)
    ts = TimeSeriesSource(paths, nodata=NOD)
    fx = FeatureExtractor()
    sim = SimilarityEngine()
    sim.load([])
    return SelectionEngine(ts, fx, sim)


def test_block_path_clamps_out_of_bounds(sel):
    """Área além da borda (viewport do fit) não pode virar boundless full-res."""
    cands = sel.candidates(-100, -100, W + 100, H + 100, n=20, seed=1)
    assert cands, "sem candidatos"
    for c in cands:
        assert 0 <= c["row"] < H and 0 <= c["col"] < W


def test_inside_filter_union(sel):
    """`inside` restringe à UNIÃO (ex.: células selecionadas), não ao bbox."""
    inside = lambda r, c: c < W // 4          # só a faixa esquerda
    cands = sel.candidates(0, 0, W, H, n=15, seed=2, inside=inside)
    assert cands
    for c in cands:
        assert c["col"] < W // 4


def test_inside_filter_survives_pointwise_fallback(sel):
    """Sem overviews + área 'grande' cai no ponto-a-ponto — o filtro tem que ir junto."""
    sel._BLOCK_NATIVE_CAP = 100                # força o fallback
    inside = lambda r, c: r >= H // 2
    cands = sel.candidates(0, 0, W, H, n=25, seed=3, inside=inside)
    for c in cands:
        assert c["row"] >= H // 2


def test_propose_many_k_and_dedup(sel):
    out = sel.propose_many(0, 0, W, H, k=6, metric="novelty", seed=4)
    assert len(out) == 6
    coords = {(c["row"], c["col"]) for c in out}
    assert len(coords) == 6                    # sem duplicatas de coordenada
