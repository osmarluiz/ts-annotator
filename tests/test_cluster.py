"""ClusterEngine — descoberta hierárquica DIRIGIDA (árvore divisiva top-down)."""
import numpy as np
import pytest

pytest.importorskip("sklearn")
rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin  # noqa: E402

from ts_annotator.core.cluster import ClusterEngine  # noqa: E402
from ts_annotator.core.features import FeatureExtractor  # noqa: E402
from ts_annotator.core.selection import SelectionEngine  # noqa: E402
from ts_annotator.core.similarity import SimilarityEngine  # noqa: E402
from ts_annotator.core.timeseries import TimeSeriesSource  # noqa: E402

W, H, NOD = 80, 60, 65535


@pytest.fixture()
def engine(tmp_path):
    """Cubo sintético: metade norte 1 ciclo, metade sul 2 ciclos (curva limpa)."""
    paths = []
    for m in range(12):
        p = str(tmp_path / f"m{m:02d}.tif")
        prof = dict(driver="GTiff", width=W, height=H, count=4, dtype="uint16",
                    crs="EPSG:32723", transform=from_origin(0, H, 1, 1), nodata=NOD)
        base = np.zeros((4, H, W), np.uint16)
        base[0] = base[1] = 800
        base[2] = 1000
        g1 = 0.5 - 0.45 * np.cos(2 * np.pi * m / 12)
        g2 = 0.5 - 0.45 * np.cos(4 * np.pi * m / 12)
        nir = np.where(np.arange(H)[:, None] < H // 2, g1, g2) * 9000 + 800
        base[3] = nir.astype(np.uint16)
        with rasterio.open(p, "w", **prof) as ds:
            ds.write(base)
        paths.append(p)
    ts = TimeSeriesSource(paths, nodata=NOD)
    sim = SimilarityEngine()
    sim.load([])
    return ClusterEngine(SelectionEngine(ts, FeatureExtractor(), sim))


def test_discover_root_splits_into_k(engine):
    root = engine.discover(0, 0, W, H, k=2, step=2, seed=1)
    assert root is not None and root.path == "raiz"
    assert len(root.children) == 2
    # densidade explícita: info reporta amostras e passo efetivo
    assert engine.info["n_valid"] > 0 and engine.info["eff_step"] >= 1


def test_density_step_controls_sample_count(engine):
    """Passo menor = mais amostras (densidade explícita)."""
    engine.discover(0, 0, W, H, k=2, step=1, seed=5)
    dense = engine.info["n_valid"]
    engine.discover(0, 0, W, H, k=2, step=4, seed=5)
    sparse = engine.info["n_valid"]
    assert dense > sparse, f"step 1 devia amostrar mais que step 4: {dense} vs {sparse}"


def test_cap_raises_effective_step(engine):
    """Teto de segurança: passo pequeno em área grande → passo efetivo sobe."""
    root = engine.discover(0, 0, W, H, k=2, step=1, seed=6, cap=200)
    assert engine.info["eff_step"] > 1 or engine.info["capped"]
    assert engine.info["n_valid"] <= 200 or engine.info["capped"]
    assert root is not None
    # ordenados por tamanho; frac dos filhos soma ~ frac do pai (aninhado)
    sizes = [c.size for c in root.children]
    assert sizes == sorted(sizes, reverse=True)
    assert abs(sum(c.frac for c in root.children) - root.frac) < 1e-6
    for c in root.children:
        assert np.isfinite(np.asarray(c.medoid["curve"])).all()


def test_two_regimes_separate_at_root(engine):
    root = engine.discover(0, 0, W, H, k=2, step=2, seed=2)
    rows = sorted(c.medoid["row"] for c in root.children)
    assert rows[0] < H // 2 <= rows[1], f"regimes não separaram: {rows}"


def test_split_child_nests_inside_parent(engine):
    """Dividir um FILHO cria netos cujos pixels são SUBCONJUNTO do filho (aninhado)."""
    root = engine.discover(0, 0, W, H, k=2, step=2, seed=3)
    child = root.children[0]
    grandkids = engine.split(child, 2)
    assert child.children is grandkids and len(grandkids) >= 1
    parent_set = set(child.idx.tolist())
    total = 0
    for g in grandkids:
        assert set(g.idx.tolist()) <= parent_set, "neto deve ser subconjunto do filho"
        assert g.parent is child and g.path.startswith(child.path + ".")
        total += g.size
    assert total == child.size          # partição exata (sem perder/duplicar pixel)


def test_resplit_replaces_children(engine):
    root = engine.discover(0, 0, W, H, k=2, step=2, seed=4)
    c = root.children[0]
    engine.split(c, 2)
    engine.split(c, 3)                  # re-dividir substitui
    assert len(c.children) <= 3
