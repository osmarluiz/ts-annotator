"""Blocagem espacial e zona morta da validação cruzada.

O que se testa aqui é a alegação que o artigo faz: um ponto de validação não é
vizinho de um ponto de treino. Bloco sozinho NÃO garante isso — células vizinhas
do k-means se tocam —, então quem garante é o buffer.
"""
import numpy as np
import pytest

from ts_annotator.core.trainer import _nearest_dist, spatial_blocks

rng = np.random.default_rng(0)


def _clumps(n_clumps=12, per=25, spread=3.0, step=500.0):
    """Aglomerados apertados e bem separados — como pontos colhidos por 'parecidos
    com esta curva' dentro de alguns talhões."""
    centres = np.array([[(i % 4) * step, (i // 4) * step] for i in range(n_clumps)], float)
    return np.repeat(centres, per, axis=0) + rng.normal(0, spread, (n_clumps * per, 2))


def test_auto_reproduces_the_old_formula():
    """n_blocks None/0 tem de dar o mesmo que a fórmula que vivia no worker."""
    coords = _clumps()
    n, folds = len(coords), 5
    esperado = int(min(n // 4, max(6 * folds, 30)))
    for arg in (None, 0):
        g = spatial_blocks(coords, folds, arg)
        assert len(set(g.tolist())) == esperado


def test_n_blocks_is_honoured():
    coords = _clumps()
    assert len(set(spatial_blocks(coords, 5, 7).tolist())) == 7
    assert len(set(spatial_blocks(coords, 5, 40).tolist())) == 40


def test_blocks_alone_do_not_separate():
    """A razão do buffer existir: pontos colados podem cair em blocos distintos."""
    # duas nuvens densas e SOBREPOSTAS: o k-means tem de cortar no meio delas
    coords = rng.normal(0, 10.0, (400, 2))
    g = spatial_blocks(coords, 5, 20)
    piores = []
    for b in set(g.tolist()):
        dentro, fora = coords[g == b], coords[g != b]
        piores.append(_nearest_dist(dentro, fora).min())
    assert min(piores) < 1.0, "esperava vizinhos imediatos em blocos diferentes"


@pytest.mark.parametrize("buf", [0.0, 5.0, 50.0, 200.0])
def test_buffer_guarantees_the_separation(buf):
    """Depois de descartar o que está a menos de `buf`, nada sobra mais perto."""
    # espalhamento bem maior que o maior buffer testado, senão o treino esvazia
    # (caso coberto por test_buffer_wider_than_the_data_is_an_error)
    coords = rng.normal(0, 2000.0, (600, 2))
    va = np.arange(0, 200)
    tr = np.arange(200, 600)
    if buf > 0:
        tr = tr[_nearest_dist(coords[tr], coords[va]) >= buf]
    d = _nearest_dist(coords[va], coords[tr])
    if buf > 0:
        assert d.min() >= buf
    assert len(tr) > 0


def test_buffer_wider_than_the_data_is_an_error():
    """Buffer maior que o espalhamento esvazia o treino: tem de falhar dizendo o número."""
    from ts_annotator.core.trainer import spatial_cv
    coords = rng.normal(0, 20.0, (60, 2))          # tudo dentro de ~100 px
    X = rng.normal(0, 1, (60, 5, 12)).astype("float32")
    y = np.tile([0, 1], 30)
    groups = spatial_blocks(coords, 2, 4)
    with pytest.raises(ValueError, match="buffer de 5000"):
        spatial_cv(X, y, groups, 2, epochs=1, k=2, coords=coords, buffer_px=5000.0)


def test_buffer_zero_drops_nobody():
    coords = _clumps()
    va, tr = np.arange(0, 50), np.arange(50, len(coords))
    mantidos = tr[_nearest_dist(coords[tr], coords[va]) >= 0.0]
    assert len(mantidos) == len(tr)


def test_nearest_dist_edges():
    a = np.zeros((3, 2))
    assert np.isinf(_nearest_dist(a, np.zeros((0, 2)))).all()
    assert len(_nearest_dist(np.zeros((0, 2)), a)) == 0
    d = _nearest_dist(np.array([[0.0, 0.0]]), np.array([[3.0, 4.0]]))
    assert abs(d[0] - 5.0) < 1e-9
