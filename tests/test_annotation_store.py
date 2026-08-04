"""Testes do AnnotationStore — sem duplicata/conflito, find_at, query, roundtrip."""
import numpy as np

from ts_annotator.core.annotation_store import AnnotationStore


def _store(tmp_path):
    return AnnotationStore(str(tmp_path / "points.json"), tol_px=3, period="2022_2023")


def test_add_and_find(tmp_path):
    s = _store(tmp_path)
    s.add_or_update(100, 200, "2c", curve=np.zeros((12, 4)))
    assert len(s) == 1
    assert s.find_at(100, 200)["class"] == "2c"
    assert s.find_at(100, 201) is not None      # dentro da tolerância
    assert s.find_at(100, 999) is None          # longe


def test_no_duplicate_same_location(tmp_path):
    s = _store(tmp_path)
    s.add_or_update(100, 200, "2c")
    pt, updated = s.add_or_update(101, 201, "2c")  # mesmo local (tol) -> atualiza, não duplica
    assert updated is True
    assert len(s) == 1


def test_no_conflict_overwrites_label(tmp_path):
    s = _store(tmp_path)
    s.add_or_update(100, 200, "1c")
    s.add_or_update(100, 200, "2c")              # reclassifica o MESMO local
    assert len(s) == 1
    assert s.find_at(100, 200)["class"] == "2c"  # vira 2c, não fica os dois


def test_query_viewport(tmp_path):
    s = _store(tmp_path)
    s.add_or_update(100, 200, "2c")
    s.add_or_update(5000, 6000, "1c")
    vis = s.query(150, 50, 300, 200)             # x0,y0,x1,y1 (col,row)
    assert len(vis) == 1 and vis[0]["class"] == "2c"


def test_roundtrip_persist(tmp_path):
    p = str(tmp_path / "points.json")
    s = AnnotationStore(p, tol_px=3)
    s.add_or_update(10, 20, "savana", curve=np.full((12, 4), 0.3))
    s2 = AnnotationStore(p, tol_px=3)            # recarrega do disco
    assert len(s2) == 1
    assert s2.find_at(10, 20)["class"] == "savana"


def test_remove(tmp_path):
    s = _store(tmp_path)
    s.add_or_update(10, 20, "agua")
    assert s.remove_at(10, 20) is True
    assert len(s) == 0
