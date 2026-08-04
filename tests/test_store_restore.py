"""AnnotationStore.restore — base do undo: volta o dict EXATO (id + metadados)."""
import numpy as np

from ts_annotator.core.annotation_store import AnnotationStore


def test_restore_after_remove_preserves_id_and_meta(tmp_path):
    s = AnnotationStore(str(tmp_path / "p.json"))
    pt, _ = s.add_or_update(10, 20, "a", curve=np.zeros((12, 4)))
    pt["_clean"] = 0.42                     # metadado de treino gravado no dict
    old = s.find_at(10, 20)
    s.remove_at(10, 20)
    assert s.find_at(10, 20) is None
    s.restore(old)
    back = s.find_at(10, 20)
    assert back["id"] == pt["id"] and back["_clean"] == 0.42 and back["class"] == "a"


def test_restore_replaces_updated_point(tmp_path):
    s = AnnotationStore(str(tmp_path / "p.json"))
    s.add_or_update(5, 5, "a", curve=np.zeros((12, 4)))
    old = s.find_at(5, 5)
    s.add_or_update(5, 5, "b", curve=np.zeros((12, 4)))   # re-rotula (troca o dict)
    assert s.find_at(5, 5)["class"] == "b"
    s.restore(old)                                        # undo → volta o antigo
    back = s.find_at(5, 5)
    assert back["class"] == "a" and back["id"] == old["id"]
    assert len(s) == 1                                    # substituiu, não duplicou


def test_restore_persists(tmp_path):
    path = str(tmp_path / "p.json")
    s = AnnotationStore(path)
    s.add_or_update(1, 1, "x", curve=np.zeros((12, 4)))
    old = s.find_at(1, 1)
    s.remove_at(1, 1)
    s.restore(old)
    s2 = AnnotationStore(path)                            # reabre do disco
    assert s2.find_at(1, 1)["class"] == "x"
