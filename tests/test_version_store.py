"""VersionStore: versões numeradas, meta.yaml, anno_hash, robustez a meta corrompido."""
import sys
import types

import pytest

from ts_annotator.core.version_store import VersionStore, anno_hash, file_sha1


@pytest.fixture()
def fake_trainer(monkeypatch):
    """save_version importa o trainer (torch) lazy — stub p/ testar sem GPU/torch."""
    mod = types.ModuleType("ts_annotator.core.trainer")

    def save_model(net, mu, sd, labs, path):
        with open(path, "wb") as f:
            f.write(b"weights:" + ",".join(labs).encode())

    mod.save_model = save_model
    # dois caminhos de resolução: sys.modules E o atributo do pacote (se o trainer
    # real já foi importado por outro teste, `from ts_annotator.core import trainer`
    # resolve pelo atributo — os dois precisam apontar pro stub)
    import ts_annotator.core as _core_pkg
    monkeypatch.setitem(sys.modules, "ts_annotator.core.trainer", mod)
    monkeypatch.setattr(_core_pkg, "trainer", mod, raising=False)
    return mod


def test_save_list_latest(tmp_path, fake_trainer):
    vs = VersionStore(str(tmp_path))
    assert vs.list_versions() == [] and vs.latest() is None and vs.next_name() == "it_v1"
    n1, p1 = vs.save_version(None, None, None, ["a", "b"], {"n_points": 10})
    n2, p2 = vs.save_version(None, None, None, ["a", "b"], {"n_points": 12})
    assert (n1, n2) == ("it_v1", "it_v2")
    vers = vs.list_versions()
    assert [v["name"] for v in vers] == ["it_v1", "it_v2"]
    assert vs.latest() == p2
    m = vers[-1]["meta"]
    assert m["labs"] == ["a", "b"] and m["n_points"] == 12
    assert m["model_sha1"] == file_sha1(p2)


def test_corrupt_meta_is_ignored(tmp_path, fake_trainer):
    """meta.yaml truncado que parseia como escalar não pode derrubar o boot."""
    vs = VersionStore(str(tmp_path))
    _, p1 = vs.save_version(None, None, None, ["a"], {})
    meta_path = tmp_path / "it_v1" / "meta.yaml"
    meta_path.write_text("apenas-uma-string")
    v = vs.list_versions()[0]
    assert v["meta"] is None          # não vira str.get(...) → AttributeError
    assert vs.next_name() == "it_v2"  # numeração continua


def test_anno_hash_sensitivity():
    pts = [{"id": 1, "row": 5, "col": 7, "class": "a"},
           {"id": 2, "row": 9, "col": 1, "class": "b"}]
    h0 = anno_hash(pts)
    assert h0 == anno_hash(list(reversed(pts)))       # ordem não importa
    pts2 = [dict(p) for p in pts]
    pts2[0]["class"] = "b"                            # mudar UM rótulo muda o hash
    assert anno_hash(pts2) != h0
    assert anno_hash(pts[:1]) != h0                   # remover ponto muda o hash


def test_file_sha1_cache_by_mtime_size(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"abc")
    h1 = file_sha1(str(p))
    p.write_bytes(b"xyz!")                            # tamanho muda → re-hash
    assert file_sha1(str(p)) != h1
