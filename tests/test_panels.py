"""GUI-panel tests (headless, offscreen). Construct the main PyQt6 panels and
exercise their data logic — hierarchy, review columns/filters, legend, class
manager, train-level selector — without a live map or GPU.

isHidden() (not isVisible()) is used to check visibility: a child of a never-shown
widget reports isVisible()==False regardless, but isHidden() reflects an explicit hide.
"""
import pytest

pytest.importorskip("PyQt6")
from PyQt6.QtWidgets import QApplication, QWidget    # noqa: E402

app = QApplication.instance() or QApplication([])


# ---------------------------------------------------------------- similarity panel
def test_similarity_hierarchy_and_super_label():
    from PyQt6.QtGui import QColor

    from ts_annotator.ui.similarity_panel import SimilarityPanel
    classes = {"1_ciclo": "#aa8866", "2_ciclos": "#bb7744", "3_ciclos": "#cc6622",
               "campo": "#88ddaa", "pasto": "#33bb66", "savana": "#99cc33", "nao_sei": "#7a7a7a"}
    cs = {"1_ciclo": "cultivo", "2_ciclos": "cultivo", "3_ciclos": "cultivo",
          "campo": "gramineas", "pasto": "gramineas", "savana": "savana", "nao_sei": "nao_sei"}
    p = SimilarityPanel(classes)
    p.set_hierarchy(cs, {"cultivo": "#e66000", "gramineas": "#339944"})
    # grupos multi-membro ganham cabeçalho (SuperCard); atômicos (savana/nao_sei) não
    assert set(p._headers) == {"cultivo", "gramineas"}
    assert len(p._super_cards["cultivo"]) == 3
    assert list(p._cards)[-1] == "nao_sei"                 # nao_sei por último

    # rotular no super: clicar o CORPO do cabeçalho emite a superclasse
    got = []
    p.classChosen.connect(got.append)
    p._headers["gramineas"].label_click()
    assert got == ["gramineas"]

    # colapsar (clique na seta do cabeçalho) esconde os filhos
    p._headers["cultivo"].toggle_click()
    assert p._cards["1_ciclo"].isHidden() and not p._cards["campo"].isHidden()

    # affordance: o cabeçalho carrega a cor FORTE da família; filhos indentados com
    # trilho (rail_color) sob o grupo; classe única = card de topo com faixa forte
    assert p._headers["gramineas"].color == QColor("#339944")
    assert p._cards["1_ciclo"].indent > 0 and p._cards["savana"].indent == 0
    assert p._cards["1_ciclo"].rail_color == QColor("#e66000")   # trilho na cor do cultivo
    assert p._cards["savana"].top_level and not p._cards["1_ciclo"].top_level

    # probabilidade do modelo no NÍVEL super vai pro cabeçalho, some do card fino
    p._pred = {"gramineas": 0.9, "1_ciclo": 0.7}
    p._sim = {}
    p.pred_toggle.setChecked(True)
    p._refresh()
    assert p._headers["gramineas"].pred == 90               # exibida no cabeçalho ("P 90%")
    assert p._cards["campo"].pred is None                   # suprimido (modelo previu no super)
    assert p._cards["1_ciclo"].pred == 70                   # fino mostra o próprio


# ---------------------------------------------------------------- review table
def _pt(i, cls, **kw):
    d = {"id": i, "row": i, "col": i, "class": cls}
    d.update(kw)
    return d


def test_review_table_columns_filter_and_disc():
    from ts_annotator.ui.review_panel import ReviewPanel, _PointsModel
    classes = {"1_ciclo": "#a", "campo": "#b", "pasto": "#c"}
    pts = [
        _pt(1, "1_ciclo", _pred="1_ciclo", _ytrain="1_ciclo", _predp=0.9, _clean=0.9, _issue=False),
        _pt(2, "campo", _pred="gramineas", _ytrain="gramineas", _predp=0.8, _clean=0.8, _issue=False),
        _pt(3, "pasto", _pred="1_ciclo", _ytrain="gramineas", _predp=0.6, _clean=0.05, _issue=True),
    ]
    rp = ReviewPanel(classes)
    rp.set_points(pts, {"1_ciclo": "#a", "campo": "#b", "pasto": "#c", "gramineas": "#d"})
    assert rp.model.rowCount() == 3
    assert "nível" in [c["label"] for c in rp.cols]         # coluna nova

    # discordância compara pred vs _ytrain (nível), não a classe fina:
    assert not _PointsModel._disc(pts[1])                   # campo->gramineas == ytrain: OK
    assert _PointsModel._disc(pts[2])                       # pasto: pred 1_ciclo != gramineas
    assert _PointsModel._susp(pts[2]) and not _PointsModel._susp(pts[0])   # _issue
    assert _PointsModel._lowq(pts[2])                       # _clean<0.5

    # filtro por classe re-sincroniza
    rp.rebuild_class_filter({"1_ciclo": "#a", "pasto": "#c"})
    datas = [rp.rfilter.itemData(i) for i in range(rp.rfilter.count())]
    assert datas == [None, "1_ciclo", "pasto"]


# ---------------------------------------------------------------- layers panel
def test_layers_panel_legend_and_signals():
    from collections import Counter

    from ts_annotator.ui.layers_panel import LayersPanel
    view = QWidget()
    view.resize(800, 600)
    classes = {"1_ciclo": "#a", "campo": "#b", "nao_sei": "#7a7a7a"}
    lp = LayersPanel(view, obs_labels=["NDVI"], result_labels=[], pred_label="Predição (modelo)",
                     vlayer_names=[], classes=classes, init_label="NDVI", grid_cols=10)
    lp.update_points("ds", 5, Counter({"1_ciclo": 3, "campo": 2}))
    assert set(lp._legend_rows) >= {"1_ciclo", "campo", "nao_sei"}
    assert set(lp._pred_rows) >= {"1_ciclo", "campo"}
    assert list(lp._legend_rows)[-1] == "nao_sei"          # nao_sei por último

    got = []
    lp.predClassVisibleToggled.connect(lambda c, on: got.append((c, on)))
    lp._pred_rows["campo"][0].setChecked(False)
    assert ("campo", False) in got and "campo" in lp._pred_hidden

    # rebuild_classes zera as legendas
    lp.rebuild_classes({"1_ciclo": "#a"})
    assert lp._legend_rows == {} and lp._pred_rows == {}


# ---------------------------------------------------------------- class manager
def test_class_manager_dialog():
    from ts_annotator.ui.class_manager import ClassManagerDialog
    state = {"classes": {"campo": "#39", "pasto": "#d9"}, "counts": {"campo": 5, "pasto": 3}}
    log = []

    def get_state():
        return dict(state["classes"]), dict(state["counts"])

    def do_rename(old, new):
        log.append(("rename", old, new))
        col = state["classes"].pop(old, "#ccc")
        state["classes"].setdefault(new, col)
        state["counts"][new] = state["counts"].get(new, 0) + state["counts"].pop(old, 0)
        return True, "ok"

    def do_remove(cls, dest):
        log.append(("remove", cls, dest))
        state["classes"].pop(cls, None)
        state["counts"].pop(cls, None)
        return True, "ok"

    dlg = ClassManagerDialog(get_state, do_rename, do_remove, non_training={"nao_sei"})
    assert dlg.list.count() == 2
    dlg._do_rename("pasto", "campo")           # mesclar pasto em campo
    dlg._reload()
    assert "pasto" not in state["classes"] and dlg.list.count() == 1
    assert ("rename", "pasto", "campo") in log


# ---------------------------------------------------------------- train levels
def test_train_panel_levels():
    from ts_annotator.ui.train_panel import TrainPanel
    tp = TrainPanel()
    tp.set_superclasses({"cultivo": ["1_ciclo", "2_ciclos", "3_ciclos"],
                         "gramineas": ["campo", "pasto"], "agua": ["agua"]})
    assert set(tp._super_cb) == {"cultivo", "gramineas"}    # só multi-membro
    tp.level_combo.setCurrentIndex(0)                       # classes finas
    assert tp.collapse_config() == set()
    tp.level_combo.setCurrentIndex(1)                       # superclasses
    assert tp.collapse_config() == {"cultivo", "gramineas"}
    tp.level_combo.setCurrentIndex(2)                       # personalizado
    tp._super_cb["cultivo"].setChecked(False)
    assert tp.collapse_config() == {"gramineas"}
