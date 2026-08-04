"""Rede de teste do AnnotatorController via demo_project headless.

Constrói a janela inteira (que instancia o controller) sobre o projeto sintético
que embarca no repo, e exercita o caminho central — wiring de estado + roundtrip
rótulo→undo. O `test_controller_surface` fixa a SUPERFÍCIE pública do controller:
é a rede que protege a decomposição do god-object (um método que suma/caia na
classe errada quebra aqui, não em produção).
"""
import os

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

DEMO = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "examples", "demo_project"))

# superfície pública esperada (handlers ligados aos sinais da UI + lifecycle).
# Guard do refactor: dividir o controller em mixins NÃO pode perder nenhum destes.
_SURFACE = [
    # overlays / dataset / grid
    "toggle_layer", "refresh_overlay", "set_overlay", "set_overlay_alpha",
    "on_dataset_change", "toggle_cell",
    # markers / labeling
    "on_class_color", "on_pred_class_visible", "refresh_markers", "select_point",
    "predict_point", "on_click", "on_label", "on_class_added",
    # classes (renomear/mesclar/remover) + undo
    "open_class_manager", "rename_class", "remove_class", "on_remove",
    "undo_last", "on_points_visible", "on_class_visible", "on_triage",
    "on_pts_alpha", "set_layer_color", "navigate",
    # propose / discover
    "fill_suggestions", "do_propose", "do_discover", "discover_split",
    "discover_enter", "apply_patterns", "clear_patterns", "discover_up",
    "discover_root", "go_to_pattern", "propose_from_pattern",
    "on_sugg_select", "advance_suggestion", "skip_suggestion",
    # hover / toolbar / scope
    "on_hover", "set_grid", "set_cols", "set_size", "set_draw",
    "on_scope_change", "on_draw_toggled", "update_scope_labels", "update_goals",
    # review / model / classify / train / lifecycle
    "refresh_review_list", "on_review_point", "zoom_to_point",
    "list_model_files", "load_active_model", "refresh_models", "on_model_change",
    "classify_scope", "classify_all", "on_classify_all_done", "do_train",
    "on_train_done", "init_workers", "shutdown",
]


@pytest.fixture
def win():
    from ts_annotator.app.workspace import load_workspace
    from ts_annotator.ui.annotator_window import AnnotatorWindow
    QApplication.instance() or QApplication([])
    ctx = load_workspace(DEMO)
    w = AnnotatorWindow(ctx)
    try:
        yield w
    finally:
        w.controller.shutdown()
        w.close()


def _finite_pixel(ctrl):
    """Um pixel com curva COMPLETA no cubo demo (varre uma grade a partir do centro)."""
    H, W = ctrl.class_src.height, ctrl.class_src.width
    for r in range(H // 2, min(H, H // 2 + 40)):
        for c in range(W // 2, min(W, W // 2 + 40)):
            if np.isfinite(ctrl.ts.read_curve(r, c)).all():
                return r, c
    raise AssertionError("nenhum pixel com curva completa no demo_project")


def test_controller_builds_and_state_is_shared(win):
    ctrl = win.controller
    # a janela aponta pros MESMOS dicts de estado (contrato do __getattr__/aliases)
    assert win.state is ctrl.state
    assert win.tstate is ctrl.tstate
    assert win.pred_state is ctrl.pred_state
    # sem modelo treinado: predict_point devolve None (não inventa classe)
    assert ctrl.predict_point(np.ones((12, 4), float)) is None


def test_controller_surface(win):
    """Todo handler público existe e é chamável — o guarda do mixin-split."""
    ctrl = win.controller
    faltando = [m for m in _SURFACE if not callable(getattr(ctrl, m, None))]
    assert not faltando, f"handlers ausentes no controller: {faltando}"


def test_label_then_undo_roundtrip(win):
    ctrl = win.controller
    r, c = _finite_pixel(ctrl)
    n0 = len(ctrl.store)
    ctrl.on_click(r, c)
    assert ctrl.state["last"] is not None            # clique pegou a curva
    cls = next(iter(ctrl.classes))
    ctrl.on_label(cls)
    assert len(ctrl.store) == n0 + 1                 # nasceu 1 ponto
    assert ctrl.store.find_at(r, c) is not None
    assert ctrl.store.find_at(r, c)["class"] == cls
    ctrl.undo_last()
    assert len(ctrl.store) == n0                      # undo desfez
    assert ctrl.store.find_at(r, c) is None


def test_scope_labels_no_throw(win):
    # handler de rótulo do chip ÁREA roda sem estado prévio (regressão comum)
    win.controller.update_scope_labels()
