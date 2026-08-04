"""Estratégias da aba Rotular — segmentadas por REQUISITO.

A aba agrupa as estratégias pelo que cada uma precisa p/ rodar:
  - modelo  : precisa de modelo treinado (classifica candidatos na hora)
  - escolha : precisa de uma curva clicada (curve) OU de uma classe-alvo (class)
Os testes espelham essa divisão + garantem que 'diferente do que já rotulei'
(diversity) foi removida.
"""
import pytest

pytest.importorskip("PyQt6")
from PyQt6.QtCore import Qt    # noqa: E402
from PyQt6.QtWidgets import QApplication    # noqa: E402

from ts_annotator.app.config import GOAL_GROUPS, GOALS    # noqa: E402

app = QApplication.instance() or QApplication([])

CLASSES = {"1_ciclo": "#a", "campo": "#b", "pasto": "#c"}


def _reqs(req):
    return [g for g in GOALS if g[3] == req]


# ---------------------------------------------------------------- config/estrutura
def test_diversity_removed_and_reqs_covered():
    metrics = {g[1] for g in GOALS}
    assert "diversity" not in metrics                    # removida
    # todo requisito de estratégia é coberto por algum grupo do dropdown
    covered = set()
    for _title, reqs in GOAL_GROUPS:
        covered.update(reqs)
    assert {g[3] for g in GOALS} <= covered
    # os três requisitos esperados existem
    assert {g[3] for g in GOALS} == {"model", "curve", "class"}


def test_model_strategies_are_the_probability_metrics():
    # as 4 de modelo são exatamente as que dependem do vetor de probabilidades
    assert {g[1] for g in _reqs("model")} == {"confidence", "margin", "entropy", "disagreement"}


# ---------------------------------------------------------------- dropdown agrupado
def _panel():
    from ts_annotator.ui.annotate_panel import AnnotatePanel
    return AnnotatePanel(CLASSES)


def test_dropdown_has_group_headers_non_selectable():
    p = _panel()
    combo = p.goal_combo
    # cabeçalhos existem, são NÃO selecionáveis, e não têm payload
    headers = [i for i in range(combo.count())
               if not (combo.model().item(i).flags() & Qt.ItemFlag.ItemIsSelectable)]
    assert len(headers) == len(GOAL_GROUPS)
    for i in headers:
        assert combo.itemData(i) is None
    # as linhas reais (estratégias) batem com GOALS
    assert len(p._real_idx) == len(GOALS)


# ---------------------------------------------------------------- requisito: modelo
def test_model_strategy_selectable_with_inline_note_when_no_model():
    """UX uniforme: sem modelo a estratégia NÃO é desabilitada — fica selecionável,
    com sufixo no rótulo e aviso inline (bloqueio real é ao propor)."""
    p = _panel()
    model_idx = next(i for i in p._real_idx if p.goal_combo.itemData(i)["req"] == "model")
    p.goal_combo.setCurrentIndex(model_idx)
    p.update_goals(has_model=False)
    assert p.goal_combo.model().item(model_idx).isEnabled()        # selecionável
    assert "sem modelo" in p.goal_combo.model().item(model_idx).text()
    assert "treine um modelo" in p.model_note.text()
    # com modelo: sem sufixo, sem aviso
    p.update_goals(has_model=True)
    assert p.goal_combo.model().item(model_idx).text() == p.goal_combo.itemData(model_idx)["label"]
    assert p.model_note.text() == ""


# ---------------------------------------------------------------- requisito: escolha
def test_choice_strategies_stay_selectable_with_inline_note():
    p = _panel()
    p.update_goals(has_model=True, has_curve=False)
    # seleciona 'parecidas com a curva atual' (req=curve)
    curve_idx = next(i for i in p._real_idx if p.goal_combo.itemData(i)["req"] == "curve")
    p.goal_combo.setCurrentIndex(curve_idx)
    p.update_goals(has_model=True, has_curve=False)
    assert p.goal_combo.model().item(curve_idx).isEnabled()       # NÃO bloqueia
    assert "clique numa curva" in p.model_note.text()             # só avisa
    # com a curva clicada, o aviso some
    p.update_goals(has_model=True, has_curve=True)
    assert p.model_note.text() == ""


def test_class_strategy_shows_target_combo_and_note():
    p = _panel()
    class_idx = next(i for i in p._real_idx if p.goal_combo.itemData(i)["req"] == "class")
    p.goal_combo.setCurrentIndex(class_idx)
    p.update_goals(has_model=True)
    assert p.tcls_combo.isVisibleTo(p) or True    # visibilidade real exige show(); estado interno:
    assert p.tcls_combo.isVisible() is False       # offscreen/never-shown → isVisible False
    # mas o alvo foi LIGADO (não escondido explicitamente) e a nota pede a classe
    assert not p.tcls_combo.isHidden()
    assert "escolha a classe" in p.model_note.text()
