"""AnnotatorController — the cross-cutting application logic behind AnnotatorWindow.

Everything that touches DATA (the active store, the time series, the model, the
selection engine, the background workers) lives here: click→curve→scores, labeling,
active-learning propose, review, model management, area/batch classification and
training. The window builds the widgets, wires their signals to these methods and
keeps widget aliases (``win.slist``, ``win.goal_combo``, …).

UI attribute resolution: the controller does not own widgets — any attribute not
found on the controller falls through ``__getattr__`` to the window (panels, canvas
items, widget aliases). Method bodies therefore read exactly as they did when they
lived on the window. State dicts (``state``/``tstate``/``pred_state``) are owned
here and shared with the window by reference.
"""

import glob
import logging
import os

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QRectF, Qt, QThread
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QApplication, QListWidgetItem

from ts_annotator.app.workers import ClassifyAllWorker, OverviewWorker, TrainWorker
from ts_annotator.core.prediction_raster import PredictionRaster
from ts_annotator.core.raster_source import RasterSource
from ts_annotator.core.vector_layer import VectorLayer
from ts_annotator.render import make_class_colorizer

log = logging.getLogger(__name__)


class AnnotatorController:
    def __init__(self, ctx, win):
        self.win = win
        self.ctx = ctx
        # --- bind services / data ---
        self.view = ctx.view
        self.curve_view = ctx.curve_view
        self.panel = ctx.panel
        self.datasets = ctx.datasets
        self.store = ctx.datasets[ctx.dataset_default]
        self.ts = ctx.ts
        self.fx = ctx.fx
        self.eng = ctx.eng
        self.sel = ctx.sel
        self.class_src = ctx.class_src
        self.TR = ctx.transform
        self.basemaps = ctx.basemaps
        self.vlayer_paths = ctx.vlayer_paths
        self.vlayer_items = {}
        self.classes = ctx.classes
        self.cls_colors = ctx.cls_colors
        self.class_super = dict(getattr(ctx, "class_super", {}) or {})    # classe -> super
        self.super_colors = dict(getattr(ctx, "super_colors", {}) or {})  # super -> cor
        self.model_dir = ctx.model_dir
        self.model_path = ctx.model_path
        self.pred_dir = ctx.pred_dir

        # --- shared state (a janela aponta pros MESMOS dicts) ---
        self.state = {"last": None, "size": 12, "pseed": 0, "grid": False,
                      "cells": set(), "last_src": "map", "sugg": [], "sugg_cur": None}
        self.tstate = {"net": None, "mu": None, "sd": None, "labs": None}  # modelo geral
        self.pred_state = {"alpha": 0.6}
        # similaridade = APENAS os pontos rotulados do dataset ativo (não uma
        # referência externa). Reconstrói lazy quando marcada suja; cacheia a
        # feature de cada ponto por (row,col) p/ o custo por rótulo ser ~1 ponto.
        self._sim_dirty = True
        self._sim_feat_cache = {}
        self.hover_pos = {"p": None}
        self._batch_running = False
        self._training = False        # exclusão mútua de GPU: treino x classificação
        self._infer_running = False   # anti-reentrância do classify-área (processEvents)
        self._undo = []               # pilha de undo: ("add"|"remove", payload) — só do dataset ativo
        self._labels_since_train = 0  # contador NEUTRO (o usuário decide quando re-treinar)
        # camada de pontos (painel de camadas): visibilidade/legenda/triagem/só-leitura
        self._pts_visible = True
        self._hidden_classes = set()
        self._disc_only = False
        self._susp_only = False
        self._pt_alpha = 1.0
        self._vlayer_colors = {}      # cor do CONTORNO por camada vetorial
        self._vlayer_fills = {}       # cor do PREENCHIMENTO (#aarrggbb) por camada

    def __getattr__(self, name):
        # widgets/painéis/itens do canvas vivem na janela (aliases incluídos)
        win = self.__dict__.get("win")
        if win is None:
            raise AttributeError(name)
        return getattr(win, name)

    # =====================================================================
    # Vector overlays / basemap overlay
    # =====================================================================
    def toggle_layer(self, name, on):
        if on and name not in self.vlayer_items:
            try:
                vl = VectorLayer(self.vlayer_paths[name], self.TR, self.class_src.crs)
                color = self._vlayer_colors.get(name, "#00e5ff")
                fill = self._vlayer_fills.get(name)
                brush = pg.mkBrush(QColor(fill)) if fill else None
                self.vlayer_items[name] = self.view.add_vector_layer(
                    vl, pen=pg.mkPen(color, width=1.5), brush=brush)
            except Exception as e:
                log.warning("layer %s n/d: %s", name, e)
                self.set_status(f"layer {name} indisponível: {e}")
                return
        if name in self.vlayer_items:
            self.vlayer_items[name].setVisible(on)

    def refresh_overlay(self, *_):
        self.canvas.refresh_overlay()

    def set_overlay(self, name):
        lbl = None if name == "(nenhum)" else name
        # sobrepor a camada que JÁ é o fundo é invisível — em vez de recusar calado,
        # troca o fundo pra a camada inicial (fenologia) e sobrepõe: é o que o
        # usuário quer ao clicar "sobrepor" na predição que está de fundo.
        if lbl and lbl == self.toolbar.combo.currentText():
            self.toolbar.combo.setCurrentText(self.ctx.init_label)
        # a PREDIÇÃO sobrepõe via camada TILED (pirâmide própria): renderiza em
        # QUALQUER zoom e atualiza ao vivo — fim do "só aparece no zoom". As demais
        # camadas seguem no overlay decimado do MapCanvas.
        if lbl == self.PRED_LABEL and lbl in self.basemaps:
            self.canvas.set_overlay("(nenhum)")      # some o overlay decimado
            self.pred_overlay.setVisible(False)       # e a pintura volátil antiga
            self.view.set_tiled_overlay(*self.basemaps[lbl], alpha=self.pred_state["alpha"])
            self._pred_overlay_tiled = True
            self.set_status(f"sobreposto: {lbl} (tiled — suave em qualquer zoom)")
            return
        if getattr(self, "_pred_overlay_tiled", False):   # saindo da predição tiled
            self.view.clear_tiled_overlay()
            self._pred_overlay_tiled = False
        self.canvas.set_overlay(name)
        if lbl:
            self.set_status(f"sobreposto: {lbl} (opacidade no chip RESULTADOS)")

    def set_overlay_alpha(self, v):
        self.canvas.set_overlay_alpha(v)
        # unifica os "dois sliders": o de RESULTADOS também rege a pintura volátil
        # (pred_overlay) E a camada tiled da predição. Todos leem o mesmo pred_state.
        self.pred_state["alpha"] = v / 100.0
        self.pred_overlay.setOpacity(self.pred_state["alpha"])
        self.view.set_tiled_overlay_alpha(self.pred_state["alpha"])

    # =====================================================================
    # Dataset selection
    # =====================================================================
    def on_dataset_change(self):
        name = self.toolbar.dataset_combo.currentText()
        if name not in self.datasets:
            return
        self.store = self.datasets[name]
        # o worker de treino guardava o store por referência — sem isto, treinar
        # após trocar de dataset treinaria no ANTIGO e gravaria métricas nos
        # pontos errados (ids colidem). Reaponta pro store ativo.
        if hasattr(self, "tworker"):
            self.tworker.store = self.store
        self._sim_dirty = True            # similaridade é do dataset ANTERIOR
        self._sim_feat_cache = {}         # (row,col) podem colidir entre datasets → zera
        self.state["last"] = None
        # sugestões/review são do dataset anterior — descarta (senão rotular uma
        # sugestão velha grava no store novo em coordenada de outro dataset).
        self.state["sugg"] = []
        self.state["sugg_cur"] = None
        if hasattr(self, "slist"):
            self.slist.clear()
            self.scount.setText("FILA — 0 sugestão(ões)")
        self.curve_view.clear()
        self.panel.clear()
        self.prop_marker.setData([])
        self._undo.clear()               # undo não pode atravessar datasets
        self._labels_since_train = 0
        self._refresh_counters()
        self.refresh_markers()
        self.refresh_review_list()
        log.info("dataset ativo: %s (%d pontos)", name, len(self.store))
        self.set_status(f"dataset ativo: {name} ({len(self.store)} pontos)")

    # =====================================================================
    # Grid tool (a geometria/itens vivem no MapCanvas)
    # =====================================================================
    def toggle_cell(self, row, col):
        self.canvas.toggle_cell(row, col)
        self.update_scope_labels()   # botões ecoam "células: N"

    # =====================================================================
    # Markers / snapping / selection / labeling
    # =====================================================================
    def on_class_color(self, cls, hexstr):
        """Edita a cor de uma classe: pontos + cards + LUT da predição + tabela +
        project.yaml (persistência). Origem = clique no swatch da legenda."""
        self.cls_colors[cls] = hexstr
        self.classes[cls] = hexstr
        if hasattr(self.panel, "classes"):
            self.panel.classes[cls] = hexstr
            if hasattr(self.panel, "refresh_colors"):
                self.panel.refresh_colors()
        self.refresh_markers()
        self.review.set_points(self.store.points, self._all_colors())
        pair = self.basemaps.get(self.PRED_LABEL)     # LUT da predição ativa
        labs = getattr(self, "_pred_labs", None)
        if pair and labs and cls in labs and hasattr(pair[1], "set_color"):
            pair[1].set_color(labs.index(cls), hexstr)
            self._refresh_pred_view()
        self._persist_class_color(cls, hexstr)
        self.set_status(f"cor de '{cls}' → {hexstr}")

    def on_pred_class_visible(self, cls, on):
        """Liga/desliga uma classe no raster de predição (transparente quando off)."""
        self._pred_hidden = getattr(self, "_pred_hidden", set())
        (self._pred_hidden.discard(cls) if on else self._pred_hidden.add(cls))
        pair = self.basemaps.get(self.PRED_LABEL)
        labs = getattr(self, "_pred_labs", None)
        if pair and labs and hasattr(pair[1], "set_hidden"):
            pair[1].set_hidden({labs.index(c) for c in self._pred_hidden if c in labs})
            self._refresh_pred_view()

    def _refresh_pred_view(self):
        """Repinta a predição após mudar cor/visibilidade (fundo tiled OU sobreposição).
        A predição sobreposta é a camada TILED (não o overlay decimado) → refresh dela."""
        if self.toolbar.combo.currentText() == self.PRED_LABEL:
            self.view.refresh()
        elif getattr(self, "_pred_overlay_tiled", False):
            self.view.refresh_tiled_overlay()

    def _persist_class_color(self, cls, hexstr):
        """Grava a nova cor no project.yaml (preserva o resto do arquivo via regex)."""
        import re
        pj = os.path.join(os.path.dirname(self.model_dir), "project.yaml")
        if not os.path.exists(pj):
            return
        try:
            txt = open(pj, encoding="utf-8").read()
            pat = re.compile(rf'(^\s*{re.escape(cls)}\s*:\s*)(["\']?)#[0-9A-Fa-f]{{3,8}}\2', re.M)
            new, n = pat.subn(rf'\g<1>"{hexstr}"', txt)
            if n:
                open(pj, "w", encoding="utf-8").write(new)
        except Exception as e:
            log.warning("persistir cor no project.yaml falhou: %s", e)

    def refresh_markers(self):
        from PyQt6.QtGui import QColor as _QC
        r = self.view.vb.viewRect()
        bbox = (r.x(), r.y(), r.x() + r.width(), r.y() + r.height())
        self.markers.setVisible(self._pts_visible)
        if self._pts_visible:
            pts = self.store.query(*bbox)
            if self._hidden_classes:
                pts = [p for p in pts if p["class"] not in self._hidden_classes]
            if self._disc_only:
                pts = [p for p in pts if p.get("_pred") and p["_pred"] != p.get("_ytrain", p["class"])]
            if self._susp_only:
                pts = [p for p in pts if p.get("_issue")]
            if len(pts) > 4000:
                pts = pts[:: max(1, len(pts) // 4000)]

            _ac = self._all_colors()   # inclui cores de super (pontos rotulados no super)
            def _brush(cls):
                c = _QC(_ac.get(cls, "#fff"))
                c.setAlphaF(self._pt_alpha)
                return pg.mkBrush(c)

            self.markers.setData([
                {"pos": (p["col"], p["row"]), "size": self.state["size"], "brush": _brush(p["class"])}
                for p in pts
            ])

    def _snap_radius(self):
        """Raio de snap em pixels; None se estiver afastado demais (não mira ponto)."""
        r = self.view.vb.viewRect()
        g = self.view.vb.geometry()
        nd = r.width() / max(g.width(), 1)
        return None if nd > 12 else max(2.0, nd * 8)

    def select_point(self, p):
        """Mostra EXATAMENTE este ponto (curva real + classe) + destaca na lista do Review."""
        curve = self.ts.read_curve(p["row"], p["col"])
        f = self.fx.extract(curve)
        self.curve_view.set_curve(curve)
        self.panel.set_scores(self._sim_score(f, exclude=p.get("id")), self.predict_point(curve),
                              False, self.fx.headline(f), p["class"], quality=p.get("_clean"))
        self.state["last"] = (p["row"], p["col"], curve)
        # origem = mapa: sem isto, rotular depois de vir de uma sugestão dispararia
        # advance_suggestion e descartaria uma sugestão que nunca foi rotulada
        self.state["last_src"] = "map"
        self.prop_marker.setData([{"pos": (p["col"], p["row"]), "size": 22,
                                   "pen": pg.mkPen("#ffffff", width=2), "brush": pg.mkBrush(255, 255, 255, 0)}])
        self.review.select_pid(p["id"])

    def predict_point(self, curve):
        """Probabilidade por classe do modelo geral (None se ainda não há modelo)."""
        net = self.tstate.get("net")
        if net is None:
            return None
        curve = np.asarray(curve, float)
        if not np.isfinite(curve).all():   # nodata parcial -> softmax NaN; igual ao batch, não prevê
            return None
        from ts_annotator.core import trainer as _tr
        X = _tr.build_X(curve.T[None], bands=getattr(self.ts, "band_names", None))
        pr = _tr.predict_proba(net, self.tstate["mu"], self.tstate["sd"], X)[0]
        labs = self.tstate["labs"]
        return {labs[i]: float(pr[i]) for i in range(len(labs))}

    def on_click(self, row, col):
        if self.state.get("pan"):
            return
        if self.state["grid"]:
            self.toggle_cell(row, col)
            return
        snap = self._snap_radius()
        existing = self.store.find_at(row, col, tol=snap) if snap else None
        if existing is not None:
            self.select_point(existing)
            return
        curve = self.ts.read_curve(row, col)
        # validade é da CURVA, não do modelo: sem modelo ainda dá pra rotular
        # (bootstrap). Curva com nodata parcial é rejeitada (não vira ponto/NaN).
        if not np.isfinite(curve).all():
            self.curve_view.clear()
            self.panel.clear()
            self.state["last"] = None
            self._mark_cursor(None, None)
            # sem isto o clique "não pega" em silêncio e o usuário acha que errou
            self.set_status("pixel sem dado completo (mês com nodata/nuvem) — não rotulável aqui")
            return
        f = self.fx.extract(curve)
        self.curve_view.set_curve(curve)
        self.panel.set_scores(self._sim_score(f), self.predict_point(curve), False, self.fx.headline(f), None)
        self.state["last"] = (row, col, curve)
        self.state["last_src"] = "map"   # ver select_point: não herdar "sugg" de antes
        self._mark_cursor(row, col)      # crosshair no pixel em foco (antes de rotular)

    def _mark_cursor(self, row, col):
        """Crosshair no pixel em foco. O ponto REAL só nasce ao escolher a classe;
        até lá isto mostra ONDE a curva exibida vem (some se row/col = None)."""
        if row is None or col is None:
            self.cursor_marker.setData([])
            return
        self.cursor_marker.setData([{"pos": (col, row), "symbol": "+", "size": 26,
                                     "pen": pg.mkPen("#ff1e56", width=3),
                                     "brush": pg.mkBrush(0, 0, 0, 0)}])

    def on_label(self, cls):
        if self.state["last"] is None:
            return
        row, col, curve = self.state["last"]
        old = self.store.find_at(row, col)   # dict antigo fica intacto p/ undo (update troca o objeto)
        _, updated = self.store.add_or_update(row, col, cls, curve)
        self._push_undo(("add", (row, col, old)))
        self._sim_dirty = True               # o novo rótulo entra na similaridade
        self._labels_since_train += 1
        self._refresh_counters()
        self._mark_cursor(None, None)        # virou ponto real — o crosshair sai de cena
        msg = f"{'atualizado' if updated else 'salvo'}: {cls} @ r{row} c{col}  (#{len(self.store)} no dataset)"
        log.info(msg)
        self.set_status(msg)
        self.refresh_markers()
        self.refresh_review_list()
        if self.state.get("last_src") == "sugg":
            self.advance_suggestion()
            return
        self.curve_view.set_curve(curve)
        f = self.fx.extract(curve)
        lp = self.store.find_at(row, col)   # leave-one-out: não pontuar contra si mesmo
        self.panel.set_scores(self._sim_score(f, exclude=lp.get("id") if lp else None),
                              self.predict_point(curve), False, self.fx.headline(f), cls)

    def on_class_added(self, name):
        """Nova classe criada no painel: propaga a cor pro mapa de cores da janela.

        Sem isto, o sinal ficava solto — a classe existia só no painel, e um ponto
        rotulado com ela saía sem cor (branco) e fora da legenda/similaridade. A cor
        é a que o painel escolheu (PALETTE); espelhamos em cls_colors e classes.
        Exemplares p/ similaridade só existem após rotular + retreinar (inerente).
        """
        color = self.panel.classes.get(name, "#cccccc")
        self.cls_colors.setdefault(name, color)
        self.classes.setdefault(name, color)
        self.refresh_markers()
        self.refresh_review_list()
        log.info("nova classe: %s (%s)", name, color)
        self.set_status(f"nova classe: {name}")

    # ---- superclasses (hierarquia) ----
    def _super_members(self):
        """{super: [classes finas...]} na ordem das classes (default: classe é seu super)."""
        out = {}
        for cls in self.classes:
            out.setdefault(self.class_super.get(cls, cls), []).append(cls)
        return out

    def _all_colors(self):
        """Cores finas + de superclasse — o pred pode ter labels de super (treino colapsado)."""
        return {**self.super_colors, **self.cls_colors}

    # =====================================================================
    # Gerenciar classes (renomear / mesclar / remover) — propaga por pontos,
    # painel, legenda, filtro da tabela, similaridade e project.yaml. Modelo/
    # predição só refletem após retreinar (as labs são fixadas no treino).
    # =====================================================================
    def open_class_manager(self):
        from ts_annotator.app.config import NON_TRAINING_CLASSES
        from ts_annotator.ui.class_manager import ClassManagerDialog
        ClassManagerDialog(
            get_state=lambda: (dict(self.classes), self.store.class_counts()),
            do_rename=self.rename_class, do_remove=self.remove_class,
            non_training=NON_TRAINING_CLASSES, parent=self.win).exec()
        # o gerenciador pode ter mexido no dataset ATIVO enquanto um treino/predição
        # roda; nada a fazer aqui além do que _after_class_change já sincronizou.

    def rename_class(self, old, new):
        new = (new or "").strip()
        if not new or new == old:
            return False, "sem mudança"
        from ts_annotator.app.config import NON_TRAINING_CLASSES
        if old in NON_TRAINING_CLASSES:
            return False, f"'{old}' é classe especial (fora do treino) — não dá pra renomear aqui"
        is_merge = new in self.classes and new != old
        n = self.store.relabel(old, new)                 # reescreve os pontos do dataset ativo
        if is_merge:
            self.classes.pop(old, None); self.cls_colors.pop(old, None)
            if hasattr(self.panel, "classes"):
                self.panel.classes.pop(old, None)
        else:
            col = self.classes.pop(old, "#cccccc"); self.classes[new] = col
            self.cls_colors[new] = self.cls_colors.pop(old, col)
            if hasattr(self.panel, "classes"):
                self.panel.classes[new] = self.panel.classes.pop(old, col)
        self._after_class_change()
        verb = "mesclada" if is_merge else "renomeada"
        msg = f"classe {verb}: '{old}' → '{new}' ({n} pontos)"
        self.set_status(msg + (self._retrain_hint() if n else ""))
        return True, msg

    def remove_class(self, cls, dest):
        from ts_annotator.app.config import NON_TRAINING_CLASSES
        if cls in NON_TRAINING_CLASSES:
            return False, f"'{cls}' é classe especial — não remover"
        n = self.store.relabel(cls, dest) if dest else 0   # move os pontos antes de sumir
        self.classes.pop(cls, None); self.cls_colors.pop(cls, None)
        if hasattr(self.panel, "classes"):
            self.panel.classes.pop(cls, None)
        self._after_class_change()
        msg = (f"classe removida: '{cls}' — {n} pontos → '{dest}'" if dest
               else f"classe vazia removida: '{cls}'")
        self.set_status(msg + (self._retrain_hint() if n else ""))
        return True, msg

    def _retrain_hint(self):
        return "  ·  retreine pra atualizar a predição"

    def _after_class_change(self):
        """Sincroniza TODA a UI após uma operação de classe."""
        if hasattr(self.panel, "set_classes"):
            self.panel.set_classes(self.classes)      # cards de rotular
        self.toolbar.pop.rebuild_classes(self.classes)  # zera a legenda…
        self._refresh_counters()                       # …e repopula (update_points)
        self.review.rebuild_class_filter(self.classes)  # filtro da tabela
        self.refresh_review_list()                     # cores/pontos da tabela
        self.refresh_markers()                         # marcadores no mapa
        self._sim_dirty = True                         # similaridade reconstrói
        self._undo.clear()                             # relabel em massa não é desfazível
        self._persist_classes()                        # paleta no project.yaml

    def _persist_classes(self):
        """Reescreve o bloco `classes:` do project.yaml com a paleta atual
        (preserva comentários indentados do bloco e o resto do arquivo)."""
        pj = os.path.join(os.path.dirname(self.model_dir), "project.yaml")
        if not os.path.exists(pj):
            return
        try:
            lines = open(pj, encoding="utf-8").read().splitlines()
            out, i, n = [], 0, len(lines)
            while i < n:
                if lines[i].startswith("classes:"):
                    out.append("classes:"); i += 1
                    comments = []
                    while i < n and lines[i].startswith((" ", "\t")):   # entradas + comentários indentados
                        s = lines[i].strip()
                        if s.startswith("#"):
                            comments.append("  " + s)
                        i += 1
                    out.extend(comments)
                    for name, color in self.classes.items():
                        out.append(f'  {name}: "{color}"')
                else:
                    out.append(lines[i]); i += 1
            open(pj, "w", encoding="utf-8").write("\n".join(out) + "\n")
        except Exception as e:
            log.warning("persistir classes no project.yaml falhou: %s", e)

    def on_remove(self):
        if self.state["last"] is None:
            return
        row, col, _ = self.state["last"]
        removed = self.store.find_at(row, col)
        if self.store.remove_at(row, col):
            self._push_undo(("remove", removed))
            self._sim_dirty = True           # sai da similaridade
            self._refresh_counters()
            msg = f"removido: {removed['class']} @ r{row} c{col}  ({len(self.store)} no dataset)"
            log.info(msg)
            self.set_status(msg)
            self.refresh_markers()
            self.refresh_review_list()

    # =====================================================================
    # Undo (Ctrl+Z) — últimas operações de rótulo do dataset ATIVO
    # =====================================================================
    def _push_undo(self, op):
        self._undo.append(op)
        del self._undo[:-20]

    def undo_last(self):
        if not self._undo:
            self.set_status("nada pra desfazer")
            return
        kind, payload = self._undo.pop()
        if kind == "add":
            row, col, old = payload
            if old is None:                      # era ponto NOVO → some
                self.store.remove_at(row, col)
                self.set_status(f"desfeito: rótulo removido @ r{row} c{col}")
            else:                                # era UPDATE → volta o dict antigo intacto
                self.store.restore(old)
                self.set_status(f"desfeito: voltou a ser {old['class']} @ r{row} c{col}")
            self._labels_since_train = max(0, self._labels_since_train - 1)
        else:                                    # remoção → re-insere como era (id/metadados)
            self.store.restore(payload)
            self.set_status(f"desfeito: {payload['class']} @ r{payload['row']} c{payload['col']} de volta")
        self._sim_dirty = True                   # o desfazer muda o conjunto rotulado
        self._refresh_counters()
        self.refresh_markers()
        self.refresh_review_list()

    def _refresh_counters(self):
        """Contadores neutros: rótulos desde o treino + pontos com curva (treino)
        + legenda/contagens da camada de pontos no painel de camadas."""
        from collections import Counter
        self.annotate.set_since_train(self._labels_since_train)
        n = sum(1 for p in self.store.points if "curve" in p)
        self.train.set_points_info(n)
        counts = Counter(p["class"] for p in self.store.points)
        active = self.toolbar.dataset_combo.currentText()
        self.toolbar.pop.update_points(active, len(self.store), counts)

    # ---- camada de pontos (sinais do painel de camadas) ----
    def on_points_visible(self, on):
        self._pts_visible = on
        self.refresh_markers()

    def on_class_visible(self, cls, on):
        (self._hidden_classes.discard(cls) if on else self._hidden_classes.add(cls))
        self.refresh_markers()

    def on_triage(self, disc, susp):
        self._disc_only, self._susp_only = disc, susp
        self.refresh_markers()

    def on_pts_alpha(self, v):
        self._pt_alpha = v / 100.0
        self.refresh_markers()

    def set_layer_color(self, name, hexcolor):
        self._vlayer_colors[name] = hexcolor
        if name in self.vlayer_items:
            self.vlayer_items[name].setPen(pg.mkPen(hexcolor, width=1.5))

    def set_layer_fill(self, name, hexargb):
        """cor do preenchimento (#aarrggbb, com alfa). Vazio/None = sem fill."""
        self._vlayer_fills[name] = hexargb
        if name in self.vlayer_items:
            brush = pg.mkBrush(QColor(hexargb)) if hexargb else pg.mkBrush(None)
            self.vlayer_items[name].setBrush(brush)
            self.view.refresh_overlays()   # reconstrói o path do fill agora, não só no próximo pan

    def navigate(self, row, col, curve, scores, existing_cls, cyc, src="map", quality=None):
        # só re-enquadra se o ponto está FORA da vista (com folga de 5%) — re-zoomar
        # a cada sugestão/item do Review destruía o zoom escolhido pelo usuário
        r = self.view.vb.viewRect()
        mx, my = r.width() * 0.05, r.height() * 0.05
        if not (r.x() + mx <= col <= r.x() + r.width() - mx
                and r.y() + my <= row <= r.y() + r.height() - my):
            self.view.vb.setRange(xRange=(col - 750, col + 750), yRange=(row - 750, row + 750), padding=0)
        self.curve_view.set_curve(curve)
        self.panel.set_scores(scores, self.predict_point(curve), self.eng.is_novel(scores),
                              cyc, existing_cls, quality=quality)
        self.state["last"] = (row, col, curve)
        self.state["last_src"] = src
        self._mark_cursor(None, None)   # foco aqui é do prop_marker; sem crosshair duplo
        self.prop_marker.setData([{"pos": (col, row), "size": 24,
                                   "pen": pg.mkPen("#ffffff", width=2), "brush": pg.mkBrush(255, 255, 255, 0)}])

    # =====================================================================
    # Propose (active learning)
    # =====================================================================
    def _propose_area(self, area=None):
        """Bounds da área de trabalho; None = escopo células SEM célula selecionada
        (recusar com mensagem — cair calado no viewport classificaria/proporia no
        lugar errado com o botão dizendo "células: 0"). `area` explícito permite a
        descoberta escolher a própria área (não o chip do mapa)."""
        area = area or self.area_combo.currentData()
        if area == "grids":
            return self.canvas.cells_bounds() if self.state["cells"] else None
        if area == "all":
            return 0, 0, self.class_src.width, self.class_src.height
        if area == "rect":
            return self.canvas.rect_bounds()
        r = self.view.vb.viewRect()
        return r.x(), r.y(), r.x() + r.width(), r.y() + r.height()

    def _cand_text(self, c, metric):
        sim = c["scores"]
        pred = c.get("pred")
        st = max(sim, key=sim.get) if sim else "?"
        txt = f"r{c['row']} c{c['col']}   ~{st} {sim.get(st, 0):.0%}"
        if pred:
            pt = max(pred, key=pred.get)
            txt += f"   modelo: {pt} {pred.get(pt, 0):.0%}"
        if metric == "disagreement" and "_m" in c:
            txt += f"   Δ {c['_m']:.0%}"
        elif metric == "similar_current" and "_m" in c:
            txt += f"   dist {-c['_m']:.2f}"   # menor = mais parecida com a curva atual
        return txt

    def fill_suggestions(self):
        metric = self.goal_combo.currentData()["metric"]
        self.slist.blockSignals(True)
        self.slist.clear()
        for i, c in enumerate(self.state["sugg"]):
            it = QListWidgetItem(self._cand_text(c, metric))
            it.setData(Qt.ItemDataRole.UserRole, i)
            st = max(c["scores"], key=c["scores"].get) if c["scores"] else None
            it.setForeground(QColor(self.cls_colors.get(st, "#ffffff")))
            self.slist.addItem(it)
        self.slist.blockSignals(False)
        self.scount.setText(f"FILA — {len(self.state['sugg'])} sugestão(ões)")

    def _ensure_similarity(self):
        """(Re)constrói a similaridade a partir dos pontos ROTULADOS do dataset
        ativo — só quando marcada suja. A feature de cada ponto é cacheada por
        (row,col), então adicionar/remover 1 rótulo só extrai esse 1 ponto."""
        if not self._sim_dirty:
            return
        from ts_annotator.app.config import NON_TRAINING_CLASSES
        cache = self._sim_feat_cache
        items, seen = [], set()
        for p in self.store.points:
            if "curve" not in p:
                continue                     # sem curva não há feature (ex.: ponto semeado cru)
            if p.get("class") in NON_TRAINING_CLASSES:
                continue                     # "don't know" fora da similaridade (poluiria as sugestões)
            key = (p["row"], p["col"])
            seen.add(key)
            f = cache.get(key)
            if f is None:
                f = self.fx.extract(np.asarray(p["curve"], float))
                if not np.isfinite(f).all():
                    continue                 # curva com nodata envenenaria a distância
                cache[key] = f
            items.append((f, p["class"], p.get("id")))
        for k in list(cache):                # poda pontos que saíram do dataset
            if k not in seen:
                del cache[k]
        if len(items) > 3000:
            self.set_status("preparando similaridade dos rotulados…")
            QApplication.processEvents()
        self.eng.load(items)                 # sel.sim É o mesmo objeto → propor também atualiza
        self._sim_dirty = False

    def _sim_score(self, features, exclude=None):
        self._ensure_similarity()
        return self.eng.score(features, exclude=exclude)

    def do_propose(self):
        area = self._propose_area()
        if area is None:
            self.scount.setText("selecione células: ligue a grade e clique nas células")
            return
        self.state["pseed"] += 1
        self._ensure_similarity()   # sel.sim = eng; candidatos pontuam vs os rotulados
        x0, y0, x1, y1 = area
        data = self.goal_combo.currentData()
        metric, order, req = data["metric"], data["order"], data["req"]
        # bloqueio no MOMENTO de propor (botão sempre ativo; requisito checado aqui)
        if req == "model" and self.tstate.get("net") is None:
            self.scount.setText("esta estratégia precisa de um modelo treinado (aba Treinar)")
            return
        target = self.tcls_combo.currentText() if metric == "class" else None
        excl = {(p["row"], p["col"]) for p in self.store.points}
        k = self.count_spin.value()
        query_f = None
        if metric == "similar_current":     # busca-por-exemplo: a curva EM FOCO é o protótipo
            if self.state["last"] is None:
                self.scount.setText("clique numa curva no mapa primeiro (p/ buscar parecidas)")
                return
            query_f = self.fx.extract(np.asarray(self.state["last"][2], float))
        self.pb.setEnabled(False)
        self.pb.setText("buscando…")
        self.propose_bar.setValue(0)
        self.propose_bar.setVisible(True)

        def _tick(i, n):
            self.propose_bar.setValue(int(100 * i / max(n, 1)))
            if i % 8 == 0 or i == n:          # não pumpar a cada pixel
                QApplication.processEvents()

        st = self.store   # o combo de dataset segue vivo durante o processEvents do _tick
        try:
            if metric == "similar_current":
                cands = self.sel.similar_to(x0, y0, x1, y1, query_f, k=k,
                                            seed=self.state["pseed"], exclude=excl,
                                            inside=self._scope_inside())
                for c in cands:              # scores p/ cor/rótulo na lista de sugestões
                    c["scores"] = self._sim_score(np.asarray(c["features"], float))
            else:
                cands = self.sel.propose_many(x0, y0, x1, y1, k=k, metric=metric, order=order,
                                              target=target, seed=self.state["pseed"], exclude=excl,
                                              predict_fn=self.predict_point,
                                              progress=_tick, inside=self._scope_inside())
        finally:
            self.propose_bar.setVisible(False)
            self.pb.setEnabled(True)
            self.pb.setText("  Propor")   # reverte o "buscando…"
            self.update_scope_labels()
        if st is not self.store:
            return   # dataset trocou no meio da busca — sugestões seriam do antigo
        self.state["sugg"] = cands
        self.fill_suggestions()
        if cands:
            self.slist.setCurrentRow(0)
        else:
            self.scount.setText("FILA — 0 (nenhuma na área; pixels sem dado completo?)")

    # =====================================================================
    # Descoberta de padrões (não-supervisionada: os padrões RECORRENTES da área)
    # =====================================================================
    def do_discover(self, scope="viewport", step=10, k=6):
        area = self._propose_area(scope)
        if area is None:
            msg = ("desenhe um retângulo no mapa (ferramenta ⬚) antes" if scope == "rect"
                   else "selecione células: ligue a grade e clique nelas" if scope == "grids"
                   else "área inválida")
            self.patterns_win.discover_info.setText("⚠ " + msg)
            self.set_status(msg)
            return
        try:
            from ts_annotator.core.cluster import ClusterEngine
        except Exception:
            self.set_status("descoberta precisa de scikit-learn (pip install .[train])")
            return
        if not hasattr(self, "_cluster_eng"):
            self._cluster_eng = ClusterEngine(self.sel)
        self._disc_area = area          # congela a área p/ propor-deste consistente
        self._disc_scope = scope        # e o escopo (p/ o filtro de células no apply)
        x0, y0, x1, y1 = area
        self.patterns_win.set_busy(True)
        self.set_status("descobrindo padrões na área…")
        QApplication.processEvents()
        try:
            root = self._cluster_eng.discover(
                x0, y0, x1, y1, k=k, step=step, seed=self.state["pseed"] + 1,
                inside=self._scope_inside(scope))
        finally:
            self.patterns_win.set_busy(False)
        info = getattr(self._cluster_eng, "info", {}) or {}
        # feedback de DENSIDADE real: nº de amostras + 1-a-cada-quantos-px (o teto
        # pode ter aumentado o passo efetivo — mostra o que de fato foi usado)
        eff = info.get("eff_step", step)
        note = f"{info.get('n_valid', 0)} amostras · 1 a cada {eff} px"
        if info.get("capped"):
            note += " · teto atingido (passo aumentado; use uma área menor p/ mais densidade)"
        elif eff != step:
            note += f" · passo ajustado de {step} pro teto de segurança"
        self.patterns_win.discover_info.setText(note)
        if root is None:
            self.patterns_win.show_node(None, 0)
            self.set_status("nenhum padrão (área sem pixels válidos?)")
            return
        self._cur_node = root
        self.patterns_win.show_node(root, 0)
        self.patterns_win.show_panel()
        self.set_status(f"{len(root.children)} padrões dominantes ({self._scope_label()}) — {note}")

    def _cur_children(self):
        n = getattr(self, "_cur_node", None)
        return (n.children or []) if n else []

    def discover_split(self, k):
        """Redivide o nó ATUAL em k (o breadcrumb não muda)."""
        n = getattr(self, "_cur_node", None)
        if n is None:
            return
        self._cluster_eng.split(n, k)
        self.patterns_win.show_node(n, 0)

    def discover_enter(self, i, k):
        """Entra no filho i (drill-down) e o subdivide em k."""
        kids = self._cur_children()
        if i >= len(kids):
            return
        node = kids[i]
        self._cluster_eng.split(node, k)
        self._cur_node = node
        self.patterns_win.show_node(node, 0)
        self.set_status(f"padrão {node.path} → {len(node.children)} sub-padrões")

    def apply_patterns(self):
        """MAPA de clusters CONTÍGUO do nível atual: cada célula da grade atribuída
        ao padrão mais próximo, pintada como raster (não pontos). Cores = cards."""
        from ts_annotator.render import CLUSTER_PALETTE
        n = getattr(self, "_cur_node", None)
        if n is None or not n.children:
            return
        area = getattr(self, "_disc_area", None) or self._propose_area()
        if area is None:
            return
        step = self.patterns_win.density_spin.value()
        self.set_status("pintando o mapa de clusters…")
        QApplication.processEvents()
        x0, y0, x1, y1 = area
        labels, rect = self._cluster_eng.segment_area(
            x0, y0, x1, y1, n, step,
            inside=self._scope_inside(getattr(self, "_disc_scope", None)))
        if labels is None:
            self.set_status("área sem dado p/ pintar")
            return
        self.canvas.set_cluster_raster(labels, rect, CLUSTER_PALETTE)
        self.set_status(f"mapa de {len(n.children)} padrões pintado (nível {n.path}) — "
                        "cores = cards; 'limpar do mapa' remove")

    def clear_patterns(self):
        self.canvas.clear_clusters()

    def discover_up(self):
        n = getattr(self, "_cur_node", None)
        if n is not None and n.parent is not None:
            self._cur_node = n.parent
            self.patterns_win.show_node(n.parent, 0)

    def discover_root(self):
        root = getattr(self._cluster_eng, "root", None) if hasattr(self, "_cluster_eng") else None
        if root is not None:
            self._cur_node = root
            self.patterns_win.show_node(root, 0)

    def go_to_pattern(self, i):
        kids = self._cur_children()
        if i < len(kids):
            m = kids[i].medoid
            self.zoom_to_point_rc(m["row"], m["col"], m["curve"])

    def propose_from_pattern(self, i):
        kids = self._cur_children()
        if i >= len(kids):
            return
        area = getattr(self, "_disc_area", None) or self._propose_area()
        if area is None:
            return
        x0, y0, x1, y1 = area
        excl = {(p["row"], p["col"]) for p in self.store.points}
        cands = self._cluster_eng.similar_to(
            x0, y0, x1, y1, kids[i].medoid["features"],
            k=self.count_spin.value(), seed=self.state["pseed"], inside=self._scope_inside(),
            exclude=excl)
        self.state["sugg"] = cands
        self.fill_suggestions()
        if cands:
            self.slist.setCurrentRow(0)
        self.set_status(f"{len(cands)} sugestões parecidas com o Padrão {i + 1} "
                        "(aba Rotular → lista de sugestões)")

    def zoom_to_point_rc(self, r, c, curve):
        """Zoom + carrega uma curva arbitrária (medóide não é ponto do store)."""
        self.view.vb.setRange(xRange=(c - 120, c + 120), yRange=(r - 120, r + 120), padding=0)
        curve = np.asarray(curve, float)
        f = self.fx.extract(curve)
        ex = self.store.find_at(r, c)
        self.navigate(r, c, curve, self._sim_score(f, exclude=ex.get("id") if ex else None),
                      ex["class"] if ex else None, self.fx.headline(f))

    def on_sugg_select(self, cur, _prev=None):
        if cur is None:
            return
        i = cur.data(Qt.ItemDataRole.UserRole)
        if i is None or i >= len(self.state["sugg"]):
            return
        c = self.state["sugg"][i]
        self.state["sugg_cur"] = i
        ex = self.store.find_at(c["row"], c["col"])
        # relê a curva EXATA (full-res) no pixel — o candidato pode ter vindo de uma
        # grade decimada (proposta sobre área grande). Assim rotula-se o pixel real,
        # não a célula grossa. Cai na curva do candidato se o pixel exato for nodata.
        curve = self.ts.read_curve(c["row"], c["col"])
        if not np.isfinite(curve).all():
            curve = np.asarray(c["curve"], float)
        f = self.fx.extract(curve)
        self.navigate(c["row"], c["col"], curve, self._sim_score(f),
                      ex["class"] if ex else None, self.fx.headline(f), src="sugg")

    def advance_suggestion(self):
        """Após rotular uma sugestão: remove da lista e seleciona a próxima."""
        i = self.state.get("sugg_cur")
        if i is None or i >= len(self.state["sugg"]):
            return
        del self.state["sugg"][i]
        self.fill_suggestions()
        if self.state["sugg"]:
            self.slist.setCurrentRow(min(i, len(self.state["sugg"]) - 1))
        else:
            self.scount.setText("FILA — 0 (proponha mais)")

    def skip_suggestion(self):
        i = self.slist.currentRow()
        if 0 <= i < self.slist.count() - 1:
            self.slist.setCurrentRow(i + 1)

    # =====================================================================
    # Hover
    # =====================================================================
    def on_hover(self, scene_pos):
        self.hover_pos["p"] = scene_pos

    def _proc_hover(self):
        sp = self.hover_pos["p"]
        if self.state["grid"]:
            self.hover_marker.setData([])
            return
        if sp is None or sp is self.hover_pos.get("last"):
            return
        self.hover_pos["last"] = sp
        p = self.view.vb.mapSceneToView(sp)
        snap = self._snap_radius()
        ex = self.store.find_at(int(p.y()), int(p.x()), tol=snap) if snap else None
        self.hover_marker.setData(
            [{"pos": (ex["col"], ex["row"]), "size": 26,
              "pen": pg.mkPen("#ffd54f", width=3), "brush": pg.mkBrush(0, 0, 0, 0)}]
            if ex is not None else []
        )

    # =====================================================================
    # Toolbar handlers
    # =====================================================================
    def _on_basemap(self, name):
        if name in self.basemaps:
            self.view.set_source(*self.basemaps[name])
            if self.overlay_state["label"] == name:   # fundo == sobreposição → limpa
                self.toolbar.ov_combo.setCurrentText("(nenhum)")
                self.set_status(f"'{name}' virou o fundo — sobreposição limpa")
            if name == self.PRED_LABEL and getattr(self, "_pred_overlay_tiled", False):
                self.view.clear_tiled_overlay()       # predição virou fundo: sem overlay duplo
                self._pred_overlay_tiled = False
                self.toolbar.ov_combo.blockSignals(True)
                self.toolbar.ov_combo.setCurrentText("(nenhum)")
                self.toolbar.ov_combo.blockSignals(False)

    def set_grid(self, on):
        self.state["grid"] = on
        self.canvas.set_grid_visible(on)
        if on and self.draw_btn.isChecked():
            self.draw_btn.setChecked(False)

    def set_cols(self, n):
        self.canvas.set_cols(n)          # limpa as células selecionadas
        self.update_scope_labels()       # "células: 0" no eco, não o número velho

    def set_size(self, v):
        self.state["size"] = v
        self.refresh_markers()

    def set_draw(self, on):
        self.state["draw"] = on
        self.canvas.set_draw(on)
        if on:
            self.toolbar.set_grid_checked(False)

    # =====================================================================
    # Área de trabalho (escopo compartilhado: Annotate propõe nela, Classify
    # classifica nela — o verbo vem da aba, a área é uma só)
    # =====================================================================
    def on_scope_change(self, data):
        self.rect_roi.setVisible(data == "rect")
        if data == "grids":
            if not self.state["grid"]:
                self.toolbar.set_grid_checked(True)    # células exigem o grid visível
        elif self.state["grid"]:
            self.toolbar.set_grid_checked(False)       # saiu do modo células → some a grade (e os realces)
        if data == "rect":
            # "desenhar retângulo" JÁ entra no modo desenhar (o rótulo promete isso):
            # arraste no mapa cria o retângulo. Antes só mostrava o ROI e "nada acontecia".
            if not self.toolbar.draw_btn.isChecked():
                self.toolbar.draw_btn.setChecked(True)   # dispara on_draw_toggled → set_draw(True)
        elif self.toolbar.draw_btn.isChecked():
            self.toolbar.draw_btn.setChecked(False)
        self.update_scope_labels()

    def on_draw_toggled(self, on):
        self.set_draw(on)
        if on and self.area_combo.currentData() != "rect":
            self.toolbar.set_scope("rect")         # desenhar implica escopo retângulo

    def _scope_label(self):
        d = self.area_combo.currentData()
        if d == "grids":
            return f"células: {len(self.state['cells'])}"
        return {"viewport": "vista", "rect": "retângulo", "all": "imagem toda"}.get(d, str(d))

    def update_scope_labels(self):
        """Ecoa o escopo no chip ÁREA e no botão classificar — 'vai rodar onde?'.
        (O propor não ecoa: o chip ÁREA já mostra o escopo e o botão fica curto.)"""
        lb = self._scope_label()
        self.classify.set_scope_text(lb)
        self.toolbar.pop.sync_scope(self.area_combo.currentData(), len(self.state["cells"]))

    def _scope_inside(self, area=None):
        """Filtro de UNIÃO p/ células: propor dentro das células selecionadas,
        não no bounding-box delas (células desconexas não cobrem o meio)."""
        area = area or self.area_combo.currentData()
        if area == "grids" and self.state["cells"]:
            cp = self.canvas.cell_px
            cells = set(self.state["cells"])
            return lambda r, c: (int(r // cp), int(c // cp)) in cells
        return None

    def _scope_tiles(self, x0, y0, x1, y1):
        """Tiles do job de lote que intersectam o escopo (células = união)."""
        T = 1024   # = PredictionRaster.tile default (o worker honra o sidecar)
        W, H = self.class_src.width, self.class_src.height
        nx, ny = -(-W // T), -(-H // T)

        def box(bx0, by0, bx1, by1):
            tj0 = max(0, int(bx0) // T); ti0 = max(0, int(by0) // T)
            tj1 = min(nx, -(-int(min(bx1, W)) // T)); ti1 = min(ny, -(-int(min(by1, H)) // T))
            return {(ti, tj) for ti in range(ti0, ti1) for tj in range(tj0, tj1)}

        if self.area_combo.currentData() == "grids" and self.state["cells"]:
            cp = self.canvas.cell_px
            tiles = set()
            for (cr, cc) in self.state["cells"]:
                tiles |= box(cc * cp, cr * cp, (cc + 1) * cp, (cr + 1) * cp)
            return sorted(tiles)
        return sorted(box(x0, y0, x1, y1))

    def update_goals(self):
        """Liga/desliga estratégias por requisito (modelo carregado + curva em foco)."""
        self.annotate.update_goals(self.tstate.get("net") is not None,
                                   has_curve=self.state.get("last") is not None)

    # =====================================================================
    # Review
    # =====================================================================
    def refresh_review_list(self):
        # a tabela é model/view sobre a lista VIVA do store — sem cópia, sem teto
        self.review.set_points(self.store.points, self._all_colors())

    def on_review_point(self, pid):
        p = next((q for q in self.store.points if q["id"] == pid), None)
        if p is None:
            return
        curve = self.ts.read_curve(p["row"], p["col"])
        f = self.fx.extract(curve)
        self.navigate(p["row"], p["col"], curve, self._sim_score(f, exclude=p.get("id")),
                      p["class"], self.fx.headline(f), quality=p.get("_clean"))

    def zoom_to_point(self, pid):
        """Duplo-clique na tabela: enquadra APERTADO no ponto (± ~120 px)."""
        p = next((q for q in self.store.points if q["id"] == pid), None)
        if p is None:
            return
        r, c = p["row"], p["col"]
        self.view.vb.setRange(xRange=(c - 120, c + 120), yRange=(r - 120, r + 120), padding=0)
        self.on_review_point(pid)

    # =====================================================================
    # Model management + prediction
    # =====================================================================
    def list_model_files(self):
        """[(rótulo, caminho)] — versões (mais nova primeiro, com métricas no rótulo)
        + .pt soltos na raiz de models/ (legado)."""
        from ts_annotator.core.version_store import VersionStore
        out = []
        for v in reversed(VersionStore(self.model_dir).list_versions()):
            m = v["meta"] or {}
            met = m.get("metrics") or {}
            bits = [v["name"]]
            if met.get("bacc") is not None:
                bits.append(f"bacc {met['bacc']:.2f}")
            if m.get("n_points"):
                bits.append(f"{m['n_points']} pts")
            if m.get("created"):
                bits.append(str(m["created"])[:10])
            out.append(("  ·  ".join(bits), v["path"]))
        for p in sorted(glob.glob(os.path.join(self.model_dir, "*.pt"))):
            out.append((os.path.basename(p), p))
        return out

    def load_active_model(self, path):
        if not path or not os.path.exists(path):
            self.tstate.update(net=None, mu=None, sd=None, labs=None)
            log.info("modelo ativo: (nenhum)")
        else:
            try:
                from ts_annotator.core import trainer as _tr
                from ts_annotator.core.version_store import model_stem
                n0, m0, s0, l0 = _tr.load_model(path)
                self.tstate.update(net=n0, mu=m0, sd=s0, labs=l0)
                log.info("modelo ativo: %s (%d classes)", model_stem(path), len(l0))
                self.set_status(f"modelo ativo: {model_stem(path)} ({len(l0)} classes)")
            except Exception as e:
                log.error("erro ao carregar modelo: %s", e)
                self.tstate.update(net=None, mu=None, sd=None, labs=None)
        self.update_goals()
        self._refresh_pred_basemap()
        from ts_annotator.core.version_store import model_stem
        self.classify.set_model(model_stem(path) if path and os.path.exists(path) else "(nenhum)")

    def _refresh_pred_basemap(self):
        """Aponta a camada 'Predição (modelo)' pro pred_<modelo>.tif do modelo ativo.

        Reabre a fonte (o raster cresce conforme classificamos) e usa o SIDECAR pra
        montar o colorizer — índice→classe→cor é do modelo, não da legenda global.
        Sem modelo/raster: a camada some do dict (selecionar vira no-op) e, se estava
        em uso como base, volta pra camada de observação inicial.
        """
        lbl = self.PRED_LABEL
        old = self.basemaps.pop(lbl, None)
        if old is not None:
            try:
                old[0].close()
            except Exception:
                pass
        mp = self.toolbar.model_combo.currentData()
        self.classify.set_coverage("—")
        self.classify.set_target("—")
        pred_info, pred_ok = "(sem modelo)", False
        if mp:
            from ts_annotator.core.version_store import model_stem
            stem = model_stem(mp)
            pred = PredictionRaster(self.pred_dir, mp)
            meta = pred.load_meta()
            self.classify.set_target(os.path.basename(pred.tif))
            pred_info = f"Predição · {stem} — sem raster (classifique)"
            if meta and os.path.exists(pred.tif) and pred.model_stale(meta):
                # raster é de OUTRA versão do modelo (identidade sha1; mtime p/
                # sidecar legado) — não exibir como se fosse do ativo
                log.info("pred raster desatualizado (modelo re-treinado) — camada oculta")
                self.set_status("camada Predição oculta: o raster é de versão anterior do modelo — "
                                "classifique de novo pra reaparecer")
                self.classify.set_coverage("— (raster é de versão anterior do modelo)")
                pred_info = f"{stem} — raster de versão anterior"
                meta = None
            if meta and os.path.exists(pred.tif):
                T = int(meta.get("tile", 1024))
                total = (-(-self.class_src.height // T)) * (-(-self.class_src.width // T))
                ndone = len(meta.get("done_tiles", []))
                cov = f"{ndone}/{total} tiles ({100 * ndone / max(total, 1):.1f}%)"
                if ndone >= total:
                    cov = f"✔ imagem toda classificada ({total} tiles)"
                self.classify.set_coverage(cov)
                pred_info, pred_ok = f"Predição · {stem} · {cov}", True
                try:
                    src = RasterSource(pred.tif)
                    # cor da classe = cls_colors (fonte de verdade, editável) caindo no
                    # sidecar; assim editar uma cor PERSISTE mesmo após rebuild da camada
                    _ac = self._all_colors()
                    cols = [_ac.get(lb, meta["colors"].get(lb, "#888888"))
                            for lb in meta["labs"]]
                    colr = make_class_colorizer(cols, nodata=PredictionRaster.NODATA)
                    self._pred_labs = list(meta["labs"])
                    hidden = {self._pred_labs.index(c) for c in getattr(self, "_pred_hidden", set())
                              if c in self._pred_labs}
                    colr.set_hidden(hidden)          # reaplica classes ocultas no rebuild
                    self.basemaps[lbl] = (src, colr)
                except Exception as e:
                    log.warning("pred basemap n/d: %s", e)
        self.toolbar.pop.set_pred_info(pred_info, pred_ok)
        # camada em uso? re-aplica a fonte nova (ou sai dela, se sumiu)
        if self.toolbar.combo.currentText() == lbl:
            if lbl in self.basemaps:
                self.view.set_source(*self.basemaps[lbl])
            else:
                self.toolbar.combo.setCurrentText(self.ctx.init_label)
        if self.overlay_state["label"] == lbl:
            self.refresh_overlay()
        # predição como overlay TILED: reaponta a fonte nova (handle reaberto → vê
        # os tiles recém-escritos) ou sai da camada se ela sumiu
        if getattr(self, "_pred_overlay_tiled", False):
            if lbl in self.basemaps:
                self.view.set_tiled_overlay(*self.basemaps[lbl], alpha=self.pred_state["alpha"])
            else:
                self.view.clear_tiled_overlay()
                self._pred_overlay_tiled = False

    def refresh_models(self, select=None):
        combo = self.toolbar.model_combo
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("(nenhum)", None)
        for label, p in self.list_model_files():
            combo.addItem(label, p)
        idx = 0
        if select:
            for i in range(combo.count()):
                if combo.itemData(i) == select:
                    idx = i
                    break
        combo.setCurrentIndex(idx)
        combo.blockSignals(False)

    def on_model_change(self):
        self.load_active_model(self.toolbar.model_combo.currentData())

    # =====================================================================
    # Classify (área + lote)
    # =====================================================================
    def _classify_area(self, col0, row0, w, h, status, pbar=None):
        """Run the active model over a pixel area and paint the class overlay.

        pbar(pct) opcional: alimenta uma barra de progresso em fases (ler/inferir/pintar).
        """
        def _p(v):
            if pbar is not None:
                pbar(v)
        if self.tstate.get("net") is None:
            status("escolha um modelo primeiro (campo 'modelo:' no topo)")
            return
        # GPU é uma só: treino/lote no ar → recusa (o invariante treino×classificação
        # só cobria os botões train_btn×all_btn; classificar área ficava liberado).
        if self._training:
            status("treino rodando na GPU — aguarde terminar")
            return
        if self._batch_running:
            status("job de lote rodando na GPU — cancele ou aguarde")
            return
        # processEvents abaixo reabre o event loop: segundo clique no botão entraria
        # aqui de novo com a leitura da 1ª chamada no meio — reentrância bloqueada.
        if self._infer_running:
            return
        self._infer_running = True
        try:
            self._classify_area_inner(col0, row0, w, h, status, _p)
        finally:
            self._infer_running = False

    def _classify_area_inner(self, col0, row0, w, h, status, _p):
        col0 = int(max(0, col0)); row0 = int(max(0, row0))
        w = int(min(self.class_src.width - col0, w)); h = int(min(self.class_src.height - row0, h))
        if w <= 1 or h <= 1:
            status("área fora da imagem")
            return
        cap = 200_000
        step = max(1, int(np.ceil((h * w / cap) ** 0.5)))
        status(f"lendo {w}×{h} (step {step})…")
        _p(10)
        QApplication.processEvents()

        def _after_read(_hs, _ws):
            _p(55)

        def _before_infer(nv):
            status(f"inferência {nv} px…")
            _p(70)
            QApplication.processEvents()

        from ts_annotator.core.inference import classify_block
        ci, valid, (Hs, Ws) = classify_block(
            self.ts, self.tstate["net"], self.tstate["mu"], self.tstate["sd"],
            row0, col0, h, w, step=step, after_read=_after_read, before_infer=_before_infer)
        rgba = np.zeros((Hs, Ws, 4), np.ubyte)
        n = int(valid.sum())
        persist = ""
        if n:
            labs = self.tstate["labs"]
            _ac = self._all_colors()
            colmap = np.array([[QColor(_ac.get(lb, "#888")).red(),
                                QColor(_ac.get(lb, "#888")).green(),
                                QColor(_ac.get(lb, "#888")).blue()] for lb in labs], np.ubyte)
            fr = rgba.reshape(Hs * Ws, 4)
            fr[valid, :3] = colmap[ci]
            fr[valid, 3] = 255
            if step == 1:
                persist = self._persist_area(row0, col0, Hs, Ws, valid, ci)
            else:
                persist = "  (step>1 — só overlay, não persisti)"
        _p(95)
        persisted = step == 1 and "→" in persist   # gravou no raster (_persist_area)
        if persisted:
            # a área já aparece pela camada "Predição (modelo)" (tiled, suave em
            # qualquer zoom) que _persist_area ligou/atualizou — sem pintura volátil
            self.pred_overlay.setVisible(False)
            if getattr(self, "_pred_overlay_tiled", False):
                self.view.refresh_tiled_overlay()
        else:
            # step>1 (preview, não persiste) ou persistência falhou: overlay fixo
            self.pred_overlay.setImage(rgba, autoLevels=False)
            self.pred_overlay.setRect(QRectF(col0, row0, Ws * step, Hs * step))
            self.pred_overlay.setOpacity(self.pred_state["alpha"])
            self.pred_overlay.setVisible(True)
        _p(100)
        status(f"classificação {Ws}×{Hs} (step {step}) — {n} px válidos{persist}")

    def _persist_area(self, row0, col0, Hs, Ws, valid, ci):
        """Grava o bloco classificado (full-res) no raster por-modelo; nota p/ o status."""
        if self._batch_running:
            return "  (job de lote ativo — não persisti)"
        mp = self.toolbar.model_combo.currentData()
        if not mp:
            return "  (modelo sem arquivo — não persisti)"
        try:
            pred = PredictionRaster(self.pred_dir, mp)
            pred.ensure(self.class_src.width, self.class_src.height, self.TR,
                        self.class_src.crs, self.tstate["labs"], self._all_colors())
            out = np.full(Hs * Ws, PredictionRaster.NODATA, np.uint8)
            out[valid] = ci.astype(np.uint8)
            pred.write_block(out.reshape(Hs, Ws), row0, col0)
            self._start_overviews(pred.tif)   # atualiza a pirâmide → aparece no zoom-out
            self._refresh_pred_basemap()
            # o resultado aparece no RESULTADOS (camada "Predição (modelo)"), igual
            # ao job grande — não depende mais do overlay volátil da aba Classify.
            if (self.PRED_LABEL in self.basemaps
                    and self.toolbar.combo.currentText() != self.PRED_LABEL):
                self.toolbar.ov_combo.setCurrentText(self.PRED_LABEL)
            return f"  → {os.path.basename(pred.tif)}"
        except Exception as e:
            return f"  (persistência falhou: {e})"

    _SYNC_CAP = 200_000   # px full-res classificados sincronamente (segundos)

    def classify_scope(self):
        """Classifica a ÁREA DE TRABALHO, sempre full-res, sempre no raster por-modelo.

        Pequena (≤ _SYNC_CAP px) → síncrono, segundos. Grande → job de tiles em
        background restrito ao escopo: progressivo (a camada de predição vai sendo
        pintada), cancelável e retomável. 'Imagem toda' é só o caso escopo = tudo.
        """
        area = self._propose_area()
        if area is None:
            self.classify.set_status("selecione células: ligue a grade e clique nas células")
            return
        x0, y0, x1, y1 = area
        x0 = max(0.0, x0); y0 = max(0.0, y0)
        x1 = min(float(self.class_src.width), x1); y1 = min(float(self.class_src.height), y1)
        w, h = x1 - x0, y1 - y0
        if w <= 1 or h <= 1:
            self.classify.set_status("área fora da imagem")
            return
        if w * h <= self._SYNC_CAP and self.area_combo.currentData() != "grids":
            def _pbar(v):
                self.classify.progress.setVisible(v < 100)
                self.classify.progress.setValue(v)
            self._classify_area(x0, y0, w, h, self.classify.set_status, pbar=_pbar)
            return
        self.classify_all(tiles=self._scope_tiles(x0, y0, x1, y1))

    def classify_all(self, tiles=None):
        """Job de lote: classifica os tiles do escopo (None = imagem toda) → raster
        por-modelo.

        Roda no ClassifyAllWorker (thread própria, handles próprios). Retomável: o
        sidecar guarda os tiles concluídos (contagem GLOBAL — jobs parciais e o de
        imagem toda se acumulam); cancelar preserva o progresso.
        """
        if self._batch_running:
            return
        if self._training:
            self.classify.set_status("treino rodando na GPU — aguarde terminar antes de classificar")
            return
        mp = self.toolbar.model_combo.currentData()
        if not mp or self.tstate.get("net") is None:
            self.classify.set_status("escolha um modelo primeiro (campo 'modelo:' no topo)")
            return
        pred = PredictionRaster(self.pred_dir, mp)
        state = pred.ensure(self.class_src.width, self.class_src.height, self.TR,
                            self.class_src.crs, self.tstate["labs"], self._all_colors())
        meta = pred.load_meta()
        done = set(meta.get("done_tiles", []))
        n_done = len(done)
        # escopo já 100% classificado com este modelo? não refaz — avisa.
        T = int(meta.get("tile", 1024))
        n_all = (-(-self.class_src.height // T)) * (-(-self.class_src.width // T))
        covered = (n_done >= n_all) if tiles is None \
            else all(f"{t[0]}_{t[1]}" in done for t in tiles)
        if covered and state == "kept":
            self.classify.set_status("✔ essa área já está classificada com este modelo")
            self.set_status("área já classificada com este modelo — nada a refazer")
            return
        note = {"kept": f"retomando ({n_done} tiles já feitos)",
                "created": "raster novo",
                "reset": "modelo mudou — raster resetado"}[state]
        if tiles is not None:
            note += f" — escopo: {len(tiles)} tile(s)"
        job = {"model_path": mp, "pred": pred, "tiles": tiles,
               "month_paths": list(self.ts.paths), "bands": list(self.ts.bands),
               "band_names": getattr(self.ts, "band_names", None),
               "row_offsets": list(self.ts.row_offsets), "col_offsets": list(self.ts.col_offsets),
               "nodata": self.ts.nodata, "scale": self.ts.scale,
               "width": self.class_src.width, "height": self.class_src.height}
        self._batch_running = True
        self._prog_n = 0
        self.train_btn.setEnabled(False)   # GPU: não treinar durante o lote
        self.train_btn.setToolTip("desabilitado: GPU ocupada pelo job de classificação")
        self.classify.job_started()
        self.classify.set_status(f"{os.path.basename(pred.tif)} — {note}")
        # a camada "Predição (modelo)" já aponta pro raster do job — o display
        # progressivo (_on_batch_prog) só precisa re-ler a fonte
        self._refresh_pred_basemap()
        # auto-mostra JÁ NO INÍCIO como sobreposição: você vê a predição preencher
        # tile a tile AO VIVO (antes ela só aparecia no fim do job).
        if (self.PRED_LABEL in self.basemaps
                and self.toolbar.combo.currentText() != self.PRED_LABEL):
            self.toolbar.ov_combo.setCurrentText(self.PRED_LABEL)
        self.cworker.start(job)

    def _on_batch_prog(self, msg, pct):
        self.classify.job_progress(msg, pct)
        # display progressivo: o raster cresce tile a tile; re-lê a fonte
        # periodicamente (refresh sem piscar) — seja a predição o FUNDO ou a SOBREPOSIÇÃO.
        self._prog_n = getattr(self, "_prog_n", 0) + 1
        if self._prog_n % 12 == 0:
            if self.toolbar.combo.currentText() == self.PRED_LABEL:
                self.view.refresh()
            elif getattr(self, "_pred_overlay_tiled", False):
                self.view.refresh_tiled_overlay()   # camada tiled: reabre handle → vê tiles novos

    def on_classify_all_done(self, res):
        self._batch_running = False
        self.train_btn.setEnabled(True)
        self.train_btn.setToolTip("")
        self.classify.job_finished()
        self._refresh_pred_basemap()  # tiles novos (e overviews, se completou)
        # auto-mostra o resultado: ativa a sobreposição "Predição (modelo)" pra ela
        # aparecer sozinha no mapa (o job pinta o raster; sem isto ficaria invisível
        # até o usuário clicar "sobrepor"). Só se disponível e não for já o fundo.
        if ("error" not in res and self.PRED_LABEL in self.basemaps
                and self.toolbar.combo.currentText() != self.PRED_LABEL):
            self.toolbar.ov_combo.setCurrentText(self.PRED_LABEL)
        if "error" in res:
            self.classify.set_status("erro no job: " + res["error"])
        elif res.get("cancelled"):
            self.classify.set_status(
                f"cancelado — {res['done']}/{res['total']} tiles salvos em "
                f"{os.path.basename(res['tif'])} (retomável)")
        else:
            self.classify.set_status(
                f"completo: {os.path.basename(res['tif'])} ({res['total']} tiles, +overviews)")

    def _on_alpha(self, v):
        self.pred_state["alpha"] = v / 100.0
        self.pred_overlay.setOpacity(self.pred_state["alpha"])

    # =====================================================================
    # Train
    # =====================================================================
    def on_train_done(self, res):
        self._training = False
        self.train_btn.setEnabled(True)
        self.classify.btn.setEnabled(True)
        self.classify.btn.setToolTip("classifica a área de trabalho em full-res; grava em "
                                     "predictions/pred_<modelo>.tif (acumula no mesmo raster)")
        if "error" in res:
            snap = getattr(self, "_pending_snapshot", None)   # descarta snapshot órfão
            if snap and os.path.exists(snap):
                try:
                    os.remove(snap)
                except Exception:
                    pass
            self._pending_snapshot = None
            self.train_status.setText("erro: " + res["error"])
            self.train_win.done(False, "erro no treino", res["error"])
            return
        self.tstate.update(net=res["net"], mu=res["mu"], sd=res["sd"], labs=res["labs"])
        self.update_goals()
        saved_name = None
        prev_bacc = None   # bacc da versão anterior → delta honesto, sem juízo de valor
        try:
            from ts_annotator.core.version_store import VersionStore
            _vs_prev = VersionStore(self.model_dir).list_versions()
            if _vs_prev and (_vs_prev[-1]["meta"] or {}).get("metrics"):
                prev_bacc = _vs_prev[-1]["meta"]["metrics"].get("bacc")
            f1 = res["f1_per_class"]
            meta = {
                "dataset": os.path.basename(getattr(res.get("store"), "path", "") or "?"),
                "n_points": int(res.get("n_points", 0)),
                "anno_hash": res.get("anno_hash"),
                "params": res.get("params"),
                "metrics": {
                    "bacc": float(res["bacc"]), "macro_f1": float(res["macro_f1"]),
                    "f1_per_class": {res["labs"][i]: float(f1[i]) for i in range(len(res["labs"]))},
                    "n_groups": int(res.get("n_groups", 0)), "k": int(res.get("k", 0)),
                    # proveniencia da spatial-CV: separacao OBTIDA entre treino e
                    # validacao, e se a estratificacao valeu. Sem isto o numero de
                    # acuracia nao diz sob que separacao foi medido.
                    "stratified": bool(res.get("stratified", True)),
                    "buffer_px": float(res.get("buffer_px", 0.0)),
                    "dropped_by_buffer": int(res.get("dropped_by_buffer", 0)),
                    "min_separation_px": res.get("min_separation_px"),
                    "cleanlab_suspects": int(len(res["issues"])),
                    "confusion": res.get("confusion"),   # antes era calculada e descartada
                },
            }
            name, mpath = VersionStore(self.model_dir).save_version(
                res["net"], res["mu"], res["sd"], res["labs"], meta, arch=res.get("arch"))
            saved_name = name
            # congela o dataset da versão: move o snapshot (tirado no início do
            # treino) pra dentro do dir da versão → models/it_vN/dataset.json é o
            # dado EXATO com que o it_vN foi treinado (reprodutibilidade modelo↔dado).
            snap = getattr(self, "_pending_snapshot", None)
            if snap and os.path.exists(snap):
                try:
                    import shutil
                    shutil.move(snap, os.path.join(os.path.dirname(mpath), "dataset.json"))
                except Exception as e:
                    log.warning("mover snapshot do dataset falhou: %s", e)
            self._pending_snapshot = None
            log.info("modelo salvo: %s (%d pts, bacc %.3f)", name, meta["n_points"], res["bacc"])
            self.refresh_models(select=mpath)
            # refresh_models repopula com blockSignals → on_model_change NÃO roda.
            # O modelo ativo acabou de mudar: sem isto a camada "Predição (modelo)"
            # continuaria exibindo o raster da versão ANTERIOR como se fosse a ativa
            # (e o rótulo "modelo:" da aba Classify ficaria com o nome velho).
            self._refresh_pred_basemap()
            self.classify.set_model(name)
        except Exception as e:
            log.error("save model: %s", e)
        labs = res["labs"]
        oof = res["oof_proba"]
        sc = res["scores"]
        ytr = res["y"]     # label EFETIVA (índice) usada no treino — nível do modelo
        # writeback no store em que o treino FOI FEITO (res["store"]) — o dataset
        # ativo pode ter mudado durante o treino e os ids colidem entre datasets
        st = res.get("store") or self.store
        by_id = {p["id"]: p for p in st.points}
        issue_ids = {res["ids"][i] for i in res.get("issues", [])}
        for k, pid in enumerate(res["ids"]):
            p = by_id.get(pid)
            if p is not None:
                p["_clean"] = float(sc[k])
                p["_pred"] = labs[int(oof[k].argmax())]
                p["_predp"] = float(oof[k].max())
                p["_issue"] = pid in issue_ids   # cleanlab issue → bate com o contador do treino
                # rótulo do ponto NO NÍVEL do treino (super se colapsado): a tabela
                # compara _pred vs _ytrain (mesmo nível), senão marcaria tudo discordante
                p["_ytrain"] = labs[int(ytr[k])]
        st.save()   # persiste as métricas (antes só sobreviviam se um rótulo salvasse depois)
        delta = f"  (anterior: {prev_bacc:.3f})" if prev_bacc is not None else "  (primeiro treino)"
        self.train_status.setText(f"OK — bacc {res['bacc']:.3f}{delta} · macro-F1 {res['macro_f1']:.3f}  "
                                  f"({res.get('n_groups', '?')} blocos · {res.get('k', '?')} folds)")
        f1 = res["f1_per_class"]
        labs = res["labs"]
        lines = " · ".join(f"{labs[i]}:{f1[i]:.2f}" for i in range(len(labs)))
        self.train_metrics.setText("F1/classe — " + lines + f"\n\ncleanlab: {len(res['issues'])} suspeito(s)")
        self.train.show_review_button(len(res["issues"]))
        self._labels_since_train = 0
        self._refresh_counters()
        self.set_status(f"treino OK — bacc {res['bacc']:.3f}{delta}")
        self.refresh_review_list()
        # aba Resultado da janela: métricas formatadas + matriz de confusão
        f1h = "<br>F1/classe: " + " · ".join(f"{labs[i]} {f1[i]:.2f}" for i in range(len(labs)))
        f1h += f"<br>cleanlab: {len(res['issues'])} suspeito(s)"
        head = (f"<b>{saved_name or 'modelo'}</b> · bacc <b>{res['bacc']:.3f}</b>{delta} · "
                f"macro-F1 {res['macro_f1']:.3f}  ({res.get('n_groups','?')} blocos · {res.get('k','?')} folds)")
        self.train_win.done(True, head, f1h, confusion=res.get("confusion"), labels=labs)

    def do_train(self):
        if self._batch_running:
            self.train_status.setText("job de classificação rodando — aguarde ou cancele antes de treinar")
            return
        if self._infer_running:   # classify síncrono bombeia processEvents — GPU ocupada
            self.train_status.setText("classificação de área em andamento — aguarde uns segundos")
            return
        self._training = True
        self.train_btn.setEnabled(False)
        self.classify.btn.setEnabled(False)  # GPU ocupada: não classificar durante o treino
        self.classify.btn.setToolTip("desabilitado: treino rodando na GPU")
        self.train_status.setText("treinando… (alguns minutos)")
        self.set_status("treinando na GPU… (janela de progresso aberta)")
        lr, ep, folds = self.train.params()
        collapsed = self.train.collapse_config()   # supers a colapsar (nível de treino)
        # congela o dataset ATUAL (estado no início do treino) — vira o snapshot
        # reprodutível da versão salva em on_train_done. Tirar aqui (não no fim)
        # casa 1:1 com os pontos que o worker acabou de ler.
        self._pending_snapshot = None
        try:
            src = getattr(self.store, "path", None)
            if src and os.path.exists(src):
                import shutil
                import tempfile
                fd, tmp = tempfile.mkstemp(suffix=".json", prefix="dsnap_")
                os.close(fd)
                shutil.copy(src, tmp)
                self._pending_snapshot = tmp
        except Exception as e:
            log.warning("snapshot do dataset falhou: %s", e)
        self.train_win.start(f"spatial-CV {folds} folds · {ep} épocas · lr {lr}")
        nblk, buf = self.train.cv_config()
        self.tworker.start(lr, ep, folds, self.class_super, collapsed,
                           n_blocks=nblk, buffer_px=buf, arch=self.train.model_type())

    # =====================================================================
    # Workers / threads (chamado pela janela DEPOIS de construir os widgets)
    # =====================================================================
    def init_workers(self):
        # modelo inicial: a VERSÃO mais nova; senão o .pt legado do ctx (it_latest)
        from ts_annotator.core.version_store import VersionStore
        sel = VersionStore(self.model_dir).latest() \
            or (self.model_path if os.path.exists(self.model_path) else None)
        self.refresh_models(select=sel)
        self.on_model_change()

        # treino threaded (GPU)
        self.tthread = QThread()
        self.tworker = TrainWorker(self.store, self.fx,
                                   band_names=getattr(self.ts, "band_names", None))
        self.tworker.moveToThread(self.tthread)
        self.tthread.start()
        self.tworker.done.connect(self.on_train_done)
        self.tworker.prog.connect(lambda m: self.train_status.setText(m))
        self.tworker.prog.connect(self.train_win.progress)   # log ao vivo na janela pop
        self.tworker.step.connect(self.train_win.step)       # barra/ETA/loss/folds ao vivo

        # job de lote (classificar imagem toda) — thread própria
        self.cthread = QThread()
        self.cworker = ClassifyAllWorker()
        self.cworker.moveToThread(self.cthread)
        self.cthread.start()
        self.cworker.done.connect(self.on_classify_all_done)
        self.cworker.prog.connect(self._on_batch_prog)
        # Direct: o event loop da cthread fica bloqueado no job; o cancel precisa
        # rodar na thread da UI (só seta uma flag, checada a cada tile).
        self.classify.cancelRequested.connect(self.cworker.cancel,
                                              Qt.ConnectionType.DirectConnection)

        # reconstrução de overviews do pred raster após classify síncrono (thread própria):
        # sem isto a área recém-classificada some no zoom-out (pirâmide desatualizada)
        self.othread = QThread()
        self.oworker = OverviewWorker()
        self.oworker.moveToThread(self.othread)
        self.othread.start()
        self.oworker.done.connect(self._on_overviews_built)
        self._ovbuild_running = False
        self._ovbuild_pending = None

        self.train.set_superclasses(self._super_members())   # popula o seletor de nível
        if hasattr(self.panel, "set_hierarchy"):
            self.panel.set_hierarchy(self.class_super, self.super_colors)  # árvore no painel
        self._refresh_counters()   # contadores nascem com o dataset inicial

    def _start_overviews(self, tif):
        """Reconstrói a pirâmide do pred raster em background (não trava a UI).
        Coalesce: se já roda, guarda o último pedido e refaz ao terminar."""
        if self._batch_running:
            return   # o job de lote constrói a própria pirâmide no fim
        if self._ovbuild_running:
            self._ovbuild_pending = tif
            return
        self._ovbuild_running = True
        self.oworker.start(tif)

    def _on_overviews_built(self, tif):
        self._ovbuild_running = False
        self._refresh_pred_view()   # reabre handles → o zoom-out vê a pirâmide nova
        if self._ovbuild_pending:
            nxt, self._ovbuild_pending = self._ovbuild_pending, None
            self._start_overviews(nxt)

    def shutdown(self):
        # job de lote no ar: cancela e espera o tile corrente terminar (progresso fica salvo)
        if self._batch_running:
            self.cworker.cancel()
        # SEMPRE encerra as duas threads: QThread destruído ainda rodando = qFatal/abort
        # na saída (era o "QThread: Destroyed while thread is still running").
        for th, timeout in ((getattr(self, "cthread", None), 30000),
                            (getattr(self, "tthread", None), 5000),
                            (getattr(self, "othread", None), 10000)):
            if th is not None:
                th.quit()
                th.wait(timeout)   # treino CUDA no ar não é cancelável — espera limitada
