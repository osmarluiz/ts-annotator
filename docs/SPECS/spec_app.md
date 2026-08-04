# `AnnotatorWindow` + `app/` + launcher — a "casca" do anotador

> A classe central (`AnnotatorWindow`) e a fiação que a alimenta. Tudo específico do
> projeto entra pelo launcher (`build_context`), vira `AppContext`, e a janela monta a UI
> + toda a lógica de interação. Convenções globais em [`README.md`](README.md).

## Convenções transversais

**Coordenadas (dois sistemas, não confundir):**
- **Pixel/raster:** `(row, col)` — row = Y (baixo), col = X (direita). `AnnotationStore`,
  `read_curve(row,col)`, `read_block(row0,col0,...)` e persistência usam `(row,col)`.
- **Cena/view (pyqtgraph):** `x = col`, `y = row`. Todo item `pg` (`pos=(col,row)`,
  `setRect(QRectF(col0,row0,...))`, `viewRect()` dá `x=col,y=row`). `mapSceneToView` dá
  ponto com `.x()=col`, `.y()=row` → `find_at(int(p.y()), int(p.x()))`.

**Exclusão mútua de GPU:** treino XOR classify. Flags criadas em `_init_model_and_thread`:
`self._training` (set em `do_train`, limpo em `on_train_done`) e `self._batch_running`
(set em `classify_all`, limpo em `on_classify_all_done`). Cada entrada checa o flag do
outro e recusa com status. Botões cruzados: `do_train` desabilita `classify.all_btn`;
`classify_all` desabilita `train_btn`. `_persist_area` recusa escrever se `_batch_running`.

**Threads:** dois `QThread` de longa vida com `QObject` workers (`moveToThread`),
disparados por um sinal interno `_go`. `tthread`→`tworker` (`TrainWorker`); `cthread`→
`cworker` (`ClassifyAllWorker`). Resultados voltam por `done`/`prog` (queued). Exceção:
`cancelRequested→cworker.cancel` é **`DirectConnection`** (o loop da cthread está preso no
job; o cancel roda na UI e só seta um flag polado por tile).

---

## `annotator_window.py` — `AnnotatorWindow(QWidget)`

Layout: **nav (esq) | (mapa sobre curva, splitter vertical) | cartões (dir)** + `MapToolbar`
flutuante. Constantes: `GRID_COLS = 10`; `PRED_LABEL = "Predição (modelo)"` (basemap
dinâmico que segue o `pred_<modelo>.tif` do modelo ativo).

### Dicts de estado

**`self.state`:** `last` (`(row,col,curve)` do pixel atual — alvo de rótulo), `size`
(marcador, 12), `review` (cursor em `store.points`, −1), `pseed` (semente do propor,
++ a cada propose), `grid` (modo grade), `cells` (set `(cell_row,cell_col)`), `ref`
(reservado), `last_src` (`"map"`/`"sugg"` — dirige o pós-rótulo), `sugg` (lista de
candidatos), `sugg_cur` (índice), `pan` (spacebar, lazy), `draw` (drag de retângulo, lazy).

**`self.tstate`:** `{net, mu, sd, labs}` (modelo ativo; `None` sem modelo). Escrito por
`load_active_model`/`on_train_done`; lido por `predict_point`/`_classify_area`/`classify_all`.

**`self.pred_state`:** `{alpha:0.6}` (opacidade do overlay de predição).
**`self.overlay_state`:** `{label:None, alpha:0.6}` (overlay basemap-sobre-basemap).
**`self.div_cache`:** `{n:-1, feats:None}` (features dos rotulados p/ a métrica diversidade;
invalidado quando `n` muda ou na troca de dataset).
**`self.hover_pos`:** `{p: scene_pos}` (última posição do mouse).
Outros: `cell_px` (=`class_src.width/GRID_COLS`), `cell_rects`, `vlayer_items`.

### Construção & montagem

- `__init__(ctx)` — liga todos os campos do `AppContext` a atributos; `store =
  datasets[dataset_default]`; inicializa os dicts; injeta `PRED_LABEL` no grupo
  "Resultado"; sequência `_build_map_items → _build_toolbar → _build_annotate →
  _build_review → _build_train → _build_classify → _build_nav → _assemble → _wire →
  _init_model_and_thread`; `QTimer.singleShot(0, self.fit)`.
- `_build_map_items` — itens pyqtgraph com Z: `markers` (60), `prop_marker` (65),
  `hover_marker` (66), `pred_overlay` (45, oculto), `overlay_item` (44, oculto),
  `grid_lines` (55, oculto).
- `_build_toolbar` — `MapToolbar(...)`; liga os 10 sinais → métodos (`datasetChanged→
  on_dataset_change`, `basemapChanged→_on_basemap`, `modelChanged→on_model_change`, etc.).
- `_build_annotate` — página Annotate: `goal_combo` (de `GOALS`), `tcls_combo` (só p/ goal
  "class"), `area_combo` (`viewport`/`all`/`grids`/`rect`), `count_spin` (1–100, 10),
  `rect_roi` (ROI arrastável), `draw_btn`, `pb`, `propose_bar`, `scount`, `slist`.
  **Monkey-patch** de `vb.mouseDragEvent → _vb_drag`.
- `_build_review` — `ReviewPanel(classes)`; liga `filtersChanged→refresh_review_list`,
  `pointSelected→on_review_point`.
- `_build_train` — página Train: `lr_spin` (0.0001–0.1, 0.001), `ep_spin` (10–600, 120),
  `fold_spin` (2–10, 5), `train_btn`, `train_status`, `train_metrics`; subseção Predição:
  `pred_run_btn`, `pred_alpha`, `pred_status`.
- `_build_classify` — `ClassifyPanel()`; liga `classifyRequested→classify_scope`,
  `classifyAllRequested→classify_all`, `alphaChanged→_on_alpha`, `clearRequested→ocultar`.
- `_build_nav` — rail + `QStackedWidget` (0 annot, 1 review, 2 train, 3 classify);
  `_nav_btn` exclusivo (index 1 chama `refresh_review_list`).
- `_assemble` — splitter vertical (view/curve) dentro de horizontal (nav|center|panel),
  `resize(1500,950)`.
- `_wire` — timers (`overlay_timer` 120ms debounced, `hover_timer` 60ms, `marker_timer`
  120ms), event filter global (spacebar pan), `scene.sigMouseMoved→on_hover`,
  `view.pixelClicked→on_click`, `panel.classChosen→on_label`, `removeRequested→on_remove`,
  `classAdded→on_class_added`.
- `_init_model_and_thread` — popula o combo de modelos + `on_model_change`; cria `tthread`+
  `TrainWorker` e `cthread`+`ClassifyAllWorker`; flags GPU; `cancelRequested→cworker.cancel`
  **DirectConnection**.
- `show` — `super().show()` + `toolbar.reposition()`.
- `closeEvent` — se job rodando: `cworker.cancel()`, `cthread.quit()`, `wait(30000)`
  (deixa o tile atual salvar o progresso).
- `fit` — encaixa a imagem no viewbox (~10% margem) + refresh markers/review.

### Basemap / overlay
- `toggle_layer(name,on)` — cria `VectorLayer` lazy (1ª vez) e liga/desliga.
- `refresh_overlay(*_)` — lê o basemap de overlay no viewRect (decimado ≲1.5Mpx),
  coloriza, pinta `overlay_item` com a opacidade; oculta se sem label/fora/erro.
- `set_overlay(name)` / `set_overlay_alpha(v)`.
- `_on_basemap(name)` — `view.set_source(*basemaps[name])`.
- `_refresh_pred_basemap()` — reconstrói o `PRED_LABEL`: fecha a fonte velha, resolve o
  modelo ativo, abre `PredictionRaster`, lê sidecar. **Guarda de stale:** se `model_mtime`
  do sidecar ≠ mtime do `.pt` → oculta (predição de versão antiga). Se ok, abre
  `RasterSource(.tif)` + colorizer das cores do sidecar (índice→classe→cor do modelo).
  Se a camada está ativa, re-aplica (ou volta pra `init_label` se sumiu).

### Dataset
- `on_dataset_change()` — troca `store`; **re-aponta `tworker.store`** (senão treina o
  dataset antigo e grava métricas em ids colididos); invalida `div_cache`; descarta
  `sugg`/`sugg_cur`/`review` (eram do dataset anterior); limpa lista; refresh.

### Grade
- `draw_grid_lines()`, `toggle_cell(row,col)` (key `(row//cell_px, col//cell_px)`),
  `set_grid(on)` (exclusivo com draw-rect), `set_cols(n)` (recalcula `cell_px`),
  `set_size(v)`.

### Marcadores / snap / hover / pan
- `refresh_markers()` — pontos no viewRect (decimado ≤4000), `pos=(col,row)`, cor por classe.
- `_snap_radius()` — raio em px por zoom (`max(2,nd*8)`) ou `None` se muito longe (`nd>12`).
- `select_point(p)` — clique num ponto existente: relê curva, `set_scores` com pred +
  classe + `quality=_clean`, `state["last"]`, prop_marker, `review.select_pid`.
- `predict_point(curve) -> dict|None` — probs do modelo ativo; `None` se sem modelo **ou
  curva não-finita** (nodata parcial → softmax NaN; igual ao batch).
- `on_click(row,col)` — no-op se `pan`; grade → `toggle_cell`; snap → `select_point`; senão
  lê curva, **rejeita parcial-nodata**, mostra curva+scores, `state["last"]`.
- `navigate(row,col,curve,scores,existing_cls,cyc,src="map",quality=None)` — centraliza
  (±750), curva+scores (novidade via `eng.is_novel`), `state["last"]`/`last_src`, prop_marker.
- `on_hover(scene_pos)` (só grava), `_proc_hover()` (timer 60ms; snap + hover_marker),
  `eventFilter` (spacebar → `pan`).

### Rótulo → store
- `on_label(cls)` — exige `state["last"]`; `store.add_or_update(...)`; refresh; se
  `last_src=="sugg"` → `advance_suggestion`, senão re-mostra.
- `on_remove()` — `store.remove_at`; refresh.
- `on_class_added(name)` — propaga a cor da classe nova (do painel) pra `cls_colors`/
  `classes`; refresh.

### Propor (active learning)
- `_propose_area() -> (x0,y0,x1,y1)` — escopo: `grids` (bbox das células), `all`, `rect`
  (`rect_roi`), senão viewport.
- `_cand_text(c,metric)`, `fill_suggestions()` (lista sob `blockSignals`), `_store_feats()`
  (features dos rotulados, cacheadas).
- `do_propose()` — botão `pb`: ++`pseed`, área+goal+target+exclude, desabilita botão,
  mostra `propose_bar`, `sel.propose_many(...)` com `_tick` (processEvents a cada 8),
  guarda `state["sugg"]`, seleciona linha 0.
- `on_sugg_select(cur,_prev)` — resolve candidato; **relê a curva full-res exata** no pixel
  (candidato pode vir de grade decimada; fallback pra curva do candidato se nodata);
  `navigate(...,src="sugg")`.
- `advance_suggestion()`, `skip_suggestion()`.
- `update_goals()` — habilita/desabilita goals por `needs_model` (tem modelo?), mostra
  `tcls_combo` só p/ goal "class".
- `review_goto(delta)` (stepper wrap-around), `refresh_review_list()` (filtros → filtra/
  ordena/`[:500]` → linhas com badges `⚑clean`/`≠pred`), `_sim_own`/`_mval` (chaves de
  ordenação), `on_review_point(pid)` (navega ao ponto).

### Modelo / predição / classify
- `list_model_files()`, `load_active_model(path)` (→ `tstate` + `update_goals` +
  `_refresh_pred_basemap`), `refresh_models(select)`, `on_model_change()`.
- `_classify_area(col0,row0,w,h,status,pbar=None)` — classificador de área síncrono: exige
  modelo; clampa; decima ≲200k px; `ts.read_block`; `build_X`+`predict_proba`; argmax;
  pinta `pred_overlay`. **Persiste só se `step==1`** (`_persist_area`). `pbar` em fases
  (10/55/70/95/100).
- `_persist_area(row0,col0,Hs,Ws,valid,ci) -> str` — grava bloco full-res no
  `PredictionRaster` (`ensure`+`write_block`) + `_refresh_pred_basemap`. **Recusa** se
  `_batch_running` ou modelo sem arquivo.
- `run_prediction()` (Train, viewport), `classify_scope()` (Classify, escopo `grade`/vista,
  barra ligada em `classify.progress`).
- `classify_all()` — job de lote imagem-toda: recusa se `_batch_running`/`_training`; exige
  modelo; `pred.ensure(...)` (`kept`/`created`/`reset`); monta `job`; `_batch_running=True`;
  desabilita `train_btn`; `job_started()`; `cworker.start(job)`.
- `on_classify_all_done(res)` — limpa flag, reabilita, `job_finished`, `_refresh_pred_basemap`;
  reporta erro/cancelado(`done/total`)/completo.
- `do_train()` — recusa se `_batch_running`; `_training=True`; desabilita ambos; `tworker.start`.
- `on_train_done(res)` — limpa `_training`, reabilita; atualiza `tstate`; salva `it_latest.pt`;
  grava `_clean`/`_pred`/`_predp` por ponto (por id); atualiza métricas; refresh review.
- `_on_alpha(v)` — opacidade do `pred_overlay`.

### Desenho de retângulo
- `_vb_drag(ev,axis)` (override do drag; desenha `rect_roi` se `state["draw"]`, senão
  delega), `set_draw(on)` (exclusivo com grade), `_sync_draw_btn()` (mostra `draw_btn` só p/ `rect`).

---

## `app/session.py` — `AppContext` (dataclass)

| campo | tipo | significado |
|---|---|---|
| `view` | TiledMapView | mapa, já no basemap inicial |
| `curve_view` | CurveView | plot da curva |
| `panel` | SimilarityPanel | cartões de classe |
| `datasets` | dict[str,AnnotationStore] | datasets selecionáveis |
| `dataset_default` | str | dataset ativo no start |
| `ts` | TimeSeriesSource | clique→curva |
| `fx` | FeatureExtractor | curva→features (`f[0]`=ciclos) |
| `eng` | SimilarityEngine | similaridade+novidade |
| `sel` | SelectionEngine | propor candidatos |
| `class_src` | RasterSource | basemap de classe (define tamanho/transform/crs) |
| `transform` | affine | geo↔pixel (vira `self.TR`) |
| `basemaps` | dict[str,(RasterSource,colorizer)] | label→fonte+colorizer |
| `basemap_groups` | list[(grupo,[labels])] | agrupamento do combo "base:" |
| `vlayer_paths` | dict[str,str] | overlays vetoriais |
| `classes` | dict[str,str] | classe→cor (painel) |
| `cls_colors` | dict[str,str] | classe→cor (marcadores/predição) |
| `model_dir` / `model_path` / `pred_dir` | str | modelos / `it_latest.pt` / predições |
| `init_label` | str | basemap de observação inicial |

## `app/workers.py`

Ambos são `QObject` em `QThread`, disparados por `_go` (privado). `start(...)` só guarda
params e emite `_go`; o loop pesado roda na thread do worker.

**`TrainWorker`** — sinais `done(object)`, `prog(str)`, `_go`. `TrainWorker(store, fx)`
guarda `store` **por referência** (por isso `on_dataset_change` re-aponta). Defaults
`lr=1e-3, epochs=120, folds=5`.
- `start(lr,epochs,folds)`; `_run()` — filtra pontos com curva **inteira finita** (1 NaN
  envenena `mu/sd`); exige **≥30**; blocos de CV via KMeans nas `(row,col)` (`nblk=
  min(N//4, max(6*folds,30))`); `spatial_cv` + `label_scores` (cleanlab) + `fit`; emite
  `res` (`labs,y,scores,issues,net,mu,sd,ids,oof_proba,bacc,macro_f1,f1_per_class,...`).

**`ClassifyAllWorker`** — sinais `done(object)`, `prog(str,int)` (msg,%), `_go`. Abre
**handles próprios** e carrega **cópia própria** do modelo (rasterio não é thread-safe).
- `start(job)`; `cancel()` (seta `_cancel`, chamado cross-thread por DirectConnection,
  polado por tile); `_run` → `_run_job`; `_fmt(sec)` (ETA).
- **`job` dict:** `model_path`, `pred` (PredictionRaster já `ensure`d), `month_paths`,
  `bands`, `row_offsets`, `col_offsets`, `nodata`, `scale`, `width`, `height`.
- **Retomável/cancel:** tile `T` (1024, do sidecar); enumera `(ti,tj)`, subtrai `done_tiles`,
  processa só `todo`; por tile: checa `_cancel`, lê cubo, `build_X`+`predict_proba
  (batch=65536)`, argmax→uint8 (fill `NODATA`), grava a janela; a cada 25 tiles
  `save_progress`. Cancel → salva + `{cancelled:True, done, total, tif}`. Fim →
  `build_overviews` + `save_progress(complete=True)` + `{ok:True, total, tif}`. `finally`
  fecha writer + TimeSeriesSource.

## `app/theme.py`
`ICON="#cfd3dc"`; `APP_QSS` (fallback dark); `ACCENT_QSS` (`#floatbar`, `#navbtn:checked`
azul, etc.); `apply_theme(app)` — tenta `qdarktheme.setup_theme("dark", additional_qss=
ACCENT_QSS)`, fallback `setStyleSheet(APP_QSS+ACCENT_QSS)`.

## `app/config.py` — `GOALS`
Lista `(label, (metric, order, needs_model))`:

| label | metric | order | needs_model |
|---|---|---|---|
| Modelo incerto (confiança baixa) | confidence | asc | True |
| Modelo ambíguo (margem pequena) | margin | asc | True |
| Modelo muito incerto (entropia alta) | entropy | desc | True |
| Modelo × similaridade discordam | disagreement | desc | True |
| Diferente do que já rotulei | diversity | desc | False |
| Exemplos típicos de uma classe | class | desc | False |

`needs_model=True` desabilitado até ter modelo (`update_goals`). `"class"` mostra
`tcls_combo`; `"diversity"` dispara `_store_feats`; `order` vira `propose_many(order=)`.

## `examples/run_annotation_demo.py` — launcher

- **Caminhos:** `SITS=D:\PROJECTS\sits`, `REF=dataset_balanced.npz` (referência limpa),
  **`AOI_DIR=E:\sits_aoi_crop`** (o app trabalha só na AOI), `AOI_COL0,AOI_ROW0=2305,4608`
  (origem do crop na grade cheia, p/ remapear pontos antigos), `PRED_DIR=AOI_DIR\predictions`,
  `MODEL_DIR=examples/models`, `MODEL_PATH=it_latest.pt`, `OUT_JSON=points.json`,
  `ANTIGOS_JSON=points_antigos.json`.
- `ORDER` — 12 períodos `202210…202309` (eixo temporal).
- `CLS_COLORS`, `CYC={1:'1_ciclo',2:'2_ciclos',3:'3_ciclos'}` (re-divide `cultivo` por ciclo).
- `load_reference(fx)` — carrega `REF`, transpõe `(N,12,4)`, re-rotula `cultivo→CYC[f[0]]`.
- `load_antigos_store(transform,crs)` — pontos DF antigos como dataset selecionável/editável;
  constrói de `ref_points.npz` (remapeando `−AOI_ROW0/−AOI_COL0`) e cacheia em JSON.
- `build_context()` — `FeatureExtractor`; abre `class_src` (`basemap_class_aoi.tif`)→`TR`;
  `TimeSeriesSource` sobre os 12 meses (offsets 0, bandas 1–4, nodata 65535, escala 10000);
  basemaps agrupados **Dado (observação)** (ndvitemporal/mnf/pheno) × **Resultado (produto)**
  (class/intensity/regen); abre na 1ª camada de observação (`init_label`, anti-viés);
  `SimilarityEngine(k=3, novelty_thr=0.45)` na referência; `SelectionEngine`; dois datasets
  (`antigos`/`trabalho`, default antigos); retorna o `AppContext`.
- `main()` — QApplication, `apply_theme`, `build_context`, `AnnotatorWindow(ctx)`, exec.

---

## Onde editar (janela)

- **Nova aba:** `_build_<x>` (guarda `self.<x>_page`) + `stack.addWidget` em `_build_nav` +
  `_nav_btn("Label","mdi6.icon",idx)` (refresh no clique se preciso, como o índice 1).
- **Novo goal de AL:** tupla em `config.GOALS`; garanta que `propose_many` entenda a
  `metric`/`order` e `_cand_text` se quiser readout; `update_goals` cuida do gating.
- **Mudar o clique numa sugestão:** `on_sugg_select` (relê full-res, `navigate(src="sugg")`);
  pós-rótulo em `on_label` (`last_src=="sugg"`→`advance_suggestion`).
- **Novo job de background:** espelhe `ClassifyAllWorker` (QObject + `done`/`prog`+`_go`,
  `start(job)`, `_run` com handles próprios, `cancel()` polado); instancie em
  `_init_model_and_thread` (thread própria); guarde com `_training`/`_batch_running`;
  limpe no `closeEvent`.
- **Novo basemap/overlay:** estenda `_defs` no launcher; a camada dinâmica `PRED_LABEL` é
  caso especial em `_refresh_pred_basemap` (não colidir).
- **Novo dataset:** adicione um `AnnotationStore` em `datasets` no `build_context` (a troca e
  o re-aponta de `tworker.store` já são tratados).
