# `ui/` — painéis, mapa tiled, colorizers (PyQt6 + pyqtgraph)

> Os painéis são **"dumb views"**: constroem controles, emitem sinais, e expõem
> setters/refresh que a `AnnotatorWindow` chama depois de rodar a lógica. Nenhum painel
> mexe no store ou no mapa direto. Contrato de sinais no fim. Ver [`README.md`](README.md).

---

### `ui/map_toolbar.py` — barra flutuante sobre o mapa (dataset/base/overlay/modelo, layers, grade, tamanho do ponto, pins)

`QFrame` semi-transparente parentado à `view`, posicionado em `(10,8)`. Constrói controles e emite sinais; toda a lógica vive na janela. Cuida só do próprio comportamento flutuante (pin / hover-reveal / reposicionar no resize).

- **`MapToolbar(QFrame)`** — `__init__(self, view, basemap_groups, overlay_labels, vlayer_names, init_label, grid_cols, dataset_names, dataset_current)`.
  - `basemap_groups` → combo agrupado (`— grupo —` desabilitado + separadores). `overlay_labels` → combo "sobre:" com `(nenhum)` prepend. `vlayer_names` → menu "layers" checkable.
  - **Sinais:** `datasetChanged()`, `basemapChanged(str)`, `overlayChanged(str)`, `overlayAlphaChanged(int)`, `layerToggled(str,bool)`, `gridToggled(bool)`, `gridColsChanged(int)`, `pointSizeChanged(int)`, `navToggled(bool)`, `modelChanged()`.
  - **Widgets:** `dataset_combo`, `combo` (base agrupado), `ov_combo`+`ov_sld` (0–100, init 60), `model_combo` (**populado pela janela depois**), `layers_btn`/`lmenu`, `gcb` (grade), `gcols` (4–40), `sld` (ponto 4–30, init 12), `pin_btn` (fixa a barra), `panel_pin` (hambúrguer → nav).
  - **Métodos:** `set_grid_checked(on)`, `reposition()`. Privados: `_check_hover()` (mostra se pinado, senão só no topo), `_on_resize` (monkey-patch de `view.resizeEvent`).
  - **Gotchas:** monkey-patcha `view.resizeEvent` (dois toolbars na mesma view colidem). `basemapChanged` é ligado **antes** do `setCurrentIndex(init)` → **emite durante a construção** (a janela tem que tolerar ou ligar depois). `model_combo` vazio na construção → `modelChanged` dispara enquanto a janela popula (use `blockSignals`).
  - **Onde editar:** novo controle → widget no `ctrl` + `pyqtSignal` + fiação em `_wire`/`_build`. Novo grupo de base/overlay → passar `basemap_groups`/`overlay_labels` diferentes (sem mudar a toolbar).

---

### `ui/review_panel.py` — aba "Revisar": filtros (classe/métrica/ordem) + lista ranqueada

Dumb view: expõe a seleção de filtros e renderiza linhas calculadas pela janela; emite `filtersChanged` (re-ranquear) e `pointSelected(pid)` (navegar). Toda a lógica de ranking fica na janela.

- **`ReviewPanel(QWidget)`** — `__init__(self, classes)`.
  - **Sinais:** `filtersChanged()` (qualquer filtro/métrica/ordem/`rdisc` muda), `pointSelected(int)` (item selecionado, pid válido).
  - **Widgets:** `rfilter` (classe, data=nome ou `None`), `rmetric` (data: `"clean"`/`"predp"`/`"disc"`/`"sim"`/`"id"`), `rorder` (`"asc"`/`"desc"`), `rdisc` (checkbox "só onde o modelo discorda"), `rcount`, `rlist` (pid no `UserRole`, cor no foreground).
  - **Métodos:** `filters() -> (classe, métrica, ordem, disc_only)`; `set_items(rows=[(pid,texto,cor)])` (sob `blockSignals`); `set_count(texto)`; `select_pid(pid)` (sob `blockSignals`, evita loop de navegação). Privado: `_on_sel` → emite `pointSelected`.
  - **Gotchas:** `set_items`/`select_pid` bloqueiam sinais pra não re-emitir `pointSelected`. Todos os filtros ligam no mesmo `filtersChanged`.
  - **Onde editar:** novo filtro → widget + ligar no `filtersChanged` + estender `filters()` + consumir no `refresh_review_list` da janela. Nova métrica → `rmetric.addItem(label, key)` + tratar a key na janela.

---

### `ui/classify_panel.py` — aba "Classify": rodar o modelo numa área + UI de progresso do job de lote

Dumb view: expõe o escopo e emite request/cancel/alpha/clear. Inferência, overlay e o job de lote vivem na janela; o painel só mostra progresso via `job_started`/`job_progress`/`job_finished`.

- **`ClassifyPanel(QWidget)`** — `__init__(self)`.
  - **Sinais:** `classifyRequested()`, `classifyAllRequested()`, `cancelRequested()`, `alphaChanged(int)` (0–100, init 60), `clearRequested()`.
  - **Widgets:** `area_combo` (data `"vista"`/`"grade"`), `btn` (classificar área), `all_btn` (imagem toda), `progress` (0–100, oculto), `cancel_btn` (oculto), `alpha`, `clear_btn`, `status`.
  - **Métodos:** `scope() -> "vista"|"grade"`, `set_status(texto)`, `job_started()` (desabilita `all_btn`, mostra progress+cancel), `job_progress(msg,pct)`, `job_finished()`.
  - **Gotchas:** o painel não guarda estado de thread/job — o worker e a lógica de resume estão na janela; `job_progress` tem que rodar na thread da UI (a janela faz o marshalling).
  - **Onde editar:** novo escopo → `area_combo.addItem(label,key)` + tratar na janela. Novo modo → botão + sinal + fiação.

---

### `ui/similarity_panel.py` — curva + cartões de classe (barras de similaridade/predição, rotular, "+ classe")

Dois widgets separados: `CurveView` (plot NDVI/bandas, embaixo do mapa) e `SimilarityPanel` (cartões à direita). Atualizados pela janela: `curve_view.set_curve(...)` e `panel.set_scores(...)`. Constantes: `MONTHS` (Out…Set), `BAND_STYLE` (nome→(idx|None, hex); NDVI idx `None`), `PALETTE` (11 cores p/ classes novas).

- **`CurveView(QWidget)`** — `set_curve(curve (T,bands))` / `clear()`. Checkboxes por banda (NDVI marcado); `_replot` (NDVI = `to_ndvi`, linha grossa com marcadores; bandas = linhas finas; NaN mascarado). Sem sinais.
- **`ClassCard(QPushButton)`** — cartão custom-painted: barra de predição (cor da classe, topo) + similaridade (cinza, base) + chip + nome + `P%`/`S%` + bordas top/rotulado. `set_state(sim, pred, top, labeled, show_sim, show_pred)`. `paintEvent` desenha tudo (geometria `bx=0.46·w`, `bw=0.34·w`, nome `0.42·w`).
- **`SimilarityPanel(QWidget)`** — `__init__(self, classes: dict)`.
  - **Sinais:** `classChosen(str)` (cartão clicado → rotular), `removeRequested()` (só visível se rotulado), `classAdded(str)` (nova classe via "+ classe", emitido após criar o cartão + cor).
  - **Estado:** `classes` (nome→cor), `_sim`/`_pred` (nome→float 0..1), `_show_sim`/`_show_pred`, `_labeled`, `info` (linha resumo), `pred_toggle`/`sim_toggle`, `_cards`, `remove_btn`.
  - **Métodos:** `set_scores(sim_scores, pred_scores=None, novel=False, cycle_n=None, existing_class=None, quality=None)` — atualização principal; monta a linha `info` (`ciclos (vale): N`, `cleanlab: X.XX`, `modelo: <top> NN%`, ou `⚠ novidade`), mostra `remove_btn` se rotulado. `clear()`. Privados: `_make_card`, `_add_class_dialog` (QInputDialog + cor do `PALETTE` round-robin + emite `classAdded`), `_refresh` (percentuais int, define `top`).
  - **Gotchas:** scores são floats `[0,1]`, cartões mostram % inteiro. `top` prefere predição se o toggle está on e há predição, senão similaridade. Toggles chamam `_refresh` direto (não sinal).
  - **Onde editar:** rendering do cartão → `ClassCard.paintEvent`. Bit novo na info → `set_scores`. Cor de classe nova → `PALETTE`.

---

### `ui/map_view.py` — view de COG single-image (`CogMapView`) + fábricas de colorizer

Renderer de imagem única (predecessor do `TiledMapView`): lê uma janela padded por gesto e deixa o pyqtgraph escalar/pan entre leituras. Constantes: `pg.setConfigOptions(imageAxisOrder="row-major", background="#111111", antialias=False)`, `CLASS_COLORS` (paleta de 11 classes).

**Colorizers (puros, testáveis). Contrato:** entrada `arr[bandas,H,W]` (do `RasterSource.read_view`); saída `(H,W,4)` uint8 RGBA; nodata → alpha 0.
- `make_class_colorizer(colors: list[str], nodata=255)` — LUT `(256,4)`: índice `i` → cor `colors[i]`; `arr[0]` como índices de classe; `==nodata` → alpha 0. Raster de 1 banda.
- `make_scalar_colorizer(vmin, vmax, cmap="viridis", nodata=None)` — colormap matplotlib; normaliza `arr[0]` em `[0,1]`; não-finito e `==nodata` → alpha 0. Contínuo 1 banda.
- `make_rgb_colorizer(nodata=0)` — `arr[:3]` → `(H,W,3)` + alpha 255; alpha 0 onde soma das 3 bandas `== nodata*3`. ≥3 bandas.

- **`_ViewReader(QObject)`** — worker em `QThread` própria, handle `RasterSource` próprio. Sinal `read_done(id, arr, win)` (None em exceção). Slot `do_read`.
- **`CogMapView(QWidget)`** — mapa single-image. Sinais `pixelClicked(row,col)`, `pixelHovered(row,col)` (declarado, não emitido), `_requestRead`. Coalescing 1-leitura-por-vez (`_inflight`/`_pending`); `_reload` **pula** a leitura se a janela carregada ainda cobre o viewRect e a decimação serve (`ldec ≤ need_dec*1.6`) → pyqtgraph só escala/pan. `set_source`/`set_colorize`/`add_vector_layer`. `closeEvent` encerra a thread.
  - **Onde editar:** novo modo de render → nova `make_*_colorizer` (contrato). Hover readout → emitir `pixelHovered` num handler de mouse-move.

---

### `ui/tiled_map_view.py` — renderer de pirâmide tiled (slippy-map) gigapixel (o ATIVO)

Mantém uma **grade de tiles por nível de overview**: pan carrega só tiles novos; troca de nível deixa os antigos embaixo (z menor) enquanto os novos carregam. Cache LRU. É o renderer usado pela `AnnotatorWindow`.

- **`_TileWorker(QObject)`** — lê 1 tile em `QThread` própria, `RasterSource` próprio. Sinal `done(key, arr, win)` (None em exceção). Slots: `read(key, params, path)`; **`invalidate()`** — fecha o handle cacheado pra reabrir na próxima (o mesmo caminho pode ter **dado novo**, ex.: o `pred_<modelo>.tif` crescendo com o job).
- **`TiledMapView(QWidget)`** — Sinais `pixelClicked(row,col)`, `_reqTile`, `_invTile`. Const `TILE=256`, `MAX_TILES=320`. `factors = [1]+source.overviews`.
  - `set_source(source, colorize)` — **swap chave da janela**: bump `_gen`, limpa tiles/cache, **emite `_invTile`** (força reabrir — mesmo caminho pode ter dado novo), troca source/colorize/factors, limites generosos, `_update`.
  - `add_vector_layer(layer, pen=None)` (z=50). Privados: `_pick_level`, `_update` (grade de tiles cobrindo o viewRect + margem; emite `_reqTile` p/ os faltantes; LRU touch; prune; overlays), `_on_tile` (**descarta se `arr is None` ou `key[0] != _gen`** — gate de geração; z=`-L`), `_prune`, `_on_click`, `closeEvent`.
  - **Gotchas:** worker off-thread com handle próprio; colorização na thread da UI. **Gate de geração:** `_gen` no key + rejeição em `_on_tile` impede tiles velhos após swap. **`invalidate` fecha o handle** — essencial quando o arquivo muda no lugar (o batch escreve o pred incrementalmente); `set_source` sempre emite `_invTile` mesmo com caminho igual. Colorizers = mesmo contrato do `map_view.py`.
  - **Onde editar:** tamanho de tile/cache → `TILE`/`MAX_TILES`. Seleção de nível → `_pick_level`. Refresh ao vivo após reescrita sem trocar de source → chamar `set_source` com o mesmo `path`.

---

## Contrato de sinais (fiado pela `AnnotatorWindow`)

As seis views são construídas pela `AnnotatorWindow` e ligadas nos métodos `_wire`/`_build`.
Os painéis **nunca** mexem no store/mapa direto — emitem os sinais acima e expõem
setters/refresh que a janela chama depois da lógica: `set_scores`, `set_items`/`set_count`/
`select_pid`, `job_started`/`job_progress`/`job_finished`, `set_source`/`set_colorize`,
`set_curve`, `set_grid_checked`, `set_status`. **Vários construtores emitem sinais durante
o build** (notavelmente `MapToolbar.basemapChanged` no preselect e `modelChanged` enquanto a
janela preenche `model_combo`) — a janela liga depois ou usa `blockSignals`.
