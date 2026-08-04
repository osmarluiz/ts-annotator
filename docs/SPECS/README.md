# TSA — Especificação detalhada do código

> Mapa de referência do `ts-annotator` para saber **onde editar**. Índice + fluxos +
> modelo de threads + convenções, com links para a referência por subsistema.
> Fonte da verdade é o código; este doc é mantido à mão — se divergir, o código vence.

## Como o app está organizado

```
src/ts_annotator/
  core/   # domínio puro — SEM PyQt, testável headless
  ui/     # apresentação (PyQt6 + pyqtgraph) — "dumb views" que emitem sinais
  app/    # camada de aplicação (independente de projeto): contexto, workers, tema, config
examples/ # launchers finos (fiação específica do projeto RIDE/AOI)
```

Referência detalhada por subsistema:
- [`spec_core.md`](spec_core.md) — dados/raster + ML/active-learning (`core/`)
- [`spec_ui.md`](spec_ui.md) — painéis, mapa tiled, colorizers (`ui/`)
- [`spec_app.md`](spec_app.md) — `AnnotatorWindow`, `AppContext`, workers, tema, config, launcher

> ⚠️ **Drift (refactor fase 3 + controller, jul/2026):** `TrainPanel` (`ui/train_panel.py`),
> `AnnotatePanel` (`ui/annotate_panel.py`) e `MapCanvas` (`ui/map_canvas.py`) foram
> extraídos da `AnnotatorWindow`; grid tool, desenhar-retângulo, overlay de basemap e os
> itens pyqtgraph vivem no `MapCanvas`. TODA a lógica descrita em `spec_app.md` sob
> `AnnotatorWindow` (on_click/on_label/navigate, propose, review, modelo, classify,
> treino, workers) mudou-se para o **`AnnotatorController`** (`app/controller.py`) — os
> corpos dos métodos são os mesmos; a janela virou só montagem + fiação + aliases.

## A fronteira (onde o projeto entra)

O **launcher** (`examples/run_annotation_demo.py`) conhece o projeto (caminhos de
raster, classes/cores, conjunto de referência, fonte da série temporal). Ele monta um
**`AppContext`** (`app/session.py`) e entrega à **`AnnotatorWindow`**, que é genérica —
opera só sobre o contexto, nunca sobre caminhos fixos.

```
build_context()  ──►  AppContext  ──►  AnnotatorWindow(ctx)
 (dados do projeto)    (serviços)       (UI + lógica genéricas)
```

Para portar a outra área/projeto: **só o launcher muda**. Nada em `core/`, `ui/`, `app/`.

## Convenções que valem para o código inteiro

- **Curva de um pixel:** array `(meses, bandas) = (12, 4)`, bandas = B, G, R, NIR em
  **reflectância** (escala /10000). `nodata` (65535) vira `NaN`.
- **Modelo:** consome `(N, 5, 12)` — as 4 bandas + NDVI como 5º canal (via
  `trainer.build_X`). NDVI = (NIR−Red)/(NIR+Red).
- **Coordenadas:** `row`/`col` em pixels da grade da imagem de classe (`class_src`). No
  pyqtgraph, **x = col, y = row** (`setRange(xRange=col…, yRange=row…)`, `pos=(col,row)`).
  Toda leitura de curva/store usa **row primeiro** (`read_curve(row, col)`,
  `find_at(row, col)`).
- **Validade:** um pixel só é "classificável/proponível" se a curva é **inteira finita**
  (`np.isfinite(curve).all()`) — o mesmo critério no propor, no classify e no treino.
  Pixel com mês nodata é rejeitado (evita NaN em feature/similaridade/modelo).
- **Índice de classe:** o valor gravado no raster de predição é o **índice na lista
  `labs`** do modelo (a ordem do treino), não um código global. As cores vêm do sidecar.

## Estado da `AnnotatorWindow` (dicts de instância)

Toda a interação vive na janela; o estado mutável está em poucos dicts:
- `self.state` — interação geral: `last` (row,col,curve do ponto corrente), `size`
  (tamanho do marcador), `review` (índice de navegação no review), `pseed` (semente do
  propor, incrementa a cada clique), `grid`/`cells` (grade e células selecionadas),
  `ref` (mostrar referência), `last_src` ("map"|"sugg"), `sugg`/`sugg_cur` (sugestões e
  índice atual), `draw`/`pan` (ferramentas de desenho/pan).
- `self.tstate` — modelo ativo: `net`, `mu`, `sd`, `labs` (None se sem modelo).
- `self.pred_state` — `alpha` do overlay de predição.
- `self.overlay_state` — `label`/`alpha` do basemap sobreposto ("sobre:").
- `self.div_cache` — cache de features do dataset p/ a métrica de diversidade
  (`n` = tamanho quando calculado, `feats` = matriz).
- `self._training` / `self._batch_running` — exclusão mútua de GPU (treino × classify).

## Os fluxos (ponta a ponta)

### 1. Clique no mapa → curva → painel
`TiledMapView.pixelClicked(row,col)` → `on_click`: se há ponto no raio de snap →
`select_point`; senão lê a curva (`ts.read_curve`), **rejeita se não for finita**,
extrai features (`fx.extract`), pontua similaridade (`eng.score`) e predição
(`predict_point`), pinta o painel (`panel.set_scores`) e guarda `state["last"]`.

### 2. Rotular → store
Cartão de classe → `panel.classChosen` → `on_label(cls)`: `store.add_or_update(row,col,
cls,curve)` (dedup por `tol_px`, id = max+1, grava JSON atômico) → atualiza marcadores +
lista de review. Se veio de sugestão, avança pra próxima.

### 3. Propor pontos (active learning)
`do_propose`: escopo (`_propose_area`: vista/grade/retângulo/imagem-toda) → métrica/ordem
(do `goal_combo`, ver `config.GOALS`) → `sel.propose_many`. A engine amostra um pool
(`candidates`: **um bloco decimado** se o TS tem overviews, senão n leituras de 1px),
pontua pela métrica (`metrics.compute`) e faz **seleção gulosa diversa** (`_greedy_diverse`
— espalha no espaço e em features). Barra de progresso determinada. Selecionar uma
sugestão relê a curva **full-res exata** no pixel.

### 4. Treinar (GPU, off-thread)
`do_train` → `TrainWorker` (na `tthread`): filtra pontos com curva finita → spatial-CV
estratificada (`trainer.spatial_cv`) → cleanlab (`label_scores`) → fit final →
`on_train_done` grava `it_latest.pt`, escreve `_clean`/`_pred` de volta nos pontos, e
recarrega o modelo. Botão de classify fica desabilitado enquanto treina.

### 5. Classificar área (preview overlay)
`classify_scope`/`run_prediction` → `_classify_area`: lê bloco (decimado via overview se
existir), infere, pinta o `pred_overlay` (com transparência). Se `step==1`, **persiste**
no raster por-modelo (`_persist_area`). Barra de progresso em fases.

### 6. Classificar imagem toda (job de lote)
`classify_all` → `ClassifyAllWorker` (na `cthread`): tiles 1024², handles próprios,
grava `pred_<modelo>.tif` (esparso, tiled, colormap) + sidecar retomável. Cancelável;
re-treino do modelo (mtime muda) reseta o raster. Overviews no fim.

### 7. Camada "Predição (modelo)"
`_refresh_pred_basemap` aponta um basemap dinâmico pro `pred_<modelo>.tif` do modelo
ativo (cores do sidecar). Atualiza ao trocar modelo, persistir área ou terminar o job.
Modelo re-treinado → oculta (não mostra predição velha); sem modelo → some.

## Modelo de threads

Três threads Qt, cada uma com **handles de raster próprios** (rasterio não é
thread-safe):
- **UI thread** — tudo da janela + leitura de curva no clique + `_classify_area`.
- **`tthread`** (`TrainWorker`) — treino na GPU. Sinais `done`/`prog`.
- **`cthread`** (`ClassifyAllWorker`) — job de lote. Sinais `done`/`prog`; `cancel` é
  conexão **Direct** (o event loop da thread fica bloqueado no job, só seta um flag).
- **Tile worker** (dentro de `TiledMapView`) — lê tiles do basemap; `set_source` emite
  `invalidate` pra fechar o handle cacheado (o mesmo caminho pode ter dado novo).

Exclusão mútua de GPU: `do_train` recusa se `_batch_running`; `classify_all` recusa se
`_training`. Botões espelhados.

## Onde editar para mudar X

| Quero… | Edite |
|---|---|
| Adicionar uma **classe** nova | launcher `CLS_COLORS`/`CLASS_COLORS`; runtime via "+ classe" → `on_class_added` |
| Adicionar um **objetivo de active learning** | `app/config.py` `GOALS` (+ métrica em `core/metrics.py` se nova) |
| Mudar os **limiares de ciclo/fenologia** | `core/features.py` (`peak_h`, `prominence`, `DRY_MONTHS`, `green_thr`) |
| Trocar a **arquitetura do modelo** | `core/trainer.py` (`IT`/`_IM`, `build_X` p/ canais) |
| Mudar o **critério de diversidade** das sugestões | `core/selection.py` `_greedy_diverse` (`min_px`, `div_thr`) |
| Mudar o que um **clique em sugestão** faz | `AnnotatorWindow.on_sugg_select` |
| Adicionar um **controle na toolbar** | `ui/map_toolbar.py` (sinal) + fiação em `AnnotatorWindow._build_toolbar` |
| Adicionar um **filtro no review** | `ui/review_panel.py` + `AnnotatorWindow.refresh_review_list` |
| Adicionar uma **aba** na navegação | `AnnotatorWindow._build_<x>` + `_build_nav` + `_assemble` |
| Adicionar um **job de background** novo | espelhe `ClassifyAllWorker` + fiação em `_init_model_and_thread` |
| Apontar o app pra **outra área/dados** | só o launcher `run_annotation_demo.py` (`AOI_DIR`, `ORDER`, basemaps, `pred_dir`) |
| Mudar **cores/colormap** de um raster | `ui/map_view.py` (colorizers) + `cls_colors` |
| Mudar a **persistência** do raster de predição | `core/prediction_raster.py` (`ensure`/`write_block`/sidecar) |

## Dados (fora do repo)

- Cubo AOI: `E:\sits_aoi_crop\cube\ride_{202210..202309}_aoi.tif` — 12 meses, 4 bandas
  uint16, nodata 65535, mesma grade (offset 0). Overviews `[2,4,8,16,32]` (average).
- Basemaps: `E:\sits_aoi_crop\{input,products}\basemap_*_aoi.tif`.
- Predições: `E:\sits_aoi_crop\predictions\pred_<modelo>.tif` + `.json` (sidecar).
- Pontos: `examples/points.json` (trabalho), `examples/points_antigos.json` (referência DF).
- Modelos: `examples/models/*.pt` (`it_latest.pt` = ativo).
