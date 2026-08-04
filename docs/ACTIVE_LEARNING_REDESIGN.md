# SPEC — Annotation tool redesign: map-native active learning (SITS)

> Objetivo: evoluir o `sits.annotation` para um anotador **ENVI-like** (mapa navegável até o pixel) + **active learning fechado** (sugestão → anota → retreina → regenera), com componentes **modulares, testáveis e performáticos** em imagens de **~10 Gigapixels** (RIDE: 107.136 × 92.077, 4,78 m, 12 meses).
> Princípio mestre: **núcleo (domínio) desacoplado da UI**; UI fina; tudo pesado em background; nada carrega a imagem inteira.

---

## 1. Princípios de design (não-negociáveis)
1. **Separation of concerns** — `core/` (lógica de domínio, sem PyQt) ⊥ `ui/` (só apresentação) ⊥ `infra/` (IO, config, jobs). O core roda e é testado **sem GUI**.
2. **Interfaces (contratos) explícitos** — cada componente é uma classe abstrata (Protocol/ABC); implementações são plugáveis (trocar IT↔RF, métrica de similaridade, sampler) sem tocar no resto.
3. **Nunca carregar a imagem toda** — leitura por **janela** + **overviews** (COG) + **cache LRU**. O viewport é sempre ~1-2 Mpixels.
4. **Async by default** — treino/regeneração/IO pesado em **worker threads** (QThread) com sinais; UI nunca congela.
5. **Reprodutibilidade** — seeds fixos; cada modelo/mapa tem **versão**; rastrear quais rótulos → qual modelo → qual célula de mapa.
6. **Config-driven** — classes, caminhos, tamanho de grid, parâmetros de estratégia: tudo em config (sem hard-code).
7. **Testável** — unit tests no core (features, similaridade, sugestão, contagem de ciclo por **vale**). A regra de ciclo TEM teste.
8. **Honestidade estatística embutida** — held-out fora do loop; proporções vêm do **mapa**, não do set enviesado pelo AL.

---

## 2. Arquitetura em camadas

```
┌───────────────────────────────────────────────────────────────┐
│ UI (PyQt6) — fina, só apresenta + emite eventos                │
│  MapView(COG,pan/zoom,overlays,grid) · TimeSeriesPanel         │
│  SimilarityPanel(% por classe+exemplares+novidade) · Controls  │
└───────────────▲───────────────────────────────┬───────────────┘
                │ sinais/slots                   │ chamadas
┌───────────────┴───────────────────────────────▼───────────────┐
│ APPLICATION — orquestra casos de uso (sessão, loop AL)         │
│  SessionService · ActiveLearningOrchestrator · TaskRunner      │
└───────────────▲───────────────────────────────┬───────────────┘
                │                                │
┌───────────────┴────────────────────────────────▼──────────────┐
│ CORE (domínio, sem UI, 100% testável)                          │
│  FeatureExtractor · SimilarityEngine · SuggestionEngine        │
│  ModelService · WallToWallInference · AnnotationStore          │
│  LabelQuality(Cleanlab)                                         │
└───────────────▲───────────────────────────────┬───────────────┘
                │                                │
┌───────────────┴────────────────────────────────▼──────────────┐
│ INFRA — IO/config/log                                          │
│  RasterIO(COG,overviews,windowed) · TimeSeriesStore            │
│  TileCache(LRU) · ConfigLoader · Logger · VersionStore         │
└───────────────────────────────────────────────────────────────┘
```

---

## 3. Componentes (responsabilidade + contrato)

### CORE
**`FeatureExtractor`** — curva → vetor de features **interpretáveis** (sem z-norm).
- `extract(ts: TimeSeries) -> Features`
- Features: nº de ciclos **por VALE** (gap colheita+re-green-up, NÃO altura), profundidade do vale, **NDVI-seca (Jun-Set)**, amplitude (max−min), mín (vai a solo nu?), mês do pico principal, mês do 2º pico, nº de meses >0.4.
- ⚠️ **NÃO z-normalizar** (amplitude = sinal de intensidade+ciclo). Tratar NaN/nodata explicitamente.
- Tem teste: curvas-fixture (1c claro, 2c vale-fundo, safrinha-fraca, persistente) → contagem esperada.

**`SimilarityEngine`** — nova curva → similaridade **por classe**.
- `score(features, labeled_set) -> {class: pct}` + `nearest_examples(features, k) -> [(class, sample_id, sim)]`
- **k-NN** (não centróide): média das top-k de cada classe. **% ABSOLUTO** por classe (calibrado p/ 0-100, estável).
- `is_novel(scores) -> bool`: todas as classes < limiar → **novidade**.
- Interface plugável (métrica: feature-Euclid / DTW-sem-znorm).

**`SelectionEngine`** (Application) — propõe próximos pontos via `SelectionQuery` componível (estratégia × alvo × **`SpatialScope`** × layers × n). Consulta proba raster / similaridade / Cleanlab; filtra por layer; diversifica (FPS). **Detalhe completo em §4b.**

**`ModelService`** — treina/prediz (contrato; impl. IT, plugável p/ RF baseline).
- `train(dataset, spatial_cv=True, warm_start=None) -> ModelVersion(metrics)`
- `predict(X) -> (labels, proba)`
- Retorna métricas spatial-CV + reserva **held-out** fora do loop.

**`WallToWallInference`** — gera o mapa.
- `run_cell(model, cell_bounds) -> raster` (rápido, segundos) — **regen por grid clicável**.
- `run_full(model) -> raster` (produto final, consistente).
- `diff(old, new) -> change_mask` (pixels que mudaram → realce antes/depois).

**`AnnotationStore`** — persistência + versionamento (reusa o existente; add versionamento).
- `add/skip/uncertain/undo` · `export_dataset` · `version_tag(model_id, map_id)`.

**`LabelQuality`** — Cleanlab (reusa existente) → score de rótulo suspeito.

### APPLICATION
**`ActiveLearningOrchestrator`** — o loop fechado.
- `round()`: coleta lote → `ModelService.train` (rápido) → re-score **pool de candidatos** (segundos) → `SuggestionEngine` atualiza. Regen completo só sob demanda/no fim.
- Critério de parada: pool incerto/novidade seca + bacc estabiliza (loop-until-dry).

**`SessionService`** — estado da sessão (imagem, classes, modelo atual, versão).
**`TaskRunner`** — fila de jobs em background (treino, regen) + progresso → sinais p/ UI.

### UI (PyQt, fina)
**`MapView`** — COG via overviews+janela; overlays toggáveis (classificação, anotações, **candidatos**, **grid clicável**); pan/zoom até pixel; clique → emite `point_selected(row,col)`; clique em célula → `cell_selected(bounds)`.
**`SimilarityPanel`** — "% por classe" (barras) + **exemplares vizinhos** lado a lado + flag de **novidade**.
**`TimeSeriesPanel`** — curva (reusa `timeseries_plot`); marca picos/vales.
**`Controls`** — botões de classe, seletor de estratégia, "sugerir N", "regen célula", "retreinar", before/after toggle.

### INFRA
**`RasterIO`** (COG, overviews, windowed) · **`TimeSeriesStore`** (cubo tiled p/ buscar série do pixel no clique) · **`TileCache`** (LRU) · **`ConfigLoader`** · **`VersionStore`** (modelos+mapas versionados).

---

## 4. Algoritmos-chave (especificados)
- **Contagem de ciclo (CORRIGIDA):** nº de ciclos = nº de pulsos com **vale/prominência ≥ ~0.10** (gap colheita) e vigor mínimo (~0.40), **independente da altura**. Vigor (altura) = eixo de intensidade **separado**. *(corrige o erro que o especialista pegou; separa count de intensidade.)*
- **Similaridade:** features (acima) → distância → top-k por classe → % absoluto calibrado → novidade se todas baixas.
- **Sugestão:** consulta `w2w_class` + `w2w_proba` (já existem) → filtra por estratégia → top-N como marcadores (em contexto espacial).
- **Cell-regen:** janela do cubo tiled → inferência batch GPU → escreve a célula + diff vs anterior.
- **Loop AL:** lote → treino (warm-start opcional) → re-score pool → sugere; regen completo periódico/final.

## 4b. Seleção, Basemaps e Layers (o copiloto de coleta)

### `BasemapService` (Infra/Core) — basemaps toggáveis (COG + overviews)
| basemap | o que mostra | como gera |
|---|---|---|
| **MNF** | estrutura espaço-temporal, ruído suprimido (ENVI-like) | autovetores num **sample** → aplica **por-tile**; stretch consistente |
| **Fenológico** | lê os NOSSOS eixos direto (pico-NDVI / NDVI-seca / amplitude → RGB) | composição por-tile |
| **Classificação** | overlay da classe (toggle) | do `w2w_class` |
- Contrato: `available() -> [Basemap]` · `tile(name, window, level)`. MNF não é interpretável (ok p/ display); **fenológico é o melhor p/ anotar ciclo/intensidade**.

### `LayerManager` (Application) — overlays vetoriais/raster + filtro
- Aceita shapefile/geojson/gpkg (pivôs ANA, talhões SAM, MapBiomas) + rasters; toggle + estilo; **R-tree** → renderiza só feições do viewport (rápido c/ 30k pivôs).
- Contrato: `add(path)` · `features_in(viewport)` · **`mask(name) -> predicate`** (vira filtro espacial p/ seleção/regen).

### `SpatialScope` (tipo componível) — alimenta seleção **E** regen
```
SpatialScope = GLOBAL | GRID_CELL(s) (grid ajustável) | DRAWN_ROI | LAYER_MASK
```
- Compõem por interseção (ex.: `ROI ∩ dentro-de-pivô`); resolvem p/ máscara/bounds.
- **Mesmo escopo dirige `SelectionQuery.region` E `WallToWallInference`** → "trabalhar nesta área" = selecionar + regenerar + inspecionar aqui.
- **Densidade por escopo:** GLOBAL → pool **amostrado** (~100k-1M, esparso); LOCAL (célula/ROI) → **denso** (todo pixel ou passo) → pega manchas pequenas (ex.: mata seca). Passo auto pelo tamanho do escopo.

### `SelectionEngine` (Application) — substitui/absorve o `SuggestionEngine`
```
SelectionQuery { strategy, target?, scope, layer_filters[], n }
  → pool(scope, densidade) → pontua(fonte) → filtra(layers) → diversifica(FPS) → [Candidate{row,col,score,classe,porque}]
```
| estratégia | fonte | regra do candidato | ranqueia |
|---|---|---|---|
| **CLEAN_X** | raster **proba** | `argmax==X & proba_X>thr` | proba_X ↓ |
| **UNCERTAIN** | **proba** | margem `top1−top2 < δ` (top-2 ∈ {1c,2c} p/ mirar) | menor margem |
| **NOVELTY** | **similaridade** | todas as classes `< limiar` | menor sim. máx. |
| **SUSPICIOUS** | **Cleanlab** (set rotulado) | qualidade baixa / modelo discorda | menor qualidade |
| **COVERAGE** | features + set rotulado | mais longe (FPS) do rotulado | distância ↑ |
- Fontes: proba raster (clean-X/uncertain → **instantâneo**); pool de similaridade (novelty/coverage → segundos); Cleanlab (1×/rodada).
- **Diversificação obrigatória:** top-K por score → **FPS espacial** → N (senão amontoa num talhão).
- **`AUTO`** opcional: blend ponderado uncertain+novelty+coverage (padrão, sem escolher).

### Parâmetros setáveis (config/UI)
`novelty_threshold` (slider) · `grid_size` (5/10/20 km ou livre) · `density_step` (auto/manual) · `clean_proba_thr` · `uncertain_margin δ` · `batch_n` · pesos do `AUTO`.

### Fluxo (o copiloto)
```
[escopo: global / célula(s) / ROI desenhada]  → estratégia [+ alvo] [+ layer]  + n
  → marcadores no mapa (cor=classe), em contexto (MNF/fenológico + pivôs/SAM)
  → triage: clica bom → painel (curva + % classe + exemplares + novidade) → anota
  → [mesmo escopo] regen + before/after/diff
  → lote anotado → retreina → re-pontua pool → seleção atualiza
```

## 4c. UI do painel + os dois modos

### Painel de anotação (keyboard-first)
Regiões (cima→baixo): **contexto** (r,c · dentro-de-pivô · talhão SAM#) → **curva NDVI** (▲picos ▼vales, **seca sombreada**, toggle banda NDVI/RGB/NIR, overlay do exemplar) + chip do pixel → **barras de similaridade** (% por classe, sugerido destacado, banner de **novidade**) → **exemplares** (#1 de cada classe, clicável p/ sobrepor) → **modelo + Cleanlab** → **picker de classe** (botões = teclas 1-9, cada um mostra a própria %, sugerido destacado).
**Teclas:** `1-9` classe · `Enter` confirma sugerido · `S`/`U` skip/incerto · `N` nova classe · `←` desfaz · setas próximo/anterior · auto-avança após anotar. Curva default = **NDVI** (lê ciclo/intensidade) + toggle de bandas.

### Modo ANOTAR (novos pontos)
Seleção (SelectionEngine) → painel → anota → próximo. (Fluxo do §4b.)

### Modo REVISAR (`ReviewMode`, Application) — pontos JÁ rotulados
- **Fila ordenada por SUSPEITA:** (1) Cleanlab qualidade ↓ · (2) modelo discorda do rótulo · (3) similaridade discorda (rótulo ≠ classe de maior %) · (4) **re-check do critério de VALE** (flag onde o ciclo muda com a regra corrigida).
- Mesmo painel, mostra o **rótulo EXISTENTE** + os sinais. Ações: `Enter` manter · tecla=corrigir · `Del` remover.
- **Filtros:** por classe · por tipo de discordância · por limiar de suspeita.
- → **Conserto sistemático** (o caso do especialista): filtra "discordância de vale" → safrinhas-fracas-1c→2c em lote.
- Contrato: `queue(filters) -> [LabeledPoint ranked]` · `resolve(point, keep|reassign|delete)`.

## 5. Performance
- **COG tiled (256) + overviews** no basemap E no cubo de entrada (uma conversão; destrava display E cell-regen).
- Leitura **por janela no nível de overview certo**; **TileCache LRU**.
- Inferência **batch na GPU**; cell-regen em segundos.
- Série do pixel buscada **sob demanda no clique** (não pré-carregar).

## 6. Reuso vs novo
| Reusar | Reescrever/criar |
|---|---|
| GUI/controllers/widgets, `store`, `samplers`, Cleanlab, helper model | `similarity_service` (k-NN, sem z-norm, % abs, novidade), `FeatureExtractor`, `SuggestionEngine`, `WallToWallInference`(cell+full+diff), `ActiveLearningOrchestrator`, `MapView`(COG), `SimilarityPanel`, `RasterIO`(COG), `VersionStore` |

## 7. Decisões/caveats embutidos
- **Sem z-norm** · **ciclo por vale** · **k-NN não centróide** · **% absoluto + novidade**.
- **Proporções vêm do mapa**, não do set rotulado (viés do AL); **class weights** no treino.
- **Held-out fora do loop** (medida honesta = S8) · **versionamento** por rodada · **re-ancorar nos textbook** (anti-drift).

## 8. Stack
Python · PyQt6 · rasterio+GDAL (COG/overviews) · numpy/scipy · PyTorch (IT) · scikit-learn · cleanlab · loguru · pytest · (opcional pyqtgraph p/ MapView rápido).

## 9. Implementação em FASES
- **F0 — Enabler:** conversão COG + overviews — `BasemapService` (**MNF** + **fenológico** + classificação) + **cubo de entrada tiled**. *(destrava display E cell-regen)*
- **F1 — MapView + Layers:** pan/zoom até pixel, `LayerManager` (pivôs/SAM/MapBiomas, R-tree), `SpatialScope` (grid clicável ajustável + ROI desenhada), clique→série. *(núcleo da UX)*
- **F2 — Núcleo de similaridade:** `FeatureExtractor` + `SimilarityEngine` (corrigidos) + `SimilarityPanel` (% por classe + exemplares + novidade) + **testes**.
- **F3 — Seleção:** `SelectionEngine` (5 estratégias + `SpatialScope` + **layers-como-filtro** + densidade por escopo + params setáveis) + marcadores em lote + clique manual.
- **F4 — Loop AL + Revisão:** `ActiveLearningOrchestrator` (lote→retreino→re-score) + `WallToWallInference.run_cell` (clicável) + before/after/diff + `TaskRunner` + **`ReviewMode`** (fila por suspeita: Cleanlab/discordância/critério-vale).
- **F5 — Produto:** `run_full` consistente + `VersionStore` + export.
- **Transversal:** config, logging, testes, async — desde a F1.

## 10. Testes (mínimo)
- `FeatureExtractor`: fixtures de curva → contagem por vale correta (incl. safrinha-fraca→2c, persistente→1c, declínio→1c).
- `SimilarityEngine`: classe óbvia → % alto + margem; outlier → novidade.
- `SuggestionEngine`: cada estratégia retorna candidatos coerentes com os rasters.
- `WallToWallInference.run_cell`: célula == subset do `run_full` (consistência).
- `SelectionEngine`: cada estratégia retorna candidatos coerentes c/ a fonte; `SpatialScope` (grid/ROI/layer) restringe corretamente; FPS espalha o lote.
- `LayerManager.mask()`: máscara da layer ∩ pool == só feições dentro.

## 11b. Publicação & Performance (tool standalone — rotular pixels)
> Meta: publicar separado (JOSS/SoftwareX). Nicho: **rotulagem nativa de SITS + navegação GIGAPIXEL (COG) + active-learning por similaridade + 2 modos (anotar/revisar)** — não existe pronto.

**Performance (suavidade):**
- **Leitura ASSÍNCRONA** (worker thread c/ handle próprio + latest-wins) → UI nunca congela, mesmo em disco lento / COG remoto. *(reads já são ~0,017s local; isto é robustez.)*
- **Cache LRU** de janelas/tiles → re-pan e zoom-return instantâneos.
- **LOD vetorial:** centroides (scatter) no zoom-out, contornos no zoom-in.
- prefetch dos tiles vizinhos (opcional).

**Generalização (genérico, não amarrado à RIDE):**
- **Config-driven:** qualquer raster + classes (nome/cor/atalho) + fonte de série — zero path hardcoded.
- CRS/transform genéricos (overlay reprojeta); fonte de série plugável; export padrão (npz/GeoJSON/CSV).

**Empacotamento (JOSS):**
- `pyproject.toml` + **entry-point CLI** (`pixel-annotator img.tif --classes classes.yml`), pip-instalável.
- **Testes + CI** (RasterSource, colorizers, VectorLayer, mapeamento de clique — headless, como já validado) + GitHub Actions.
- **Docs + tutorial** + **dados de exemplo** (raster PEQUENO embutido) + licença + CITATION.cff.
- (opcional) extrair como repo/pacote próprio.

## 11. Decisões em aberto (debater antes de implementar)
- **Basemap default:** MNF (estrutura) vs fenológico (lê os eixos). *(tendência: fenológico default, MNF alternativo.)*
- **Densidade local:** todo pixel vs passo (1 a cada k) p/ ROIs grandes — auto pelo tamanho?
- **Grid:** faixa fixa (5/10/20 km) ou livre.
- **`AUTO` blend:** pesos uncertain/novelty/coverage.
- **Snap-to-SAM:** já no paper 1 (object) ou só visual agora (paper 1 = pixel; object = paper 2).
- **Tamanho do pool global** (100k vs 1M): velocidade × cobertura.
- ✅ **RESOLVIDO — MapView lib: `pyqtgraph`** (rápido, LOD/downsampling embutido, encaixa no PyQt).
</content>
