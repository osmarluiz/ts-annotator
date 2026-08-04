# `core/` — domínio puro (sem PyQt, testável headless)

> Referência detalhada da camada de domínio. Convenções globais em [`README.md`](README.md).
> Duas famílias: **dados/raster** (leitura, store, raster de predição) e **ML/active-learning**
> (features, similaridade, seleção, métricas, treino).

---

## Dados / raster

### `core/timeseries.py` — lê a curva (12 meses × bandas) de um pixel/bloco sob demanda de N rasters mensais, com offset por mês e nodata→NaN.

- **`TimeSeriesSource`** — a fonte do clique→curva. Lê curvas de reflectância `(n_meses, n_bandas)` de uma lista de rasters mensais, cada um opcionalmente deslocado por offset de linha/coluna pra alinhar grades. **Construído por:** launcher (`build_context`), 1 handle rasterio aberto por mês. **Atributos:**
  - `paths: list[str]` — caminhos mensais (ordem = ordem dos meses; n=12).
  - `bands: list[int]` — índices 1-based; default `[1,2,3,4]` = B/G/R/NIR.
  - `row_offsets`/`col_offsets: list[int]` — deslocamentos de pixel por mês; default zeros.
  - `nodata: int` — sentinela de ausente; default `65535` (igualdade exata após cast float).
  - `scale: float` — divisor DN→reflectância; default `10000.0`.
  - `_ds: list[rasterio.DatasetReader]` — handles abertos no `__init__`.
  - `has_overviews: bool` — `True` só se **todos** os meses têm overview na banda 1. GOTCHA: um mês sem overview → `False` → o app trata leitura decimada como cara (força full-res).
- `read_curve(row, col) -> np.ndarray (12,4)` — curva de 1 pixel; aplica offset por mês; mês fora dos limites ou nodata fica `NaN`; lê `Window(c,r,1,1)`, `/scale`. **Sem overview** (sempre 1px full-res).
- `read_block(row0, col0, h, w, step=1) -> np.ndarray (Hs,Ws,12,4) float32` — bloco de todos os meses decimado por `step` (`Hs=max(1,h//step)`). **PONTO CRÍTICO DE PERF:** se a janela está `in_bounds`, usa `read(out_shape=...)` normal → rasterio puxa do **overview** (preview rápido); se cruza a borda (offset por mês pode empurrar), cai em `boundless=True` que **ignora overview e lê full-res** (era a causa do congelamento).
- `close()` — fecha os handles.
- **Onde editar:** bandas/escala nos defaults; nodata (65535); alinhamento nos offsets; comportamento overview×full-res no ramo `in_bounds` de `read_block`.

### `core/raster_source.py` — leitor windowed de COG (sem GUI) com seleção automática de overview e cache LRU; nunca carrega a imagem inteira.

- **`RasterSource`** — wrapper rasterio que lê uma janela em pixel decimada ao tamanho de saída (rasterio auto-escolhe overview quando `out_shape` < janela), com cache LRU por geometria. **Construído por:** camada de mapa, 1 por basemap. **Atributos:** `path`, `_ds`, `height`/`width`/`count`, `nodata`, `dtype`, `overviews: list[int]`, `transform`, `crs`, `_cache: OrderedDict`, `_cache_size=96`, `cache_hits`/`cache_miss`.
- `read_view(x0, y0, w, h, out_w, out_h, resampling=nearest) -> (arr, win) | (None,None)` — leitura de viewport. `x0,y0` = canto full-res (x=col, y=row); clampa aos limites; retorna `(None,None)` se janela vazia; nunca faz upsample; chave de cache `(x0c,y0c,ww,hh,ow,oh,resampling)`. Retorna `arr (count,oh,ow)` no dtype original (sem escala/nodata) + `win` (janela full-res usada, pra posicionar). GOTCHA: **não é thread-safe** (mexe no cache sem lock); nodata não é aplicado.
- `read_pixel(row, col) -> (count,)|None` — 1 pixel full-res (arg order row,col). Sem chamadores hoje.
- `close()` / `__del__` — fecham o handle.
- **Onde editar:** capacidade do cache (`96`); resampling default; aplicar nodata/escala após o `read`; adicionar lock pra thread-safety.

### `core/annotation_store.py` — conjunto JSON de pontos rotulados com dedup por proximidade (1 local = 1 rótulo), query de viewport e escrita atômica.

- **`AnnotationStore`** — store persistente; re-rotular perto de um ponto **atualiza** em vez de duplicar. **Construído por:** launcher, 1 por dataset+período. **Atributos:** `path`, `transform` (opcional; se setado, pontos ganham `x,y`), `crs`, `period="default"`, `tol=3` (px, de `tol_px`), `points: list[dict]`, `_rc: (N,2)` (índice espacial).
  - Schema do ponto: `row,col:int`, `period`, `class` (rótulo — string/valor do chamador, **não** o índice `labs`), `source`, `id:int`; opcionais `x,y:float` (geo no centro do pixel), `curve` (lista aninhada, 4 casas), `confidence`.
- `load()` / `save()` — `save` escreve `{format_version:"1.0", crs, period, points}` **atômico** (mkstemp + `os.replace`). Sem `indent`. GOTCHA: sem `fsync` (seguro contra crash de processo, não contra queda de energia).
- `_reindex()` — reconstrói `_rc` (chamado após toda mutação).
- `find_at(row, col, tol=None) -> dict|None` — ponto mais próximo dentro da tolerância (default 3 px).
- `query(x0,y0,x1,y1) -> list` — pontos no box (x=col, y=row), inclusivo. Pros marcadores.
- `add_or_update(row, col, cls, curve=None, source="manual", confidence=None) -> (pt, updated)` — adiciona ou atualiza o vizinho de `find_at`; `id = max(id)+1`; geo no centro `transform*(col+0.5,row+0.5)`; curva arredondada 4 casas. Efeito: `_reindex()` + `save()` **a cada chamada** (O(n) por rótulo).
- `bulk_add(pts)` — insere muitos (semear); 1 reindex + 1 save; **não faz dedup nem atribui id**.
- `remove_at(row, col) -> bool`, `__len__`.
- **Onde editar:** raio de dedup (`tol_px=3`); schema/versão em `save`; convenção geo (centro vs canto); atribuição de id; precisão da curva.

### `core/prediction_raster.py` — GeoTIFF uint8 de classificação por-modelo + sidecar JSON de proveniência/progresso; retomável e detecta re-treino.

- **`PredictionRaster`** — 1 raster por modelo: `pred_<stem>.tif` (full-image, esparso, tiled uint8) + `pred_<stem>.json` (proveniência: modelo+mtime, classe→índice, cores; progresso `done_tiles`). **Valor do pixel = índice na lista `labs`** do modelo; `255` = sem predição. **Construído por:** `AnnotatorWindow` (`_persist_area`, `classify_all`, `_refresh_pred_basemap`). **Const/atributos:** `NODATA=255`, `dir`, `model_path`, `tile=1024` (≠ block do GeoTIFF que é 512), `tif`, `meta_path`.
- `load_meta() -> dict|None`, `_write_meta()` (atômico), `save_progress(done_tiles, complete=False)` — grava tiles feitos (torna retomável).
- `ensure(width, height, transform, crs, labs, colors) -> "kept"|"created"|"reset"` — garante raster+sidecar coerentes com o modelo **atual**. `"kept"` só se sidecar+tif existem E `labs`, `model_mtime`, `width/height` batem (preserva job em andamento). Senão (re)cria zerado (esparso, tiled 512, deflate, BIGTIFF, colormap das classes; 255→transparente). GOTCHA: mtime do modelo mudou (re-treino) → `"reset"`. Verificado empiricamente: blocos não escritos leem **255 (nodata) → transparente** (não classe 0).
- `write_block(arr, row0, col0)` — grava bloco uint8 em `Window(col0,row0,...)` (abre `r+` por chamada).
- `open_writer() -> ds (r+)` — handle longo pro job de lote (o worker fecha).
- `build_overviews(factors=(2,4,8,16,32))` — overviews nearest (preserva índice de classe) no fim do job.
- **Onde editar:** `NODATA` (255); block size (512) / `tile` (1024); compressão/esparso/BIGTIFF no `prof`; detecção de reset (mtime/labs/wh); cores/colormap; níveis de overview.

### `core/vector_layer.py` — carrega um vetor como overlay, projeta anéis pra pixel e devolve polilinhas (x,y) separadas por NaN via STRtree.

- **`VectorLayer`** — overlay vetorial cujos anéis externos são pré-projetados pra **pixel** e indexados com `STRtree` (shapely 2.0, sem `rtree`). **Atributos:** `polys_px: list[(cols,rows)]` (multipolígonos explodidos; buracos ignorados), `tree: STRtree|None`, `n`.
- `__init__(path, transform, crs)` — lê com GeoPandas, reprojeta se `crs`, inverte `transform` (`inv=~transform`), projeta cada anel externo. Leitura inteira em memória no construtor. GOTCHA: se o `transform` do raster muda, reconstruir.
- `query_outlines(x0,y0,x1,y1) -> (xs, ys)` — polilinhas dos anéis cujo bbox intersecta o viewport; concatena com separador `[nan]` (pronto pro `PlotDataItem`). GOTCHA: query é por bbox (pode incluir anéis cujo bbox sobrepõe mas geometria não).
- **Onde editar:** renderizar buracos (só `exterior` hoje); projeção pixel (`inv`); culling em `query_outlines`; formato de saída.

---

## ML / active-learning

### `core/features.py` — features fenológicas interpretáveis; contagem de ciclo por prominência (vale), não por altura do pico.

- **`DRY_MONTHS = (8,9,10,11)`** — índices na curva de 12 meses (0 = Out; 8..11 = Jun/Jul/Ago/Set). Média de NDVI na seca.
- `to_ndvi(curve) -> (12,)` — aceita `(12,4)` reflectância **B,G,R,NIR** ou `(12,)` NDVI já pronto. `NDVI = clip((NIR−Red)/(NIR+Red+1e-6), -1, 1)`, `Red=c[:,2]`, `NIR=c[:,3]`. GOTCHA: ordem de banda fixa.
- `count_cycles(ndvi, peak_h=0.40, prominence=0.10, max_c=3) -> (n, pos, heights, proms)` — nº de ciclos = nº de picos com **prominência (vale) suficiente**, não altura → safrinha fraca-mas-real conta. Padding com zeros nas pontas; `find_peaks(height=peak_h, prominence=prominence)`; `n = clip(len(pk), 1, max_c)` (sempre ≥1, ≤3).
- **`FeatureExtractor`** — vetor de 8 features. **NÃO** z-normaliza a curva (amplitude é sinal). **`NAMES`** (ordem exata do `extract`):
  0. `n_cycles` — nº de ciclos (1..3).
  1. `dry_ndvi` — média NDVI na seca (irrigação/perene).
  2. `amplitude` — `nanmax−nanmin` (vigor).
  3. `peak_month` — mês do pico máximo.
  4. `peak_val` — altura do pico máximo.
  5. `second_peak` — altura do 2º pico qualificado (senão 0).
  6. `valley_depth` — prominência máxima (profundidade do vale).
  7. `n_green` — nº de meses com `NDVI ≥ green_thr`.
  - Params: `peak_h=0.40`, `prominence=0.10`, `green_thr=0.40`.
  - `extract(curve) -> (8,)` — se curva toda não-finita → zeros; **rede de segurança final `np.nan_to_num(f, 0.0)`** (nunca devolve NaN; NaN em feature envenenaria a distância na similaridade). GOTCHA: `extract` é tolerante e zera; `SelectionEngine` rejeita a montante com `.all()`.
  - `extract_dict(curve) -> dict` — nome→valor.
- **Onde editar:** limiares de ciclo (`peak_h`/`prominence`/`max_c`); adicionar feature → `NAMES` **e** `extract` na mesma ordem (mudar o tamanho invalida referências de similaridade salvas); `DRY_MONTHS`; ordem de banda em `to_ndvi`.

### `core/similarity.py` — similaridade kNN por classe em features padronizadas; similaridade **absoluta** + novidade.

- **`SimilarityEngine`** — kNN (não centróide) nas features, padronizadas. Retorna similaridade **absoluta** `[0,1]` por classe (não softmax) → permite detectar novidade. **Params:** `k=3`, `novelty_thr=0.45`. Estado (via `load`): `_mean`/`_std` (padronização, `+1e-6`), `_classes` (classe→matriz `(N,F)` padronizada), `_ids`.
- `load(items)` — `items=[(features, class[, id])]`. `_mean/_std` calculados **uma vez sobre TODOS os itens** (não por classe). GOTCHA: stats congeladas no load.
- `_kdist(M, z) -> float` — `sort(||M-z||)[:k].mean()`. GOTCHA: classe com <k exemplares → média dos disponíveis (sem erro).
- `score(features) -> {classe: 1/(1+kdist)}` — distância→similaridade `(0,1]`.
- `nearest(features) -> {classe: (id, sim)}` — exemplar mais próximo por classe.
- `ranked(features)`, `is_novel(scores) -> bool` (`max(scores) < 0.45`).
- **Onde editar:** `k`/`novelty_thr`; mapeamento `1/(1+d)`; centróide em vez de kNN (`_kdist`); padronização por classe (em `load`).

### `core/metrics.py` — métricas escalares de active learning a partir de dicts sim/pred por classe.

- `_dist(d, classes)` — dict→vetor de probabilidade normalizado (missing→0).
- `confidence(pred)` = max prob (baixa=incerto). `margin(pred)` = top1−top2 (baixa=ambíguo). `entropy(pred)` = Shannon **ln** da dist normalizada (alta=incerto). `similarity(sim)` = max sim (baixa=novidade). `novelty(sim)` = 1−max sim. `disagreement(sim,pred)` = **TV distance** `0.5·Σ|dist(sim)−dist(pred)|` na união de classes (0=concordam).
- `compute(sim, pred, name) -> float` — dispatch; `pred/sim` vazio → 0.0; nome desconhecido → 0.0.
- **Onde editar:** nova métrica → escreva `def x(sim,pred)` + ramo em `compute` (o `else` de `rank` já pega); métrica que precisa de `target`/`store_feats` → ramo em `SelectionEngine.rank`. Entropia usa ln (trocar por log2 pra bits).

### `core/selection.py` — amostragem de candidatos, pontuação, ranking e proposta top-k **diversa** (active learning).

- `_margin(scores)` — utilitário top1−top2.
- **`SelectionEngine(ts, fx, sim)`** — propõe pontos a rotular numa região.
- `_eval(row, col) -> dict|None` — lê 1 curva full-res; **rejeita se `not isfinite(curve).all()`** (mesmo critério do classificador); senão `{row,col,curve,features,scores}`.
- `candidates(x0,y0,x1,y1,n=80,seed=0,progress=None)` — despacha: TS com `read_block` → `_candidates_block` (rápido); senão `_candidates_pointwise` (fallback, testes).
- `_candidates_pointwise(...)` — amostra `n` pixels aleatórios e lê 1 a 1 (lento, n×12 leituras). `progress(i+1,n)`.
- **`_BLOCK_NATIVE_CAP = 8_000_000`** — teto de px full-res que um bloco decimado pode tocar **sem overview**; acima disso cai no ponto-a-ponto (senão leria a imagem toda).
- `_candidates_block(...)` — caminho rápido: se `not has_overviews and w*h > CAP` → ponto-a-ponto; senão `step = max(1, sqrt(w*h/pool))`, `pool=max(n*40,20000)`; `cube = read_block(...)`; `valid = isfinite(cube).all(axis=(2,3))`; amostra `min(n,#válidos)` células; coords na grade decimada (curva exata é relida ao **selecionar** a sugestão). `progress(1,2)`/`(2,2)`.
- `_diversity_setup(store_feats)` — matriz padronizada dos pontos rotulados (usa `sim._mean/_std`, ou padroniza pelos próprios se sem referência).
- `rank(x0,y0,x1,y1, metric="disagreement", order="desc", target=None, n=80, seed=0, exclude=None, predict_fn=None, store_feats=None, progress=None)` — amostra, calcula `pred=predict_fn(curve)` e `_m` por candidato, ordena. Roteamento de métrica: `"class"` → prob do target (ou sim, fallback −1); `"diversity"` → distância mín. em features ao rotulado (1.0 se nada rotulado); senão `metrics.compute`.
- `propose(...) -> dict|None` = `rank(...)[0]`.
- **`_greedy_diverse(ranked, k, area, min_px=None, div_thr=0.5)`** — escolhe `k` gulosamente na ordem da métrica, rejeitando quem está perto **no espaço** (`min_px` adaptativo `max(3, 0.5·sqrt(area/pool))`) **e em features** (`div_thr=0.5`, requer referência); 2º passe preenche até `k` sem duplicar coord → **nunca devolve menos que o top-k**; mantém o top-1.
- `propose_many(..., k=10, ..., min_px=None, div_thr=0.5)` — pool `n=max(80,k*8)`, `rank`, `_greedy_diverse` com `area=abs((x1-x0)(y1-y0))`. GOTCHA: diversidade **intra-lote** (entre os picks), diferente da métrica `"diversity"` (vs. o store).
- **Onde editar:** nova métrica (ramo em `rank` ou `metrics.compute`); diversidade (`min_px`/`div_thr`); critério de completude (`.all()` em `_eval`/`_candidates_block`); cap de segurança (`_BLOCK_NATIVE_CAP`); densidade (`pool`/`step`, `k*8`).

### `core/trainer.py` — CNN temporal InceptionTime (GPU), spatial-CV, IO de modelo, cleanlab.

- `DEV = "cuda" if disponível else "cpu"` (fixo no import).
- `_IM(ic, nf=32, ks=(9,19,39), bn=32)` — módulo Inception (bottleneck 1×1 + 3 convs multi-escala + maxpool→1×1, concat `nf*4`, BN, ReLU).
- `IT(ic, nc, nf=32, dp=6)` — classificador: 6 blocos `_IM` + AdaptiveAvgPool1d + Dropout(0.2) + Linear(`nf*4=128`→`nc`). Sem skip connections.
- `build_X(cur4) -> (N,5,12)` — recebe `(N,4,12)` reflectância e **anexa NDVI como 5º canal**. GOTCHA: layout channel-first `(N,4,meses)`, ≠ do `FeatureExtractor`.
- `fit(Xt, yt, nc, lr=1e-3, epochs=200, mu=None, sd=None, progress=None) -> (net, mu, sd)` — `mu/sd` por canal (samples×tempo); class weights inverse-freq; AdamW(wd=1e-2) + CosineAnnealingLR; CE(label_smoothing=0.05); batch 256.
- `predict_proba(net, mu, sd, X, batch=8192) -> (N,nc)` — softmax, `no_grad`.
- `spatial_cv(X, y, groups, nc, lr, epochs, k=5, progress=None) -> dict` — `StratifiedGroupKFold` (fallback `GroupKFold`); `k=max(2,min(k,ng))`; devolve `oof_proba`, `pred`, `bacc`, `macro_f1`, `f1_per_class`, `confusion`, `n_groups`, `k`.
- `save_model(net,mu,sd,labs,path)` / `load_model(path) -> (net,mu,sd,labs)` — salva/rebuild com `ic`. GOTCHA: `weights_only=False` (só carregue arquivos confiáveis).
- `label_scores(y, oof_proba) -> (scores, issues)` — cleanlab: qualidade por ponto + índices de prováveis erros (worst-first); `n_jobs=1` (Windows).
- **Onde editar:** arquitetura (`IT`/`_IM`, mantendo `fc` in=`nf*4`); canais de entrada (`build_X`, `ic` gravado no checkpoint); hiperparâmetros (`fit`); estratégia de CV (`spatial_cv`); device (`DEV`); ranking de erro (`label_scores`).
