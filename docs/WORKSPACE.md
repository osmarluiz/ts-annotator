# Workspace / projeto (convention-over-configuration)

> O software **descobre** o conteúdo do projeto a partir de pastas — solta um arquivo, ele aparece. Isso generaliza o tool (qualquer raster/projeto) e é a base "config-driven" pra publicação (JOSS).

> **STATUS (jul/2026): IMPLEMENTADO** em `app/workspace.py` (`load_workspace(dir) -> AppContext`)
> + CLI `tsa <pasta-do-projeto>` (`app/main.py`, entry-point no pyproject). Cobre: project.yaml
> (timeseries por glob/lista, classes, overrides de visualização, similarity, reference npz,
> init, groups, period, **season** — `months:` rótulos do eixo da curva + `dry_months:` índices
> dos meses secos p/ as features fenológicas, **descriptors** — qual extrator alimenta
> similaridade/descoberta/proposta: `phenology` (padrão, o conjunto fenológico de NDVI;
> `dry_months` só vale aqui), `shape` (descritores de forma por canal, sem limiar absoluto —
> serve a dB, temperatura, qualquer série) ou `curve` (a curva achatada, sem suposição
> nenhuma)), descoberta de visualizations/ (colorização inferida: ≥3 bandas→RGB,
> 1 banda com colormap embutido→LUT do arquivo, 1 banda contínua→scalar com percentis 2–98),
> layers/ (gpkg/geojson/shp), annotations/*.json (datasets selecionáveis; cria points.json
> vazio), models/ e predictions/. A grade de referência é o 1º raster da série (as
> visualizações devem estar na MESMA grade).
>
> **VersionStore TAMBÉM IMPLEMENTADO** (`core/version_store.py`): cada treino salva
> `models/it_vN/{model.pt, meta.yaml}` (params, métricas do spatial-CV, n_pontos, `anno_hash`
> do estado da rotulagem, `model_sha1`, dataset, data) — nada é sobrescrito; o combo de modelos
> mostra `it_vN · bacc · pts · data`. A identidade modelo↔predição é o **sha1 do .pt** (robusta
> a cópia/migração; sidecars antigos caem no mtime). O sidecar `pred_<versão>.json` das
> predições já carrega a proveniência (modelo, sha1, done_tiles, data) — cumpre o papel do
> `pred_vN.yaml` desenhado abaixo. **Pendente:** JSONL p/ >50k pontos, sidecar .json por
> visualização (hoje o override é no project.yaml).

## Estrutura
```
<projeto>/
  project.yaml          # config (classes, CRS, fonte da série temporal, estilos/overrides)
  visualizations/       # basemaps (COGs) — LISTADOS automaticamente como opções
      class.tif  intensity.tif  mnf.tif  ndvi_temporal.tif  pheno.tif ...
  layers/               # overlays vetoriais (shapefile/geojson/gpkg) — listados automaticamente
      pivos.gpkg  talhoes_sam.gpkg  mapbiomas.gpkg ...
  annotations/          # NOSSOS pontos rotulados (a camada de curvas)
      points.json       # = camada no mapa + dataset de treino (auto-contido; spec abaixo)
  models/               # modelos treinados, versionados
      it_v1/ model.pt + meta.yaml   # tipo, lr, epochs, classes(ordem), mu/sd, n_pontos, hash anotações, métricas, data
      it_v2/ ...
  predictions/          # saídas dos modelos (overlays transparentes versionados)
      pred_v1.tif + pred_v1.yaml    # modelo=it_v1, escopo (células/ret/tudo), data
```
Trocar visualização/layer = **dropar arquivo na pasta + recarregar**.

## 3 decisões de design
1. **Colorização de cada visualização — inferida:** 3-bandas uint8 → RGB (MNF, NDVI-temporal, pheno); 1-banda contínua → colormap (intensidade); categórica → LUT das classes (do `project.yaml`). **Override opcional** via sidecar (ex.: `mnf.json {"type":"rgb"}`).
2. **Camada de pontos = DUPLA:** (a) layer no mapa (marcadores por classe), (b) **dataset de treino**. É a fonte do active learning (similaridade, sugestão, treino leem dela). Formato: **JSON único auto-contido** (estilo iSAGE — *o JSON É o dataset*), com a **curva embutida** → treina **sem re-ler o cubo (10 GB)**. Spec na seção "Annotation JSON".
3. **Fonte da série temporal** no `project.yaml` (caminho do cubo/.dat) → o clique→curva lê dela; as `visualizations/` são **derivadas** dela (pré-computadas) — ou geradas on-the-fly no futuro.

## project.yaml (rascunho)
```yaml
crs: EPSG:32723
classes:                      # nome -> cor/atalho
  1c: {color: "#fee08b", key: 1}
  2c: {color: "#74add1", key: 2}
  3c: {color: "#313695", key: 3}
  savana: {color: "#b8e186", key: 4}
  # ...
timeseries:
  source: data/cube.vrt       # stack multi-banda (clique->curva)
  months: [Out, Nov, ..., Set]
  bands: [B, G, R, NIR]
  # row_offsets/col_offsets: correção de desregistragem por mês que NÃO está no
  # georreferenciamento (o grid diz alinhado, mas o conteúdo do pixel está deslocado).
  # 1 valor por mês, aplicado na leitura por (row,col) — conserta curva, similaridade
  # e classificação de uma vez. Ex. RIDE-2023: mai-set com -8px vertical → +8 linhas.
  # row_offsets: [0,0,0,0,0,0,0, 8,8,8,8,8]
visualizations:               # opcional (senão infere da pasta)
  mnf: {type: rgb}
  intensity: {type: scalar, cmap: RdYlGn, vmin: 30, vmax: 180, nodata: 255}
  class: {type: class}        # usa `classes`
```

## Componentes do software (mapeiam no spec principal)
- **`WorkspaceLoader`** (novo) — varre as pastas + lê `project.yaml` → popula `BasemapService` (visualizations) e `LayerManager` (layers).
- **`AnnotationStore`** → lê/grava `annotations/points.json` (camada + dataset; escrita atômica temp+rename).
- `BasemapService` / `LayerManager` / `SelectionEngine` consomem o que o `WorkspaceLoader` montou.

## Quando implementar
Fase de **generalização** (tira paths hardcoded). Por ora o demo usa caminhos fixos; este doc fixa o desenho.

## Annotation JSON (`annotations/points.json`) — spec (FECHADO)
Filosofia iSAGE: **o JSON É o dataset** (auto-contido, diffável, versionável, editável à mão, mergeável). Adaptação TSA: **curva embutida** (treina sem re-ler o cubo) + **período** (curva/rótulo são por ano-safra).
```json
{
  "format_version": "1.0",
  "project": "ride_df",
  "crs": "EPSG:32723",
  "timeseries": {
    "months": ["Out","Nov","Dez","Jan","Fev","Mar","Abr","Mai","Jun","Jul","Ago","Set"],
    "bands": ["B","G","R","NIR"],
    "scale": "reflectance_0_1"
  },
  "points": [
    {
      "id": 1,
      "x": 234567.0, "y": 8123456.0,        // geo (UTM) — portável (sobrevive a re-grid)
      "row": 50000, "col": 46000,            // pixel — display rápido
      "period": "2022_2023",                 // ano-safra (destrava multi-ano)
      "class": "2c",
      "source": "manual|suggested|reviewed",
      "confidence": 0.9,
      "curve": [[0.12,0.15,0.11,0.32], "...12 meses (B,G,R,NIR)..."]  // null no mês nuvem/nodata
    }
  ]
}
```
**Decisões fechadas:**
- **Curva = 4 bandas × 12 meses** (reflectância 0-1); NDVI (5º canal) é derivado. Mês inválido = `null`.
- **Coord = geo (x,y) + pixel (row,col)** — geo portável, pixel rápido.
- **`period`** por ponto → mesmo local em anos diferentes (multi-ano) num só JSON, ou 1 JSON por ano.
- **Curva embutida** (auto-contido) + **coord guardada** → re-extraível do cubo se reprocessar.
- **Escrita atômica** (temp+rename). Escala futura: JSONL (1 ponto/linha, append-only) se passar de ~50k.

## Modelos & predições (proveniência)
Cadeia: **anotações → modelo (params) → predição**. Cada elo é auto-descrito e versionado.
- **`models/it_vN/meta.yaml`**: tipo (IT), `lr`, `epochs`, **classes (ordem)**, **mu/sd (normalização)**, n_pontos, hash/versão das anotações usadas, métricas (ver treino), data. → um `.pt` salvo é **re-aplicável** (mu/sd+classes), não um peso solto.
- **`predictions/pred_vN.yaml`**: aponta pro `model=it_vN` + escopo (células/retângulo/tudo) + data. → sempre dá pra dizer "essa predição veio de qual modelo, com quais pontos".
- **WorkspaceLoader** varre `models/` (lista da aba Train: treinar novo OU re-aplicar salvo) e `predictions/` (dropdown de versão do overlay + slider de opacidade).

## Treino: train / val / test (split ESPACIAL)
Em sensoriamento remoto, split **aleatório vaza** (pixels vizinhos são autocorrelacionados → teste perto do treino = acurácia inflada). Então:
- **Split por BLOCO espacial** (não aleatório): cada ponto cai num bloco (ex.: as células do grid, ou a `region` da referência). **Blocos inteiros** vão pra treino OU teste → teste fica espacialmente separado.
- **Default:** hold-out espacial (~20% dos blocos) → métricas honestas: **balanced accuracy, macro-F1, F1 por classe, matriz de confusão**. (Opcional: k-fold espacial p/ estimativa robusta — é o que o pipeline atual já faz.)
- **Métricas salvas** no `meta.yaml` + mostradas na aba Train → você vê se o modelo presta **antes** de classificar.
- Pontos do usuário (poucos, escolhidos a dedo) entram no **treino** (sinal de melhoria); a avaliação é nos blocos held-out da referência.

## Resolvido
- Pastas `visualizations` / `layers` / `annotations` / **`models`** / **`predictions`** — **OK**.
- Config em **YAML**.
- Pontos em **JSON único auto-contido com curva + período** (acima).
- Modelos versionados auto-descritos + predições versionadas (overlay) + **split espacial** no treino.
