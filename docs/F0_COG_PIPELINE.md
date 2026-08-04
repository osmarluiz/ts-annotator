# F0 — COG pipeline (sub-spec): o enabler de performance

> Objetivo: produzir os produtos **COG (tiled + overviews)** que tornam navegação, clique e cell-regen rápidos numa imagem de ~10 Gpixels. Saída do `BasemapService` + (opcional) cubo tiled.

## Produtos (todos COG: tile 512, overviews, DEFLATE, nodata)
| produto | bandas | uso | tamanho aprox. |
|---|---|---|---|
| **`basemap_pheno.tif`** | 3 (R=pico-NDVI, G=NDVI-seca, B=amplitude), uint8 | display — **lê os eixos** (default) | ~30 GB c/ overviews |
| **`basemap_mnf.tif`** | 3 (MNF comp 1-3), uint8 | display — estrutura (ENVI-like) | ~30 GB |
| **`class_cog.tif` / `proba_cog.tif`** | 1 / 11, uint8 | overlay + **consulta do SelectionEngine** | re-tile do existente |
| **`cube_tiled.tif`** *(OPCIONAL — ver decisão)* | 48 (12m×4b), uint16 | clique→série + cell-regen | **~300-500 GB** ⚠️ |

## ⚠️ A decisão real: COG-ar o cubo de 48 bandas, ou não?
- **COG do cubo:** clique e cell-regen **rápidos** (lê tiles), mas **~300-500 GB de disco**.
- **Ler do `.dat` sob demanda:** **zero disco extra**, mas leitura por janela mais lenta (o `.dat` é strip-based → lê faixas).
- **Recomendação:** **NÃO COG-ar o cubo na F0.** Display (basemaps pequenos) é o que precisa ser rápido pra navegar; clique = 1 pixel (tolerável do `.dat`); cell-regen é pontual. **COG-ar o cubo só se clique/regen ficarem lentos** (otimização tardia). Economiza centenas de GB.

## Passos
1. **Alinhar o cubo** (os 2 grids desalinhados: Out-Abr vs Mai-Set, offset por mês — já temos a regra `OFFS`). Trabalhar **por-tile** (não montar o array inteiro).
2. **MNF** (transform global):
   a. **Sample** ~200k pixels espalhados (cultivo + cover).
   b. Covariância de **ruído** (shift-difference) + sinal → autovetores MNF (matriz de transform) + percentis 2-98 p/ stretch.
   c. **Aplicar por-tile** na escrita: tile → 48 bandas → MNF → comp 1-3 → stretch → uint8.
3. **Fenológico por-tile:** lê NDVI 12m → pico (max), seca (Jun-Set mean), amplitude (max-min) → stretch (percentis do sample) → uint8 RGB.
4. **Re-tile class/proba** (já existem) → COG + overviews.
5. **Overviews** em todos: níveis [2,4,8,16,32]; resampling **average** (contínuo) / **nearest** (classe).
   - ⚠️ **LIÇÃO (corrompemos um 15GB):** a causa foi **dois `build_overviews` CONCORRENTES** no mesmo bigtiff (lancei o 2º achando que o 1º tinha acabado) → quebra a cadeia IFD (arquivo não abre). Regras: (a) **esperar a task terminar pela NOTIFICAÇÃO** (não adivinhar por PID/tamanho); (b) **build ÚNICO**; (c) p/ arquivos grandes preferir **COG-translate** (arquivo novo, sem `r+` append) ou `gdaladdo` único monitorado.
6. **Validar:** abrir por janela em cada nível → checar velocidade + alinhamento espacial (sobrepor pivôs ANA como sanity).

## Contrato resultante (alimenta o `BasemapService`)
- `available() -> ["pheno","mnf","class"]`
- `tile(name, window, level) -> array` (lê só a janela no nível de overview certo)
- Série do pixel (clique): `TimeSeriesStore.series(row,col)` — lê do `.dat` (ou do cube COG se existir).

## Saídas / onde
`outputs/cog/` no diretório do projeto RIDE. Versionar o **transform MNF** (autovetores + stretch) p/ reprodutibilidade.

## Decisões em aberto da F0
- ✅ **RESOLVIDO — Cubo COG: NÃO** (ler do `.dat` sob demanda; revisitar só se clique/cell-regen ficarem lentos). Economiza ~300-500 GB.
- **MNF no stack 48-band** (espaço-temporal) vs por-época — recomendo 48-band (mais limpo).
- **Stretch** fixo (percentis do sample) vs adaptativo por-viewport.
