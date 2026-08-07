# TSA User Guide

This guide walks the full loop: open a project → label → propose → train →
review → classify. UI labels are currently in Portuguese (the tool grew inside
a Brazilian research group); this guide gives both.

## 1. Open a project

```bash
tsa path/to/project        # folder containing project.yaml
```

The map opens on the `init:` visualization (or the first one found). The top
floating bar holds the session-wide state:

| control | meaning |
|---|---|
| `dataset:` | which point dataset (annotations/*.json) you are editing |
| `base:` | the raster displayed under everything |
| `sobre:` + slider | a second raster blended on top (e.g. classification over NDVI) |
| `modelo:` | the **active model** — drives suggestions, cards and classification |
| `área:` | the **work area** — *vista atual* (current view), *retângulo* (drawn rectangle), *células da grade* (grid cells), *imagem toda* (whole image) |
| `desenhar` | toggle: drag on the map draws/replaces the rectangle |
| `grade` + spinner | grid overlay; with it on, clicking toggles cell selection |
| `?` | all keyboard shortcuts |

The status bar at the bottom echoes everything that happens (saves, model
loads, warnings).

## 2. Label points (aba **Rotular** / Annotate)

- **Click a pixel** → its temporal curve appears below the map (NDVI by
  default; toggle B/G/R/NIR), and the class cards on the right show
  per-class **similarity** (S) and, once a model exists, **prediction** (P).
- **Label** by clicking a card or pressing its number key (`1`–`9`, `0`).
- Points save immediately (atomic JSON). **Ctrl+Z** undoes; **Backspace**
  removes the current point's label.
- A pixel with missing months (cloud/nodata) is not labelable — the status bar
  says so.
- Hold **Space** to pan without labeling; clicks near an existing point snap to
  it instead of creating a duplicate.

### Propose points (active learning)

Pick an objective (hover for a hint of when each is useful):

- *Diferente do que já rotulei* (diversity) — best at the start; no model needed.
- *Modelo incerto / ambíguo / muito incerto* — low confidence / small margin /
  high entropy; need a trained model.
- *Modelo × similaridade discordam* — systematic-error hunting.
- *Exemplos típicos de uma classe* — reinforce a weak class.

Set the work area (`área:`), the count, and press **propor sugestões**.
Suggestions are diverse (spatially and in feature space). Click one → the map
navigates to it (only re-framing if it is off-screen) → label with a key →
the list auto-advances. **Enter** skips.

The panel shows *labels since last training* — when that grows, retrain.

## 3. Train (aba **Treinar**)

Needs ≥ 30 points with valid curves (the tab shows your count). Training runs
on the GPU off the UI thread: an InceptionTime-style temporal CNN over the
declared bands (plus NDVI as an extra channel when red and NIR are among
them), evaluated with **stratified spatial cross-validation**
(k-means spatial blocks; test folds are spatially separated from training, so
the reported balanced accuracy and macro-F1 are honest), then refit on all
points. cleanlab ranks likely label errors from the out-of-fold probabilities.

Every run is saved as a **new version** — `models/it_vN/model.pt` +
`meta.yaml` (hyper-params, metrics, number of points, a hash of the exact
labeling state, the model file's sha1, date). Nothing is overwritten; the model
combo shows `it_vN · bacc · pts · date` and selects the new version. The
result line shows the delta vs the previous version.

## 4. Review (aba **Revisar**)

A sortable table over **all** labeled points — click a column header:

- `cleanlab` ascending → likely label errors first (or use the
  *revisar os N suspeitos* button on the Train tab, which jumps here already
  sorted);
- `≠` / *só ≠* filter → points where the model disagrees with your label.

Clicking a row navigates the map to the point; fix it with a number key or
remove it. The header line shows totals (points, disagreements, suspects).

## 5. Classify (aba **Classificar**)

**classificar (área)** runs the active model, always at full resolution:

- small areas (≤ ~200k px) run synchronously in seconds and paint a
  semi-transparent overlay;
- larger areas become a **background tile job**: progress + ETA, cancelable,
  resumable, and the *Predição (modelo)* layer fills in progressively. Partial
  jobs and the whole-image job accumulate into the same per-model raster
  (`predictions/pred_it_vN.tif`) — nothing is recomputed.

The tab shows the active model and its **coverage** (tiles done / total).
Select *Predição (modelo)* as `base:` or `sobre:` to inspect the map. If you
retrain, the layer hides rasters from older versions instead of silently
showing stale predictions.

## 6. Datasets

Each `annotations/*.json` is a selectable dataset (top bar). The JSON is
self-contained (coordinates + class + embedded curve), diffable and
versionable — it *is* the training set. Switching datasets clears suggestions
and the undo stack; training and its metrics writeback always go to the
dataset the training started on.
