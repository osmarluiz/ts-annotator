"""Background workers that run heavy work (training, batch classify) off the UI thread."""

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from ts_annotator.app.config import NON_TRAINING_CLASSES  # bucket "don't know" (fora do treino)


class TrainWorker(QObject):
    """Train the IT model on the GPU (spatial-CV + cleanlab + final fit) off the UI thread."""

    done = pyqtSignal(object)
    prog = pyqtSignal(str)
    step = pyqtSignal(dict)     # {phase, frac, fold, k, epoch, epochs, loss, fold_bacc}
    _go = pyqtSignal()

    def __init__(self, store, fx, band_names=None):
        super().__init__()
        self.store = store
        self.fx = fx
        self.band_names = band_names
        self.lr = 1e-3
        self.epochs = 120
        self.folds = 5
        self.n_blocks = None     # None = formula automatica de sempre
        self.buffer_px = 0.0     # 0 = comportamento anterior (sem zona morta)
        self.class_super = {}     # classe fina -> superclasse
        self.collapsed = set()    # supers a COLAPSAR (treinar nesse nível)
        self._go.connect(self._run)

    def start(self, lr, epochs, folds, class_super=None, collapsed=None,
              n_blocks=None, buffer_px=0.0, arch=None):
        from collections import Counter
        self.lr, self.epochs, self.folds = lr, epochs, folds
        self.n_blocks, self.buffer_px = n_blocks, float(buffer_px or 0.0)
        self.arch = arch
        self.class_super = dict(class_super or {})
        self.collapsed = set(collapsed or ())
        # supers "de grupo" (>1 membro fino) — um rótulo com esse nome é um SUPER-rótulo
        self._grouping = {s for s, n in Counter(self.class_super.values()).items() if n > 1}
        self._go.emit()

    def _eff(self, cls):
        """Label efetiva no nível do treino. Retorna None se o ponto for INVÁLIDO
        neste nível (super-rótulo enquanto se treina fino → ambíguo, sai do treino)."""
        if cls in self.class_super:                 # classe FINA
            s = self.class_super[cls]
            return s if s in self.collapsed else cls
        if cls in self._grouping:                   # SUPER-rótulo (ex.: 'gramineas')
            return cls if cls in self.collapsed else None   # válido só se o super colapsa
        return cls

    @pyqtSlot()
    def _run(self):
        try:
            import numpy as np
            from ts_annotator.core import trainer
            # snapshot ÚNICO do store: on_dataset_change reaponta self.store no meio
            # do treino — pts e res["store"] têm que vir do MESMO objeto, senão o
            # writeback das métricas vai pro dataset errado (ids colidem).
            store = self.store
            # exige curva COMPLETA: um único ponto com NaN torna mu/sd NaN e
            # propaga NaN por toda a rede (modelo prevê classe 0 na imagem toda).
            cand = [p for p in store.points
                    if "curve" in p and np.isfinite(np.asarray(p["curve"], float)).all()]
            pts, ylab = [], []
            for p in cand:
                e = self._eff(p["class"])            # colapsa/valida por super
                if e is None or e in NON_TRAINING_CLASSES:
                    continue                          # super-rótulo em treino fino, ou nao_sei
                pts.append(p)
                ylab.append(e)
            if len(pts) < 30:
                self.done.emit({"error": f"só {len(pts)} ponto(s) válido(s) neste nível — "
                                         f"rotule pelo menos 30 antes de treinar"})
                return
            cur4 = np.array([np.asarray(p["curve"]).T for p in pts], "float32")  # (N,4,12)
            coords = np.array([[p["row"], p["col"]] for p in pts], float)
            groups = trainer.spatial_blocks(coords, self.folds, self.n_blocks)
            labs = sorted(set(ylab))
            y = np.array([labs.index(c) for c in ylab])
            X = trainer.build_X(cur4, bands=self.band_names)
            self.prog.emit("spatial-CV (GPU)…")
            # progresso global: CV (k folds × E épocas) + fit final (E épocas)
            total = (self.folds + 1) * self.epochs

            def _cv_epoch(f, k, e, E, loss):
                self.step.emit({"phase": "spatial-CV", "fold": f, "k": k, "epoch": e, "epochs": E,
                                "loss": loss, "frac": ((f - 1) * E + e) / total})

            def _cv_fold(f, k, fold_bacc):
                self.prog.emit(f"spatial-CV: fold {f}/{k} — bacc {fold_bacc:.3f}")
                self.step.emit({"phase": "spatial-CV", "fold": f, "k": k, "fold_bacc": fold_bacc,
                                "frac": f * self.epochs / total})

            res = trainer.spatial_cv(X, y, groups, len(labs), lr=self.lr, epochs=self.epochs,
                                     k=self.folds, progress=_cv_fold, epoch_cb=_cv_epoch,
                                     coords=coords, buffer_px=self.buffer_px, arch=self.arch)
            self.prog.emit("cleanlab: analisando qualidade dos rótulos…")
            self.step.emit({"phase": "cleanlab", "frac": self.folds * self.epochs / total})
            scores, issues = trainer.label_scores(y, res["oof_proba"])

            def _fit_epoch(e, E, loss):
                self.step.emit({"phase": "treino final", "epoch": e, "epochs": E, "loss": loss,
                                "frac": (self.folds * self.epochs + e) / total})

            net, mu, sd = trainer.fit(X, y, len(labs), lr=self.lr, epochs=self.epochs,
                                      progress=_fit_epoch, arch=self.arch)
            # store de ORIGEM vai junto: o usuário pode trocar de dataset durante os
            # minutos de treino — o writeback das métricas tem que ir no store em que
            # o snapshot foi tirado (ids colidem entre datasets). params/anno_hash/
            # n_points alimentam o meta.yaml da versão (proveniência).
            from ts_annotator.core.version_store import anno_hash
            res.update(labs=labs, y=y, scores=scores, issues=issues,
                       net=net, mu=mu, sd=sd, ids=[p["id"] for p in pts], store=store,
                       arch=self.arch or "it",
                       params={"lr": self.lr, "epochs": self.epochs, "folds": self.folds,
                               "arch": self.arch or "it"},
                       n_points=len(pts), anno_hash=anno_hash(pts),
                       collapsed=sorted(self.collapsed),   # supers colapsados neste treino
                       confusion=res["confusion"].tolist())
            self.done.emit(res)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.done.emit({"error": str(e)})


class ClassifyAllWorker(QObject):
    """Classifica a IMAGEM TODA com um modelo salvo, tile a tile, gravando num
    PredictionRaster (por-modelo, retomável via sidecar).

    Roda numa QThread própria; abre os SEUS handles de raster (não compartilha
    datasets abertos com a UI — rasterio não é thread-safe) e carrega a SUA cópia
    do modelo. Cancelável via cancel() (flag checada a cada tile; o progresso já
    feito fica salvo e o próximo start retoma de onde parou).
    """

    done = pyqtSignal(object)
    prog = pyqtSignal(str, int)   # (mensagem, percent 0-100)
    _go = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.job = None
        self._cancel = False
        self._go.connect(self._run)

    def start(self, job):
        """job: dict com model_path, params do TimeSeriesSource, width/height e
        pred (PredictionRaster já garantido via ensure() na UI thread)."""
        self.job = job
        self._cancel = False
        self._go.emit()

    def cancel(self):
        self._cancel = True

    @pyqtSlot()
    def _run(self):
        try:
            self._run_job()
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.done.emit({"error": str(e)})

    def _run_job(self):
        import time

        import numpy as np
        from rasterio.windows import Window

        from ts_annotator.core import trainer
        from ts_annotator.core.inference import classify_block
        from ts_annotator.core.timeseries import TimeSeriesSource

        j = self.job
        pr = j["pred"]
        meta = pr.load_meta() or {}
        T = int(meta.get("tile", 1024))
        W, H = int(j["width"]), int(j["height"])
        tiles = [(ti, tj) for ti in range(-(-H // T)) for tj in range(-(-W // T))]
        # escopo opcional (retângulo/células): classifica só estes tiles; done_tiles
        # é GLOBAL, então jobs parciais e o de imagem toda se acumulam sem refazer.
        scope = j.get("tiles")
        sel = [t for t in tiles if t in set(map(tuple, scope))] if scope else tiles
        done = set(meta.get("done_tiles", []))
        todo = [t for t in sel if f"{t[0]}_{t[1]}" not in done]
        total = len(sel)
        k0 = total - len(todo)

        self.prog.emit("carregando modelo…", int(100 * k0 / max(total, 1)))
        net, mu, sd, _labs = trainer.load_model(j["model_path"])
        ts = TimeSeriesSource(j["month_paths"], bands=j["bands"],
                              band_names=j.get("band_names"),
                              row_offsets=j["row_offsets"], col_offsets=j["col_offsets"],
                              nodata=j["nodata"], scale=j["scale"])
        t0 = time.time()
        cancelled = False
        ds = pr.open_writer()
        try:
            for i, (ti, tj) in enumerate(todo):
                if self._cancel:
                    cancelled = True
                    break
                row0, col0 = ti * T, tj * T
                h, w = min(T, H - row0), min(T, W - col0)
                ci, valid, _ = classify_block(ts, net, mu, sd, row0, col0, h, w,
                                              step=1, batch=65536)
                out = np.full(h * w, pr.NODATA, np.uint8)
                out[valid] = ci.astype(np.uint8)
                ds.write(out.reshape(h, w), 1, window=Window(col0, row0, w, h))
                done.add(f"{ti}_{tj}")
                if (i + 1) % 25 == 0:
                    pr.save_progress(done)
                k = k0 + i + 1
                eta = (time.time() - t0) / (i + 1) * (len(todo) - i - 1)
                self.prog.emit(f"tile {k}/{total} — ETA {self._fmt(eta)}",
                               int(100 * k / total))
        finally:
            ds.close()
            ts.close()
        pr.save_progress(done)
        # cancel entre o último tile e os overviews também conta: build_overviews
        # leva minutos e não é interrompível — não iniciar se o usuário já cancelou.
        if cancelled or self._cancel:
            # contagem relativa ao ESCOPO deste job (done é o ledger global — um job
            # de 4 tiles após um de imagem toda diria "102/4" sem este filtro)
            done_sel = sum(1 for t in sel if f"{t[0]}_{t[1]}" in done)
            self.done.emit({"cancelled": True, "done": done_sel, "total": total,
                            "tif": pr.tif})
            return
        # Overviews (pirâmide) SÃO necessários p/ ver o resultado em ZOOM-OUT (sem
        # eles, ou desatualizados, a área nova some no zoom-out). SEMPRE reconstrói —
        # mesmo job pequeno: o usuário dá zoom-out e a área tem que aparecer. Raster
        # esparso → build lê só os tiles escritos, é rápido.
        complete = (len(done) == len(tiles))
        self.prog.emit("gerando overviews…", 100)
        pr.build_overviews()
        pr.save_progress(done, complete=complete)
        self.done.emit({"ok": True, "total": total, "tif": pr.tif})

    @staticmethod
    def _fmt(sec):
        sec = int(sec)
        if sec >= 3600:
            return f"{sec // 3600}h{(sec % 3600) // 60:02d}"
        return f"{sec // 60}min{sec % 60:02d}s"


class OverviewWorker(QObject):
    """Reconstrói a pirâmide do pred raster FORA da UI. Um classify síncrono grava
    dados novos via write_block que os overviews ainda não cobrem → sumiria no
    zoom-out; isto os atualiza sem travar a interface."""

    done = pyqtSignal(str)   # tif reconstruído
    _go = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._go.connect(self._run)

    def start(self, tif):
        self._go.emit(tif)

    @pyqtSlot(str)
    def _run(self, tif):
        try:
            import rasterio
            from rasterio.enums import Resampling
            with rasterio.open(tif, "r+") as ds:
                ds.build_overviews([2, 4, 8, 16, 32], Resampling.nearest)
        except Exception:
            import traceback
            traceback.print_exc()
        self.done.emit(tif)
