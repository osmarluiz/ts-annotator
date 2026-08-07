"""Extratores de descritores a partir da curva.

Os motores que leem isto — SimilarityEngine, ClusterEngine, SelectionEngine —
não sabem o que os eixos significam: recebem um vetor de tamanho fixo, padronizam
e medem distância. Trocar o extrator troca o domínio sem tocar em nenhum deles.

Três estão disponíveis:

``phenology``  o conjunto original, fenológico, calibrado para safra no Cerrado
               (ano Out–Set, limiares absolutos de NDVI). É o PADRÃO, e projetos
               que não declaram nada continuam recebendo exatamente ele.
``shape``      descritores de forma sem limiar absoluto, por canal. Servem a NDVI,
               a retroespalhamento em dB, a temperatura de superfície ou a
               qualquer série.
``curve``      a própria curva achatada. Nenhuma suposição.

Regra-chave do ``phenology`` (correção validada por especialista): a CONTAGEM DE
CICLO é por ESTRUTURA (pico com VALE/prominência), NÃO por altura do pico — assim
a safrinha fraca-mas-real (vale claro, pico baixo) conta como ciclo, e o vigor
fica num eixo separado (amplitude/dry). NÃO se z-normaliza a curva (amplitude é
sinal).

Todo extrator devolve vetor finito (nunca NaN: NaN quebra a padronização na
similaridade e a ordenação do ranking) e expõe ``headline``, o número que o painel
mostra ao lado da curva. Só o fenológico tem um; os outros devolvem None e o chip
some.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

DRY_MONTHS = (8, 9, 10, 11)  # Jun, Jul, Ago, Set (índices 0=Out..11=Set)
# contrato de ordem de bandas do tool: B, G, R, NIR — FONTE ÚNICA dos índices
# usados por features e trainer (antes estavam espalhados/hardcoded)
RED_IDX, NIR_IDX = 2, 3


def to_ndvi(curve) -> np.ndarray:
    """Aceita (meses,4)=B,G,R,NIR (reflectância) ou (meses,)=NDVI. Retorna NDVI (meses,)."""
    c = np.asarray(curve, dtype=float)
    if c.ndim == 2 and c.shape[1] >= 4:
        red, nir = c[:, RED_IDX], c[:, NIR_IDX]
        return np.clip((nir - red) / (nir + red + 1e-6), -1.0, 1.0)
    return c.ravel()


def as_channels(curve) -> np.ndarray:
    """Aceita (meses,) ou (meses,bandas). Retorna (bandas,meses) — um canal por linha."""
    c = np.asarray(curve, dtype=float)
    if c.ndim == 1:
        return c[None, :]
    return c.T


def count_cycles(ndvi, peak_h=0.40, prominence=0.10, max_c=3):
    """Nº de ciclos = nº de picos com VALE (prominência) suficiente — não altura.
    Retorna (n, peak_positions, peak_heights, prominences)."""
    padded = np.r_[0.0, np.nan_to_num(ndvi, nan=0.0), 0.0]
    pk, props = find_peaks(padded, height=peak_h, prominence=prominence)
    n = min(max(len(pk), 1), max_c)
    return n, pk - 1, props.get("peak_heights", np.array([])), props.get("prominences", np.array([]))


class PhenologyFeatures:
    """Oito descritores fenológicos do NDVI. Limiares ABSOLUTOS: só faz sentido
    onde a curva é índice de vegetação numa estação de crescimento."""

    NAMES = [
        "n_cycles", "dry_ndvi", "amplitude", "peak_month",
        "peak_val", "second_peak", "valley_depth", "n_green",
    ]

    def __init__(self, dry=DRY_MONTHS, peak_h=0.40, prominence=0.10, green_thr=0.40):
        self.dry = list(dry)
        self.peak_h = peak_h
        self.prominence = prominence
        self.green_thr = green_thr

    def extract(self, curve) -> np.ndarray:
        nd = to_ndvi(curve)
        if not np.isfinite(nd).any():
            return np.zeros(len(self.NAMES))
        n_cyc, pk, heights, proms = count_cycles(nd, self.peak_h, self.prominence)
        dry = float(np.nanmean(nd[self.dry]))
        amp = float(np.nanmax(nd) - np.nanmin(nd))
        peak_month = int(np.nanargmax(np.nan_to_num(nd, nan=-1)))
        peak_val = float(np.nanmax(nd))
        hs = np.sort(heights)[::-1] if heights.size else np.array([0.0])
        second = float(hs[1]) if hs.size >= 2 else 0.0
        valley = float(proms.max()) if proms.size else 0.0
        n_green = int(np.nansum(nd >= self.green_thr))
        f = np.array(
            [n_cyc, dry, amp, peak_month, peak_val, second, valley, n_green], dtype=float
        )
        # rede de segurança: curva com nodata parcial (ex.: meses secos ausentes)
        # deixaria dry_ndvi=NaN, que quebra a distância na similaridade e a
        # ordenação do ranking. Nunca devolver NaN.
        return np.nan_to_num(f, nan=0.0)

    def extract_dict(self, curve) -> dict:
        return dict(zip(self.NAMES, self.extract(curve)))

    def headline(self, f):
        """Nº de ciclos, que o painel mostra ao lado da curva."""
        return int(f[0])


# O nome antigo continua valendo: projetos, testes e imports existentes apontam
# para ele, e o padrão do workspace é este extrator.
FeatureExtractor = PhenologyFeatures


class ShapeFeatures:
    """Oito descritores de forma POR CANAL, sem nenhum limiar absoluto.

    Contagem de extremos e prominência são medidas em fração da amplitude da
    própria curva, então a escala não importa: NDVI em [-1,1], dB negativo ou
    kelvin dão descritores comparáveis entre si depois da padronização.
    """

    BASE = ["mean", "std", "amplitude", "argmax", "argmin",
            "n_extrema", "top_prominence", "acf1"]

    def __init__(self, prominence_frac=0.25):
        self.prominence_frac = prominence_frac

    def names(self, n_channels=1):
        if n_channels == 1:
            return list(self.BASE)
        return [f"{n}_b{b}" for b in range(n_channels) for n in self.BASE]

    @property
    def NAMES(self):
        return self.names(1)

    def _one(self, x) -> list:
        x = np.asarray(x, float)
        good = np.isfinite(x)
        if not good.any():
            return [0.0] * len(self.BASE)
        x = np.where(good, x, np.nanmean(x[good]))
        n = len(x)
        lo, hi = float(x.min()), float(x.max())
        amp = hi - lo
        # prominência exigida em FRAÇÃO da amplitude: é isto que dispensa limiar
        prom = max(amp * self.prominence_frac, 1e-9)
        pk, props = find_peaks(np.r_[lo, x, lo], prominence=prom)
        proms = props.get("prominences", np.array([]))
        top = float(proms.max() / amp) if proms.size and amp > 0 else 0.0
        xc = x - x.mean()
        denom = float((xc * xc).sum())
        acf1 = float((xc[:-1] * xc[1:]).sum() / denom) if denom > 0 and n > 1 else 0.0
        return [
            float(x.mean()),
            float(x.std()),
            amp,
            float(np.argmax(x)) / max(n - 1, 1),
            float(np.argmin(x)) / max(n - 1, 1),
            float(min(len(pk), 8)),
            top,
            acf1,
        ]

    def extract(self, curve) -> np.ndarray:
        chans = as_channels(curve)
        f = np.array([v for ch in chans for v in self._one(ch)], dtype=float)
        return np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)

    def extract_dict(self, curve) -> dict:
        f = self.extract(curve)
        return dict(zip(self.names(len(as_channels(curve))), f))

    def headline(self, f):
        return None


class CurveFeatures:
    """A curva achatada, canal por canal. Nenhuma suposição sobre o que ela mede.

    A padronização acontece nos motores, por eixo e através da amostra, então a
    amplitude sobrevive como sinal — diferente de z-normalizar cada curva."""

    NAMES: list = []

    def names(self, n_channels=1, n_steps=0):
        if n_channels == 1:
            return [f"t{t}" for t in range(n_steps)]
        return [f"t{t}_b{b}" for b in range(n_channels) for t in range(n_steps)]

    def extract(self, curve) -> np.ndarray:
        chans = as_channels(curve)
        return np.nan_to_num(chans.reshape(-1).astype(float),
                             nan=0.0, posinf=0.0, neginf=0.0)

    def extract_dict(self, curve) -> dict:
        chans = as_channels(curve)
        return dict(zip(self.names(len(chans), chans.shape[1]), self.extract(curve)))

    def headline(self, f):
        return None


EXTRACTORS = {
    "phenology": PhenologyFeatures,
    "shape": ShapeFeatures,
    "curve": CurveFeatures,
}


def make_extractor(name=None, **kw):
    """Extrator pelo nome declarado no projeto. Sem nome, o fenológico — que é o
    que projetos existentes recebiam antes de haver escolha."""
    key = (name or "phenology").strip().lower()
    if key not in EXTRACTORS:
        raise ValueError(
            f"descriptors: {name!r} desconhecido. Disponíveis: {', '.join(sorted(EXTRACTORS))}"
        )
    return EXTRACTORS[key](**kw)
