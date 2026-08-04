"""FeatureExtractor — features fenológicas interpretáveis a partir da curva.

Regra-chave (correção validada por especialista): a CONTAGEM DE CICLO é por
ESTRUTURA (pico com VALE/prominência), NÃO por altura do pico — assim a
safrinha fraca-mas-real (vale claro, pico baixo) conta como ciclo, e o vigor
fica num eixo separado (amplitude/dry). NÃO se z-normaliza a curva (amplitude
é sinal).
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


def count_cycles(ndvi, peak_h=0.40, prominence=0.10, max_c=3):
    """Nº de ciclos = nº de picos com VALE (prominência) suficiente — não altura.
    Retorna (n, peak_positions, peak_heights, prominences)."""
    padded = np.r_[0.0, np.nan_to_num(ndvi, nan=0.0), 0.0]
    pk, props = find_peaks(padded, height=peak_h, prominence=prominence)
    n = min(max(len(pk), 1), max_c)
    return n, pk - 1, props.get("peak_heights", np.array([])), props.get("prominences", np.array([]))


class FeatureExtractor:
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
