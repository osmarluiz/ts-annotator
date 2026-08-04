"""Cores por família: superclasse = cor forte; filhas = tons mais suaves do mesmo matiz.

GUI-free — usado pelo workspace (cores default de filhas sem `color` no yaml)
e testável sem Qt.
"""
from __future__ import annotations

import colorsys


def _hex_to_rgb(h: str):
    h = h.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def _rgb_to_hex(rgb) -> str:
    return "#" + "".join(f"{round(c * 255):02x}" for c in rgb)


def family_shades(base_hex: str, n: int) -> list[str]:
    """``n`` versões mais SUAVES de ``base_hex``: mesmo matiz, mais claras e menos
    saturadas, da mais suave (i=0) à mais próxima da base (i=n-1). Nenhuma coincide
    com a base — a superclasse fica sempre a cor mais forte da família."""
    if n < 1:
        return []
    h, l, s = colorsys.rgb_to_hls(*_hex_to_rgb(base_hex))
    l_hi = 0.76                                  # filha mais suave (clara, ainda visível no mapa)
    l_lo = min(0.66, max(l + 0.14, 0.48))        # filha mais forte, ainda distinta da base
    out = []
    for i in range(n):
        t = i / (n - 1) if n > 1 else 0.5
        li = l_hi + (l_lo - l_hi) * t
        si = s * (0.72 + 0.20 * t)               # suaviza pouco a saturação (mapa pede cor viva)
        out.append(_rgb_to_hex(colorsys.hls_to_rgb(h, li, si)))
    return out
