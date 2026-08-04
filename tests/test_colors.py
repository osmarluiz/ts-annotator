"""family_shades (cores por família) + derivação automática no workspace."""
import colorsys

from ts_annotator.core.colors import _hex_to_rgb, family_shades


def _hls(hexstr):
    return colorsys.rgb_to_hls(*_hex_to_rgb(hexstr))


def test_shades_same_hue_softer_than_base():
    base = "#d35400"
    h0, l0, s0 = _hls(base)
    shades = family_shades(base, 3)
    assert len(shades) == 3 and base not in shades
    for sh in shades:
        h, l, s = _hls(sh)
        assert abs(h - h0) < 0.02          # mesmo matiz
        assert l > l0                      # sempre mais clara que a base (super = mais forte)
    # ladder: i=0 é a mais suave, i=n-1 a mais próxima da base
    ls = [_hls(sh)[1] for sh in shades]
    assert ls == sorted(ls, reverse=True)


def test_shades_n1_and_n0():
    assert family_shades("#1e8449", 0) == []
    (one,) = family_shades("#1e8449", 1)
    assert _hls(one)[1] > _hls("#1e8449")[1]


def test_workspace_derives_missing_child_colors():
    from ts_annotator.app.workspace import _class_colors
    cfg = {
        "classes": {
            "a": {"super": "s"},               # sem cor -> tom derivado do super
            "b": {"super": "s"},
            "c": "#123456",                    # cor explícita vence
            "d": {"color": "#abcdef", "super": "s"},
        },
        "superclasses": {"s": "#d35400"},
    }
    out = _class_colors(cfg)
    assert out["c"] == "#123456" and out["d"] == "#abcdef"
    assert out["a"] != out["b"]                # tons distintos entre irmãs
    ha, la, _ = colorsys.rgb_to_hls(*_hex_to_rgb(out["a"]))
    h0, l0, _ = colorsys.rgb_to_hls(*_hex_to_rgb("#d35400"))
    assert abs(ha - h0) < 0.02 and la > l0     # família do super, mais suave
