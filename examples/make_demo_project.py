"""Generate a small self-contained demo project — no external data needed.

Creates ``examples/demo_project/`` with a synthetic 12-month, 4-band cube
(256×256 px), three visualizations (RGB / continuous / categorical with an
embedded colormap), one vector layer and a ready ``project.yaml``. The NIR band
carries a seasonal signal: the northern half behaves like a single-cycle crop,
the southern half like a double-cycle one — so labeling, similarity, proposing
and training all do something meaningful.

Usage::

    python examples/make_demo_project.py
    tsa examples/demo_project
"""
import json
import os

import numpy as np
import rasterio
from rasterio.transform import from_origin

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_project")
W = H = 256
TR = from_origin(500000, 8200000, 10, 10)
CRS = "EPSG:32723"
NODATA = 65535

PROJECT_YAML = """\
name: demo_project
period: "2023"
classes:
  one_cycle: "#fee08b"
  two_cycles: "#74add1"
timeseries:
  paths: cube/*.tif
  bands: [1, 2, 3, 4]
  nodata: 65535
  scale: 10000
visualizations:
  continuous: {type: scalar, cmap: RdYlGn}
init: rgb
"""


def write(path, arr, dtype, nodata=None, colormap=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    prof = dict(driver="GTiff", width=W, height=H, count=arr.shape[0], dtype=dtype,
                crs=CRS, transform=TR, nodata=nodata, compress="deflate")
    with rasterio.open(path, "w", **prof) as ds:
        ds.write(arr.astype(dtype))
        if colormap:
            ds.write_colormap(1, colormap)


def main():
    rng = np.random.default_rng(0)
    for m in range(12):
        base = rng.integers(500, 1500, (4, H, W)).astype(np.uint16)
        green1 = 0.5 + 0.4 * np.sin(2 * np.pi * m / 12)          # one seasonal cycle
        green2 = 0.5 + 0.4 * np.sin(4 * np.pi * m / 12)          # two cycles
        nir = np.where(np.arange(H)[:, None] < H // 2, green1, green2) * 8000 + 1000
        base[3] = nir.astype(np.uint16)
        base[:, :8, :] = NODATA                                   # nodata strip (cloud-like)
        write(os.path.join(ROOT, "cube", f"m{m:02d}.tif"), base, "uint16", nodata=NODATA)

    rgb = rng.integers(0, 255, (3, H, W)).astype(np.uint8)
    write(os.path.join(ROOT, "visualizations", "rgb.tif"), rgb, "uint8")
    cont = (np.arange(H * W).reshape(1, H, W) % 200).astype(np.uint8)
    write(os.path.join(ROOT, "visualizations", "continuous.tif"), cont, "uint8", nodata=255)
    cat = (np.arange(W)[None, None, :] * 3 // W).astype(np.uint8) * np.ones((1, H, 1), np.uint8)
    write(os.path.join(ROOT, "visualizations", "categorical.tif"), cat, "uint8", nodata=255,
          colormap={0: (254, 224, 139, 255), 1: (116, 173, 209, 255), 2: (49, 54, 149, 255)})

    os.makedirs(os.path.join(ROOT, "layers"), exist_ok=True)
    square = {"type": "FeatureCollection", "features": [{
        "type": "Feature", "properties": {},
        "geometry": {"type": "Polygon", "coordinates": [[
            [500100, 8199000], [501000, 8199000], [501000, 8198000],
            [500100, 8198000], [500100, 8199000]]]}}]}
    with open(os.path.join(ROOT, "layers", "square.geojson"), "w") as f:
        json.dump(square, f)

    with open(os.path.join(ROOT, "project.yaml"), "w", encoding="utf-8") as f:
        f.write(PROJECT_YAML)
    print(f"demo project ready: {ROOT}")
    print("open it with:  tsa " + os.path.relpath(ROOT))


if __name__ == "__main__":
    main()
