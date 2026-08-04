"""Criação de projeto pela API pura (sem GUI) — o writer que o wizard chama."""
import os

import pytest
import yaml

from ts_annotator.core.project_init import (
    ProjectInitError,
    _rgb_index_from_1based,
    build_config,
    find_images,
    init_project,
)


def _touch(path):
    with open(path, "wb") as f:
        f.write(b"\0")
    return path


def _make_months(folder, n=12):
    months = [202210, 202211, 202212, 202301, 202302, 202303,
              202304, 202305, 202306, 202307, 202308, 202309][:n]
    return [_touch(os.path.join(folder, f"ride_{m}.tif")) for m in months]


def test_find_images_sorts_chronologically(tmp_path):
    d = tmp_path / "imgs"
    d.mkdir()
    _make_months(str(d))
    _touch(str(d / "leia-me.txt"))          # não-raster é ignorado
    imgs = find_images(str(d))
    assert len(imgs) == 12
    assert imgs[0].endswith("ride_202210.tif")
    assert imgs[-1].endswith("ride_202309.tif")


def test_find_images_missing_folder():
    with pytest.raises(ProjectInitError):
        find_images("/nao/existe/mesmo")


def test_rgb_index_1based_to_0based():
    assert _rgb_index_from_1based("3,2,1", 4) == [2, 1, 0]
    assert _rgb_index_from_1based("", 4) is None
    with pytest.raises(ProjectInitError):
        _rgb_index_from_1based("3,2", 4)       # precisa de 3
    with pytest.raises(ProjectInitError):
        _rgb_index_from_1based("3,2,9", 4)     # fora do intervalo


def test_init_project_writes_valid_yaml_and_scaffold(tmp_path):
    (tmp_path / "src").mkdir()
    imgs = _make_months(str(tmp_path / "src"))
    proj = tmp_path / "proj"
    pdir = init_project(str(proj), imgs, name="t", period="2023", rgb="3,2,1")
    cfg = yaml.safe_load(open(os.path.join(pdir, "project.yaml"), encoding="utf-8"))
    assert cfg["name"] == "t"
    assert cfg["period"] == "2023"
    assert cfg["rgb_index"] == [2, 1, 0]
    assert len(cfg["timeseries"]["paths"]) == 12
    assert cfg["timeseries"]["bands"] == [1, 2, 3, 4]
    for sub in ("visualizations", "layers", "annotations", "models", "predictions"):
        assert os.path.isdir(os.path.join(pdir, sub))


def test_init_project_refuses_overwrite(tmp_path):
    imgs = _make_months(str(tmp_path))
    proj = tmp_path / "proj"
    init_project(str(proj), imgs)
    with pytest.raises(ProjectInitError):
        init_project(str(proj), imgs)             # já existe project.yaml
    init_project(str(proj), imgs, overwrite=True)  # com overwrite, ok


def test_init_project_requires_images(tmp_path):
    with pytest.raises(ProjectInitError):
        init_project(str(tmp_path / "p"), [])


def test_init_project_rejects_missing_image(tmp_path):
    with pytest.raises(ProjectInitError):
        init_project(str(tmp_path / "p"), [str(tmp_path / "nope.tif")])


def test_build_config_relative_when_in_tree(tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "imgs").mkdir()
    inside = _make_months(str(proj / "imgs"))
    cfg = build_config(inside, name="x", project_dir=str(proj))
    # imagens DENTRO do projeto → caminho relativo (projeto portátil)
    assert not os.path.isabs(cfg["timeseries"]["paths"][0])
    assert cfg["timeseries"]["paths"][0].startswith("imgs/")


def test_build_config_absolute_when_out_of_tree(tmp_path):
    proj = tmp_path / "proj"
    other = tmp_path / "elsewhere"
    other.mkdir()
    outside = _make_months(str(other))
    cfg = build_config(outside, name="x", project_dir=str(proj))
    assert os.path.isabs(cfg["timeseries"]["paths"][0])


def test_build_config_default_class(tmp_path):
    imgs = _make_months(str(tmp_path))
    cfg = build_config(imgs, name="x")
    assert cfg["classes"] == {"nao_sei": "#7a7a7a"}
