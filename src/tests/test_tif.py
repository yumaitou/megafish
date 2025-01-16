# src/tests/test_tif.py
# src/tests/test_tif.py
import pytest
import numpy as np
import xarray as xr
import os
import tifffile
from megafish.tif import save, save_tile_montage, save_whole_image, save_chunk, load


def create_test_data(zarr_path, group, shape=(64, 64), dtype="uint16"):
    data = np.random.randint(0, 65535, shape).astype(dtype)
    if len(shape) == 3:
        dims = ["cycle", "y", "x"]
    elif len(shape) == 2:
        dims = ["y", "x"]
    ds = xr.DataArray(data, dims=dims).to_dataset(name="data")
    ds.to_zarr(zarr_path, group=group + "/0", mode="w")


def test_save(tmp_path):
    zarr_path = str(tmp_path / "test_tif.zarr")
    group = "testgroup"
    create_test_data(zarr_path, group, shape=(64, 64))
    # add a fake cycle dim
    ds = xr.open_zarr(zarr_path, group=group + "/0")
    ds = ds.expand_dims({"cycle": [0]})
    ds.to_zarr(zarr_path, group=group + "/0", mode="w")
    save(zarr_path, group, zoom=0)
    tif_dir = zarr_path.replace(".zarr", "_tif")
    assert os.path.exists(tif_dir)


def test_save_tile_montage(tmp_path):
    zarr_path = str(tmp_path / "test_montage.zarr")
    group = "testgroup"
    # single tile to avoid dimension issues
    data = np.random.randint(0, 65535, (1, 1, 1, 32, 32)).astype("uint16")
    ds = xr.DataArray(
        data, dims=["cycle", "tile_y", "tile_x", "y", "x"]).to_dataset(name="data")
    ds.to_zarr(zarr_path, group=group + "/0", mode="w")

    save_tile_montage(zarr_path, group, tile_size=(16, 16))
    tif_dir = zarr_path.replace(".zarr", "_tif")
    files = os.listdir(tif_dir)
    assert any("mtg.tif" in f for f in files)


def test_save_whole_image(tmp_path):
    zarr_path = str(tmp_path / "test_whole.zarr")
    group = "testgroup"
    data = np.random.randint(0, 65535, (1, 64, 64)).astype("uint16")
    ds = xr.DataArray(data, dims=["cycle", "y", "x"]).to_dataset(name="data")
    ds.to_zarr(zarr_path, group=group + "/0", mode="w")

    save_whole_image(zarr_path, group, zoom=0)
    tif_dir = zarr_path.replace(".zarr", "_tif")
    assert os.path.exists(tif_dir)


def test_save_chunk(tmp_path):
    zarr_path = str(tmp_path / "test_chunk.zarr")
    group = "testgroup"
    data = np.random.randint(0, 65535, (1, 64, 64)).astype("uint16")
    ds = xr.DataArray(data, dims=["cycle", "y", "x"]).to_dataset(name="data")
    ds.to_zarr(zarr_path, group=group + "/0", mode="w")

    chunk = [[0, 0, 0]]  # cycle=0,ychunk=0,xchunk=0
    save_chunk(zarr_path, group, chunk, footer="chk")
    tif_dir = zarr_path.replace(".zarr", "_tif")
    assert os.path.exists(tif_dir)


def test_load(tmp_path):
    zarr_path = str(tmp_path / "test_load.zarr")
    group_template = "template"
    group_load = "loadgroup"

    data = np.random.randint(0, 65535, (64, 64)).astype("uint16")
    ds = xr.DataArray(data, dims=["y", "x"]).to_dataset(name="data")
    ds.to_zarr(zarr_path, group=group_template + "/0", mode="w")

    tif_dir = zarr_path.replace(".zarr", "_tif")
    os.makedirs(os.path.join(tif_dir, group_load, "0"), exist_ok=True)
    img = np.random.randint(0, 65535, (64, 64)).astype("uint16")
    tifffile.imwrite(os.path.join(tif_dir, group_load, "0", "0_0.tif"), img)

    load(zarr_path, group_load, group_template, ".tif", dtype="uint16")
    ds = xr.open_zarr(zarr_path, group=group_load + "/0")
    assert "data" in ds
