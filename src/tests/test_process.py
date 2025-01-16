# src/tests/test_process.py
import pytest
import numpy as np
import xarray as xr
import os

from megafish.process import projection, masking, gaussian_blur, binarization


def create_test_data(zarr_path, group, shape=(1, 1, 1, 64, 64), dtype="float32"):
    data = np.random.rand(*shape).astype(dtype)
    ds = xr.DataArray(
        data, dims=["cycle", "tile_y", "tile_x", "y", "x"]).to_dataset(name="data")
    ds.to_zarr(zarr_path, group=group + "/0", mode="w")


def test_projection(tmp_path):
    zarr_path = str(tmp_path / "test_projection.zarr")
    group = "testgroup"
    create_test_data(zarr_path, group)
    projection(zarr_path, group, dim="y", method="max")
    out_group = group + "_mip"
    ds = xr.open_zarr(zarr_path, group=out_group + "/0")
    assert "data" in ds


def test_masking(tmp_path):
    zarr_path = str(tmp_path / "test_masking.zarr")
    group_int = "intensity"
    group_mask = "mask"
    create_test_data(zarr_path, group_int)
    mask_data = (np.random.rand(64, 64) > 0.5).astype("uint8")
    ds_m = xr.DataArray(mask_data, dims=["y", "x"]).expand_dims(
        {"cycle": [0], "tile_y": [0], "tile_x": [0]})
    ds_m = ds_m.to_dataset(name="data")
    ds_m.to_zarr(zarr_path, group=group_mask + "/0", mode="w")

    masking(zarr_path, group_int, group_mask)
    out_group = group_int + "_msk"
    ds = xr.open_zarr(zarr_path, group=out_group + "/0")
    assert "data" in ds


def test_gaussian_blur(tmp_path):
    zarr_path = str(tmp_path / "test_gaussian.zarr")
    group = "testgroup"
    data = np.random.rand(1, 100, 100).astype("float32")
    ds = xr.DataArray(
        data, dims=["cycle", "y", "x"]).to_dataset(name="data")
    ds.to_zarr(zarr_path, group=group + "/0", mode="w")
    gaussian_blur(zarr_path, group, sigma=1.0)
    out_group = group + "_gbr"
    ds = xr.open_zarr(zarr_path, group=out_group + "/0")
    assert "data" in ds


def test_binarization(tmp_path):
    zarr_path = str(tmp_path / "test_bin.zarr")
    group = "testgroup"
    create_test_data(zarr_path, group)
    binarization(zarr_path, group, threshold=0.5)
    out_group = group + "_bin"
    ds = xr.open_zarr(zarr_path, group=out_group + "/0")
    assert "data" in ds
