# src/tests/test_view.py
import pytest
import numpy as np
import xarray as xr
import os

from megafish.view import make_pyramid, mask_edge, max_filter


def test_make_pyramid(tmp_path):
    zarr_path = str(tmp_path / "test_pyramid.zarr")
    group = "testgroup"
    # create data with cycle,y,x dims
    data = np.random.randint(0, 255, (1, 64, 64)).astype("uint8")
    ds = xr.DataArray(data, dims=["cycle", "y", "x"]).to_dataset(name="data")
    ds.to_zarr(zarr_path, group=group + "/0", mode="w")

    make_pyramid(zarr_path, group)
    # Check if level 1 created
    root = xr.open_zarr(zarr_path, group=group + "/1")
    assert "data" in root


def test_mask_edge(tmp_path):
    zarr_path = str(tmp_path / "test_mask_edge.zarr")
    group = "testgroup"
    data = (np.random.rand(64, 64) > 0.5).astype("uint8")
    ds = xr.DataArray(data, dims=["y", "x"]).expand_dims({"cycle": [0]})
    ds = ds.to_dataset(name="data")
    ds.to_zarr(zarr_path, group=group + "/0", mode="w")

    mask_edge(zarr_path, group, radius=1)
    out_group = group + "_edg"
    ds_out = xr.open_zarr(zarr_path, group=out_group + "/0")
    assert "data" in ds_out


def test_max_filter(tmp_path):
    zarr_path = str(tmp_path / "test_max_filter.zarr")
    group = "testgroup"
    # use a simple 2D image to avoid dimension issues
    data = np.random.rand(64, 64).astype("float32")
    ds = xr.DataArray(data, dims=["y", "x"]).to_dataset(name="data")
    ds.to_zarr(zarr_path, group=group + "/0", mode="w")

    # use radius=1 and axes=[0,1] for a 2D image
    max_filter(zarr_path, group_name=group, radius=1, axes=[0, 1])
    out_group = group + "_max"
    ds_out = xr.open_zarr(zarr_path, group=out_group + "/0")
    assert "data" in ds_out
