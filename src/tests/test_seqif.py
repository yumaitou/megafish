# src/tests/test_seqif.py
import pytest
import numpy as np
import xarray as xr
import os
import pandas as pd

from megafish.seqif import TCEP_subtraction, skip_odd_cycle, get_intensity


def test_TCEP_subtraction(tmp_path):
    zarr_path = str(tmp_path / "test_TCEP_sub.zarr")
    group = "testgroup"
    data = np.random.randint(0, 100, (4, 64, 64)).astype("uint16")
    ds = xr.DataArray(data, dims=["cycle", "y", "x"]).to_dataset(name="data")
    ds.to_zarr(zarr_path, group=group + "/0", mode="w")

    TCEP_subtraction(zarr_path, group)
    out_group = group + "_sub"
    ds_out = xr.open_zarr(zarr_path, group=out_group + "/0")
    assert "data" in ds_out


def test_skip_odd_cycle(tmp_path):
    zarr_path = str(tmp_path / "test_skip_odd.zarr")
    group = "testgroup"
    data = np.random.randint(0, 100, (4, 64, 64)).astype("uint16")
    ds = xr.DataArray(data, dims=["cycle", "y", "x"]).to_dataset(name="data")
    ds.to_zarr(zarr_path, group=group + "/0", mode="w")

    skip_odd_cycle(zarr_path, group)
    out_group = group + "_skc"
    ds_out = xr.open_zarr(zarr_path, group=out_group + "/0")
    # only even cycles remain
    assert ds_out["data"].shape[0] == 2


def test_get_intensity(tmp_path):
    zarr_path = str(tmp_path / "test_get_intensity.zarr")
    group_int = "int"
    group_lbl = "lbl"
    data_int = np.random.randint(0, 100, (2, 64, 64)).astype("uint16")
    ds_int = xr.DataArray(
        data_int, dims=["cycle", "y", "x"]).to_dataset(name="data")
    ds_int.to_zarr(zarr_path, group=group_int + "/0", mode="w")

    data_lbl = np.ones((64, 64), dtype="int32")
    ds_lbl = xr.DataArray(
        data_lbl, dims=["y", "x"])
    ds_lbl = ds_lbl.to_dataset(name="data")
    ds_lbl.to_zarr(zarr_path, group=group_lbl + "/0", mode="w")

    get_intensity(zarr_path, group_int, group_lbl)
    csv_root = zarr_path.replace(".zarr", "_csv")
    assert os.path.exists(csv_root), "CSV should be created for intensity."
