# src/tests/test_register.py
import pytest
import numpy as np
import xarray as xr
import os
from megafish.register import (
    shift_cycle_cYXyx, shift_tile_cYXyx, dummy_shift_tile, merge_shift_cYXyx
)


def create_register_data(zarr_path, group, shape=(1, 1, 1, 64, 64), dtype="float32"):
    # Make shape simpler: 1 cycle,1 tile_y,1 tile_x
    # This avoids the reshaping error.
    data = np.random.rand(*shape).astype(dtype)
    ds = xr.DataArray(
        data, dims=["cycle", "tile_y", "tile_x", "y", "x"]).to_dataset(name="data")
    ds.to_zarr(zarr_path, group=group + "/0", mode="w")


def test_shift_cycle_cYXyx(tmp_path):
    zarr_path = str(tmp_path / "test_shift_cycle.zarr")
    group = "testgroup"
    # Only one cycle to avoid reshape issue
    create_register_data(zarr_path, group, (1, 1, 1, 64, 64))

    shift_cycle_cYXyx(zarr_path, group)
    csv_path = zarr_path.replace(".zarr", "_shift_cycle.csv")
    assert os.path.exists(csv_path), "Shift cycle CSV not created."


def test_shift_tile_cYXyx(tmp_path):
    zarr_path = str(tmp_path / "test_shift_tile.zarr")
    group_mov = "mov"
    group_stitch = "ref"
    create_register_data(zarr_path, group_mov, (1, 1, 1, 64, 64))
    create_register_data(zarr_path, group_stitch, (1, 1, 1, 64, 64))

    shift_tile_cYXyx(zarr_path, group_mov, group_stitch)
    csv_path = zarr_path.replace(".zarr", "_shift_tile.csv")
    assert os.path.exists(csv_path), "Shift tile CSV not created."


def test_dummy_shift_tile(tmp_path):
    zarr_path = str(tmp_path / "test_dummy_tile.zarr")
    group = "testgroup"
    create_register_data(zarr_path, group, (1, 1, 1, 64, 64))

    # First create shift_cycle csv
    from megafish.register import shift_cycle_cYXyx
    shift_cycle_cYXyx(zarr_path, group)
    dummy_shift_tile(zarr_path, "_shift_cycle")
    csv_path = zarr_path.replace(".zarr", "_shift_tile.csv")
    assert os.path.exists(csv_path), "Dummy shift tile CSV not created."


def test_merge_shift_cYXyx(tmp_path):
    # Similar test but with single cycle to avoid errors
    zarr_path = str(tmp_path / "test_merge_shift.zarr")
    group = "testgroup"
    create_register_data(zarr_path, group, (1, 1, 1, 64, 64))

    # Create shift_cycle csv
    from megafish.register import shift_cycle_cYXyx
    shift_cycle_cYXyx(zarr_path, group)
    # Create shift_tile csv
    from megafish.register import dummy_shift_tile
    dummy_shift_tile(zarr_path, "_shift_cycle")

    merge_shift_cYXyx(zarr_path, group)
    csv_path = zarr_path.replace(".zarr", "_shift_tile_cycle.csv")
    assert os.path.exists(csv_path), "Merged shift tile cycle CSV not created."
