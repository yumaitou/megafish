# src/tests/test_napari.py
import pytest
import os
import numpy as np
import xarray as xr
import zarr

from megafish.napari import tile_2d, registered, registered_is, segment as napari_segment


@pytest.mark.skip(reason="Requires napari GUI environment.")
def test_tile_2d(tmp_path):
    # This test is skipped because napari requires a GUI to run.
    # We only check that the function can be called without error in an appropriate environment.

    zarr_path = str(tmp_path / "test_tile_2d.zarr")
    group = "testgroup"
    data = np.random.rand(1, 1, 1, 1, 64, 64).astype("float32")
    ds = xr.DataArray(
        data, dims=["cycle", "tile_y", "tile_x", "z", "y", "x"]).to_dataset(name="data")
    ds.to_zarr(zarr_path, group=group + "/0", mode="w")

    # Just call the function
    tile_2d(zarr_path, groups=[group], pitch=(
        0.1, 0.1), colors=["gray"], limits=[(0, 1)])


@pytest.mark.skip(reason="Requires napari GUI environment.")
def test_registered(tmp_path):
    zarr_path = str(tmp_path / "test_registered.zarr")
    group = "testgroup"
    data = np.random.rand(2, 64, 64).astype("float32")
    ds = xr.DataArray(data, dims=["cycle", "y", "x"]).to_dataset(name="data")
    ds.to_zarr(zarr_path, group=group + "/0", mode="w")

    registered(zarr_path, pitch=(0.1, 0.1), max_level=0,
               groups=[group], colors=["gray"], limits=[(0, 1)])


@pytest.mark.skip(reason="Requires napari GUI environment.")
def test_registered_is(tmp_path):
    zarr_path = str(tmp_path / "test_registered_is.zarr")
    group = "testgroup"
    data = np.random.rand(2, 64, 64).astype("float32")
    ds = xr.DataArray(data, dims=["cycle", "y", "x"]).to_dataset(name="data")
    ds.to_zarr(zarr_path, group=group + "/0", mode="w")

    registered_is(zarr_path, pitch=(0.1, 0.1), max_level=0,
                  groups=[group], colors=["gray"], limits=[(0, 1)])


@pytest.mark.skip(reason="Requires napari GUI environment.")
def test_napari_segment(tmp_path):
    zarr_path = str(tmp_path / "test_napari_segment.zarr")
    group = "testgroup"
    data = np.random.randint(0, 2, size=(2, 64, 64)).astype("uint16")
    ds = xr.DataArray(data, dims=["cycle", "y", "x"]).to_dataset(name="data")
    ds.to_zarr(zarr_path, group=group + "/0", mode="w")

    napari_segment(zarr_path, pitch=(0.1, 0.1), max_level=0, groups=[
                   group], colors=["label"], limits=[(0, 1)])
