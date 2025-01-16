# src/tests/test_seqfish.py
import pytest
import numpy as np
import pandas as pd
import xarray as xr
import os

from megafish.seqfish import (
    dog_sds, DoG_filter, local_maxima_footprint, local_maxima,
    select_by_intensity_sd, select_by_intensity_threshold, count_spots,
    count_summary, spot_coordinates, spot_intensity
)


def test_dog_sds():
    NA = 1.4
    wavelength = 0.5
    pitch = 0.1
    dog_sd1, dog_sd2 = dog_sds(NA, wavelength, pitch)
    assert dog_sd1 > 0, "dog_sd1 should be positive."
    assert dog_sd2 > 0, "dog_sd2 should be positive."


@pytest.mark.parametrize("mask_radius", [None, 1])
def test_DoG_filter(tmp_path, mask_radius):
    zarr_path = str(tmp_path / "test_dog.zarr")
    group = "testgroup"
    # create data
    data = np.random.rand(1, 1, 1, 64, 64).astype("float32")
    ds = xr.DataArray(
        data, dims=["cycle", "tile_y", "tile_x", "y", "x"]).to_dataset(name="data")
    ds.to_zarr(zarr_path, group=group + "/0", mode="w")

    DoG_filter(zarr_path, group, 1.0, 2.0,
               axes=[-2, -1], mask_radius=mask_radius)
    out_group = group + "_dog"
    ds_out = xr.open_zarr(zarr_path, group=out_group + "/0")
    assert "data" in ds_out


def test_local_maxima_footprint():
    NA = 1.4
    wavelength = 0.5
    pitch = 0.1
    fp = local_maxima_footprint(NA, wavelength, pitch)
    assert fp.ndim == 2, "Footprint should be 2D."


def test_local_maxima(tmp_path):
    zarr_path = str(tmp_path / "test_lmx.zarr")
    group = "testgroup"
    # use a 2D image
    data = np.random.rand(64, 64).astype("float32")
    ds = xr.DataArray(data, dims=["y", "x"]).to_dataset(name="data")
    ds.to_zarr(zarr_path, group=group + "/0", mode="w")

    fp = np.ones((3, 3))
    # no axes needed, since image is 2D.
    local_maxima(zarr_path, group, fp, axes=[0, 1])
    out_group = group + "_lmx"
    ds_out = xr.open_zarr(zarr_path, group=out_group + "/0")
    assert "data" in ds_out


def test_select_by_intensity_sd(tmp_path):
    zarr_path = str(tmp_path / "test_isd.zarr")
    group = "testgroup"
    data = np.random.rand(64, 64) * 100
    ds = xr.DataArray(data, dims=["y", "x"]).expand_dims(
        {"cycle": [0]}).to_dataset(name="data")
    ds.to_zarr(zarr_path, group=group + "/0", mode="w")

    select_by_intensity_sd(zarr_path, group, sd_factor=1)
    out_group = group + "_isd"
    ds_out = xr.open_zarr(zarr_path, group=out_group + "/0")
    assert "data" in ds_out


def test_select_by_intensity_threshold(tmp_path):
    zarr_path = str(tmp_path / "test_ith.zarr")
    group = "testgroup"
    data = np.random.rand(64, 64) * 100
    ds = xr.DataArray(data, dims=["y", "x"]).expand_dims(
        {"cycle": [0]}).to_dataset(name="data")
    ds.to_zarr(zarr_path, group=group + "/0", mode="w")

    select_by_intensity_threshold(zarr_path, group, threshold=50)
    out_group = group + "_ith"
    ds_out = xr.open_zarr(zarr_path, group=out_group + "/0")
    assert "data" in ds_out


def test_count_summary(tmp_path):
    zarr_path = str(tmp_path / "test_count_summary.zarr")
    group_seg = "segment"
    groups = ["group1", "group2"]
    channels = [1, 2]

    csv_root = zarr_path.replace(".zarr", "_csv")
    os.makedirs(csv_root, exist_ok=True)
    seg_csv = os.path.join(csv_root, os.path.splitext(
        os.path.basename(zarr_path))[0] + "_" + group_seg + ".csv")
    pd.DataFrame({"segment_id": [1], "area_pix2": [100], "area_um2": [10],
                  "centroid_y_pix": [32.5], "centroid_x_pix": [32.5],
                  "centroid_y_um": [3.25], "centroid_x_um": [3.25]}).to_csv(seg_csv, index=False)

    for g in groups:
        g_csv = os.path.join(csv_root, os.path.splitext(
            os.path.basename(zarr_path))[0] + "_" + g + ".csv")
        pd.DataFrame({"cycle": [0], "segment_id": [1], "count": [5]}).to_csv(
            g_csv, index=False)

    count_summary(zarr_path, groups, group_seg, "outgroup", channels)
    out_csv = os.path.join(csv_root, os.path.splitext(
        os.path.basename(zarr_path))[0] + "_outgroup.csv")
    assert os.path.exists(out_csv)


def test_spot_coordinates(tmp_path):
    zarr_path = str(tmp_path / "test_spot_coordinates.zarr")
    group = "spots"
    # shape: (cycle,y,x)
    data = np.zeros((1, 64, 64), dtype="float32")
    data[0, 10, 10] = 50
    ds = xr.DataArray(data, dims=["cycle", "y", "x"]).to_dataset(name="data")
    ds.to_zarr(zarr_path, group=group + "/0", mode="w")

    spot_coordinates(zarr_path, group)
    csv_root = zarr_path.replace(".zarr", "_csv")
    assert os.path.exists(csv_root)


def test_spot_intensity(tmp_path):
    zarr_path = str(tmp_path / "test_spot_intensity.zarr")
    group_spt = "spot"
    group_seg = "seg"
    group_int = "intensity"
    # spot: (cycle,y,x)
    data_spt = np.zeros((1, 64, 64), dtype="float32")
    data_spt[0, 10, 10] = 100
    ds_spt = xr.DataArray(
        data_spt, dims=["cycle", "y", "x"]).to_dataset(name="data")
    ds_spt.to_zarr(zarr_path, group=group_spt + "/0", mode="w")

    # seg
    data_seg = np.ones((64, 64), dtype="int32")
    ds_seg = xr.DataArray(
        data_seg, dims=["y", "x"]).to_dataset(name="data")
    ds_seg.to_zarr(zarr_path, group=group_seg + "/0", mode="w")

    # intensity: same shape (1,64,64)
    data_int = np.random.randint(0, 100, size=(1, 64, 64)).astype("float32")
    ds_int = xr.DataArray(
        data_int, dims=["cycle", "y", "x"]).to_dataset(name="data")
    ds_int.to_zarr(zarr_path, group=group_int + "/0", mode="w")

    spot_intensity(zarr_path, group_spt, group_seg, group_int)
    csv_root = zarr_path.replace(".zarr", "_csv")
    assert os.path.exists(csv_root)
