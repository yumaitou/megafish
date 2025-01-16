import pytest
import pandas as pd
import os
from megafish.load import make_dirlist, make_imagepath_cYX_from_dirlist, make_imagepath_cYX


def test_make_dirlist(tmp_path):
    # Test make_dirlist function
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "cycle1").mkdir()
    (image_dir / "cycle2").mkdir()

    dirlist_path = str(tmp_path / "dirlist.csv")
    make_dirlist(dirlist_path, str(image_dir))

    assert os.path.exists(dirlist_path), "dirlist.csv should be created."
    df = pd.read_csv(dirlist_path)
    assert len(df) == 2, "There should be two directories listed."


def test_make_imagepath_cYX_from_dirlist(tmp_path):
    # Test make_imagepath_cYX_from_dirlist function
    # Create directories and files
    zarr_path = str(tmp_path / "test.zarr")
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "cycle1").mkdir()
    # Create dummy tif file
    with open(image_dir / "cycle1" / "image_001.tif", "w") as f:
        f.write("dummy")

    dirlist_path = str(tmp_path / "dirlist.csv")
    pd.DataFrame({"folder": [str(image_dir / "cycle1")]}
                 ).to_csv(dirlist_path, index=False)

    groups = ["group1"]
    channels = [1]
    n_cycle = 1
    n_tile_y = 1
    n_tile_x = 1
    scan_type = "snake_down_left"

    make_imagepath_cYX_from_dirlist(
        zarr_path, groups, channels, n_cycle, n_tile_y, n_tile_x, scan_type,
        dirlist_path, ext=".tif")

    imagepath_path = zarr_path.replace(".zarr", "_imagepath.csv")
    assert os.path.exists(imagepath_path), "imagepath CSV should be created."
    df = pd.read_csv(imagepath_path)
    assert len(df) == 1, "There should be one entry in the imagepath CSV."


def test_make_imagepath_cYX(tmp_path):
    # Test make_imagepath_cYX function
    zarr_path = str(tmp_path / "test.zarr")
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "cycle1").mkdir()
    # Create dummy ims file
    with open(image_dir / "cycle1" / "image_001.ims", "w") as f:
        f.write("dummy")

    groups = ["group1"]
    channels = [1]
    n_cycle = 1
    n_tile_y = 1
    n_tile_x = 1
    scan_type = "snake_down_left"

    make_imagepath_cYX(
        zarr_path, groups, channels, n_cycle, n_tile_y, n_tile_x, scan_type,
        str(image_dir), ext=".ims")

    imagepath_path = zarr_path.replace(".zarr", "_imagepath.csv")
    assert os.path.exists(imagepath_path), "imagepath CSV should be created."
    df = pd.read_csv(imagepath_path)
    assert len(df) == 1, "There should be one entry in the imagepath CSV."
