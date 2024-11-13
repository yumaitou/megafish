import os

import dask.config
import numpy as np
import pandas as pd
from tqdm import tqdm
import zarr
import xarray as xr
import dask
from dask.diagnostics import ProgressBar
import dask.array as da
from imaris_ims_file_reader.ims import ims
import tifffile

from .utils import natural_sort, get_tile_yx
from .config import show_resource


def make_dirlist(dirlist_path, image_dir):
    """
    Generates a CSV file listing all cycle directories within a specified image directory.
    Each cycle directory must contain tiled images organized by color, z, y, and x.


    Args:
        dirlist_path (str): The file path to save the generated directory list CSV.
        image_dir (str): The path to the main directory containing subfolders for each cycle.

    Returns:
        None: The function creates a CSV file at dirlist_path.
    """
    dirs = os.listdir(image_dir)
    dirs = [os.path.join(image_dir, dir_) for dir_ in dirs]
    dirs = natural_sort(dirs)
    dirs = [dir_ for dir_ in dirs if os.path.isdir(dir_)]

    df = pd.DataFrame({"folder": dirs})

    if not os.path.exists(os.path.dirname(dirlist_path)):
        os.makedirs(os.path.dirname(dirlist_path))
    df.to_csv(dirlist_path, index=False)


def make_imagepath_cYX_from_dirlist(
        zarr_path, groups, channels, n_cycle, n_tile_y, n_tile_x, scan_type,
        dirlist_path, subfooter="", footer="_imagepath", ext=".tif"):
    """
    Generates a CSV file mapping image paths to cycle, tile, and channel information for spatial-omics images.

    Args:
        zarr_path (str): Path to the base .zarr file.
        groups (list of str): List of group names, each corresponding to a specific analysis group.
        channels (list of str): List of channels corresponding to the groups.
        n_cycle (int): Total number of cycles to process.
        n_tile_y (int): Number of tiles along the y-axis.
        n_tile_x (int): Number of tiles along the x-axis.
        scan_type (str): Type of scan, determining the tile layout.
        dirlist_path (str): Path to the CSV file with the list of cycle directories.
        subfooter (str, optional): String to append before the footer in the output CSV filename; defaults to an empty string.
        footer (str, optional): String appended to the output CSV filename; defaults to "_imagepath".
        ext (str, optional): File extension of the image files; defaults to ".tif".

    Returns:
        None: The function creates a CSV file with the generated image paths and associated metadata at the modified `zarr_path`.
    """
    # Define output path for image paths CSV
    imagepath_path = zarr_path.replace(
        ".zarr", subfooter + footer + ".csv")

    # Generate a list of tile coordinates (y, x) based on scan type
    tile_yxs = get_tile_yx(n_tile_y, n_tile_x, scan_type)

    # Initialize lists to store data for each CSV column
    group_rows = []
    cycle_rows = []
    tile_y_rows = []
    tile_x_rows = []
    path_rows = []
    channel_rows = []

    # Read the list of cycle directories from dirlist_path CSV
    df_dirlist = pd.read_csv(dirlist_path)
    dirs = df_dirlist["folder"].values

    # Iterate over each cycle directory
    for cycle, dir_ in enumerate(dirs):
        # List all files in the cycle directory
        files = os.listdir(dir_)
        for group_name, channel in zip(groups, channels):
            # Filter and sort files with specified extension
            files = [file for file in files if file.endswith(ext)]
            files = natural_sort(files)
            # Map each tile coordinate to a file path
            for tile_yx, file in zip(tile_yxs, files):
                tile_y, tile_x = tile_yx
                path = os.path.join(dir_, file)
                # Append data for each row
                group_rows.append(group_name)
                cycle_rows.append(cycle + 1)
                tile_y_rows.append(tile_y + 1)
                tile_x_rows.append(tile_x + 1)
                path_rows.append(path)
                channel_rows.append(channel)

    # Create a DataFrame and save the output CSV with image paths
    df = pd.DataFrame({
        "group": group_rows, "cycle": cycle_rows,
        "tile_y": tile_y_rows, "tile_x": tile_x_rows,
        "path": path_rows, "channel": channel_rows})
    df.to_csv(imagepath_path, index=False)


def make_imagepath_cYX(
        zarr_path, groups, channels, n_cycle, n_tile_y, n_tile_x, scan_type,
        image_dir, subfooter="", footer="_imagepath", ext=".ims"):
    """
    Generates a CSV file mapping image paths to cycle, tile, and channel information for spatial-omics data.

    Args:
        zarr_path (str): Path to the base .zarr file.
        groups (list of str): List of group names, each corresponding to a specific analysis group.
        channels (list of str): List of channels corresponding to the groups.
        n_cycle (int): Total number of cycles to process.
        n_tile_y (int): Number of tiles along the y-axis.
        n_tile_x (int): Number of tiles along the x-axis.
        scan_type (str): Type of scan, determining the tile layout.
        image_dir (str): Path to the main directory containing subfolders for each cycle, each with images organized by color, z, y, and x.
        subfooter (str, optional): String to append before the footer in the output CSV filename; defaults to an empty string.
        footer (str, optional): String appended to the output CSV filename; defaults to "_imagepath".

    Returns:
        None: The function creates a CSV file with image paths and associated metadata at the modified `zarr_path`.
    """
    # Define output path for image paths CSV
    imagepath_path = zarr_path.replace(
        ".zarr", subfooter + footer + ".csv")

    # Generate tile coordinates (y, x) based on scan type
    tile_yxs = get_tile_yx(n_tile_y, n_tile_x, scan_type)

    # Initialize lists to store data for each CSV column
    group_rows = []
    cycle_rows = []
    tile_y_rows = []
    tile_x_rows = []
    path_rows = []
    channel_rows = []

    # List and sort subdirectories in image_dir, assumed to be cycles
    sub_dirs = os.listdir(image_dir)
    sub_dirs = natural_sort(sub_dirs)

    # Iterate over each cycle directory
    for cycle, sub_dir in enumerate(sub_dirs):

        sub_img_dir = os.path.join(image_dir, sub_dir)
        files = os.listdir(sub_img_dir)
        # Filter and sort files with specified extension
        for group_name, channel in zip(groups, channels):
            files = [file for file in files if file.endswith(ext)]
            files = natural_sort(files)

            # Map each tile coordinate to a file path
            for tile_yx, file in zip(tile_yxs, files):
                tile_y, tile_x = tile_yx
                path = os.path.join(sub_img_dir, file)

                # Append data for each row
                group_rows.append(group_name)
                cycle_rows.append(cycle + 1)
                tile_y_rows.append(tile_y + 1)
                tile_x_rows.append(tile_x + 1)
                path_rows.append(path)
                channel_rows.append(channel)

    # Create a DataFrame and save the output CSV with image paths
    df = pd.DataFrame({
        "group": group_rows, "cycle": cycle_rows,
        "tile_y": tile_y_rows, "tile_x": tile_x_rows,
        "path": path_rows, "channel": channel_rows})

    df.to_csv(imagepath_path, index=False)


def ims_cYXzyx(zarr_path, n_z, n_y, n_x, imagepath_footer="_imagepath"):
    """
    Creates empty Zarr arrays for image data in cycle, tile, and spatial (z, y, x) dimensions,
    then loads .ims images into these arrays using metadata from an image path CSV.

    Args:
        zarr_path (str): Path to the base .zarr file to store image data.
        n_z (int): Number of z-slices per tile.
        n_y (int): Image height (pixels) for each tile.
        n_x (int): Image width (pixels) for each tile.
        imagepath_footer (str, optional): String to append to the CSV filename; defaults to "_imagepath".

    Returns:
        None: The function creates Zarr arrays with image data and writes to `zarr_path`.
    """
    # Define the CSV path based on zarr_path
    imagepath_path = zarr_path.replace(".zarr", imagepath_footer + ".csv")
    # Load image paths and metadata from CSV
    df_imagepath = pd.read_csv(imagepath_path)

    # Determine the number of cycles, tile_y, and tile_x from the CSV data
    n_cycle = df_imagepath["cycle"].max()
    n_tile_y = df_imagepath["tile_y"].max()
    n_tile_x = df_imagepath["tile_x"].max()

    # Unique group names to create datasets for each group
    groups = df_imagepath["group"].unique()

    # Set array dimensions and coordinates for DataArray
    dims = ("cycle", "tile_y", "tile_x", "z", "y", "x")
    coords = {
        "cycle": np.arange(n_cycle),
        "tile_y": np.arange(n_tile_y), "tile_x": np.arange(n_tile_x),
        "z": np.arange(n_z), "y": np.arange(n_y), "x": np.arange(n_x), }
    # Define chunk sizes for optimal storage
    chunks = (1, 1, 1, n_z, n_y, n_x)

    # Initialize and save empty Zarr arrays for each group
    empty_data = da.zeros(
        (n_cycle, n_tile_y, n_tile_x, n_z, n_y, n_x),
        chunks=chunks, dtype=np.uint16)

    print("Saving empty images: ")
    with ProgressBar():
        for group in groups:
            xar = xr.DataArray(empty_data, dims=dims, coords=coords)
            ds = xar.to_dataset(name="data")
            ds.to_zarr(zarr_path, group=group + "/0", mode="w")

    # Define function to load .ims images into zarr array blocks
    def _load_ims_zyx(zar, df_group, block_info=None):
        # Get cycle and tile coordinates for the current block
        cycle = block_info[0]["chunk-location"][0]
        tile_y = block_info[0]["chunk-location"][1]
        tile_x = block_info[0]["chunk-location"][2]

        # Filter DataFrame to obtain metadata for the current block
        df_group = df_group[
            (df_group["cycle"] == cycle + 1) &
            (df_group["tile_y"] == tile_y + 1) &
            (df_group["tile_x"] == tile_x + 1)]

        # If no matching image is found, return a zero array
        if len(df_group) == 0:
            return np.zeros(zar.shape, dtype=np.uint16)

        # Load .ims file for the specified cycle, tile, and channel
        channel = df_group["channel"].values[0] - 1
        path = df_group["path"].values[0]
        img_ims = ims(path)

        # Check .ims image shape and adjust as needed
        if len(img_ims.shape) != 5:  # Expected shape: (zoom, channel, z, y, x)
            print("Unexpected shape " + str(img_ims.shape) + ": " + path)
            return np.zeros(zar.shape, dtype=np.uint16)
        if img_ims.shape[1] < channel + 1:
            print("No channel found: " + path)
            return np.zeros(zar.shape, dtype=np.uint16)
        img = img_ims[0][channel]
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=0)

        # Slice to fit specified dimensions and initialize output with zeros
        img = img[:n_z, :n_y, :n_x]
        output = np.zeros(zar.shape, dtype=np.uint16)
        output[:, :, :, :img.shape[0], :img.shape[1], :img.shape[2]] = img
        return output

    # Load and map images to the Zarr arrays for each group
    for group in groups:
        dar = da.from_zarr(zarr_path, component=group + "/0/data")
        print(f"Loading cYXzyx ims images: {group}" + show_resource())
        group_df = df_imagepath[df_imagepath["group"] == group]

        # Apply _load_ims_zyx function to each block
        res = da.map_blocks(_load_ims_zyx, dar, group_df, dtype=np.uint16)

        # Convert results to DataArray and save to Zarr with appropriate chunking
        with ProgressBar():
            out = xr.DataArray(res, dims=dims, coords=coords)
            out = out.to_dataset(name="data")
            chunks = {"cycle": 1, "tile_y": 1, "tile_x": 1,
                      "z": n_z, "y": n_y, "x": n_x}
            out = out.chunk(chunks=chunks)
            out.to_zarr(zarr_path, mode="w", group=group + "/0")


def tif_cYXzyx(zarr_path, n_z, n_y, n_x, imagepath_footer="_imagepath",
               ext=".tif", dtype=None, tif_dims="czyx"):
    """
    Creates empty Zarr arrays for image data in cycle, tile, and spatial (z, y, x) dimensions,
    then loads TIFF images into these arrays using metadata from an image path CSV.

    Args:
        zarr_path (str): Path to the base .zarr file to store image data.
        n_z (int): Number of z-slices per tile.
        n_y (int): Image height (pixels) for each tile.
        n_x (int): Image width (pixels) for each tile.
        imagepath_footer (str, optional): String to append to the CSV filename; defaults to "_imagepath".
        ext (str, optional): File extension of the image files; defaults to ".tif".
        dtype (str, optional): Data type to cast the image to; defaults to None.
        dims (str, optional): Order of dimensions in the image data; defaults to "czyx".

    Returns:
        None: The function creates Zarr arrays with image data and writes to `zarr_path`.
    """
    # Define the CSV path based on zarr_path
    imagepath_path = zarr_path.replace(".zarr", imagepath_footer + ".csv")
    # Load image paths and metadata from CSV
    df_imagepath = pd.read_csv(imagepath_path)

    # Determine the number of cycles, tile_y, and tile_x from the CSV data
    n_cycle = df_imagepath["cycle"].max()
    n_tile_y = df_imagepath["tile_y"].max()
    n_tile_x = df_imagepath["tile_x"].max()

    # Unique group names to create datasets for each group
    groups = df_imagepath["group"].unique()

    # Set array dimensions and coordinates for DataArray
    dims = ("cycle", "tile_y", "tile_x", "z", "y", "x")
    coords = {
        "cycle": np.arange(n_cycle),
        "tile_y": np.arange(n_tile_y), "tile_x": np.arange(n_tile_x),
        "z": np.arange(n_z), "y": np.arange(n_y), "x": np.arange(n_x), }
    # Define chunk sizes for optimal storage
    chunks = (1, 1, 1, n_z, n_y, n_x)

    if dtype is None:
        dtype = np.uint16

    # Initialize and save empty Zarr arrays for each group
    empty_data = da.zeros(
        (n_cycle, n_tile_y, n_tile_x, n_z, n_y, n_x),
        chunks=chunks, dtype=dtype)

    print("Saving empty images: ")
    with ProgressBar():
        for group in groups:
            xar = xr.DataArray(empty_data, dims=dims, coords=coords)
            ds = xar.to_dataset(name="data")
            ds.to_zarr(zarr_path, group=group + "/0", mode="w")

    # Define function to load .ims images into zarr array blocks
    def _load_tif_zyx(zar, df_group, _dtype, block_info=None):
        # Get cycle and tile coordinates for the current block
        cycle = block_info[0]["chunk-location"][0]
        tile_y = block_info[0]["chunk-location"][1]
        tile_x = block_info[0]["chunk-location"][2]

        # Filter DataFrame to obtain metadata for the current block
        df_group = df_group[
            (df_group["cycle"] == cycle + 1) &
            (df_group["tile_y"] == tile_y + 1) &
            (df_group["tile_x"] == tile_x + 1)]

        # If no matching image is found, return a zero array
        if len(df_group) == 0:
            return np.zeros(zar.shape, dtype=_dtype)

        # Load tif file for the specified cycle, tile, and channel
        channel = df_group["channel"].values[0] - 1
        path = df_group["path"].values[0]
        img_tif = tifffile.imread(path)
        img_tif = img_tif.astype(_dtype)

        # Check tif image shape and adjust as needed
        # Expected shape: (c, y, x) or (c, z, y, x)
        if len(img_tif.shape) not in [3, 4]:
            print("Unexpected shape " + str(img_tif.shape) + ": " + path)
            return np.zeros(zar.shape, dtype=_dtype)
        if img_tif.shape[0] < channel + 1:
            print("No channel found: " + path)
            return np.zeros(zar.shape, dtype=_dtype)
        if tif_dims == "czyx":
            img = img_tif[channel]
        elif tif_dims == "zyxc":
            img = img_tif[:, :, :, channel]
        elif tif_dims == "cyx":
            img = img_tif[channel]
            img = np.expand_dims(img, axis=0)
        elif tif_dims == "yxc":
            img = img_tif[:, :, channel]
            img = np.expand_dims(img, axis=0)
        else:
            raise ValueError("Unsupported tif_dims")

        # Slice to fit specified dimensions and initialize output with zeros
        img = img[:n_z, :n_y, :n_x]
        output = np.zeros(zar.shape, dtype=_dtype)
        output[:, :, :, :img.shape[0], :img.shape[1], :img.shape[2]] = img
        return output

    # Load and map images to the Zarr arrays for each group
    for group in groups:
        dar = da.from_zarr(zarr_path, component=group + "/0/data")
        print(f"Loading cYXzyx tif images: {group}" + show_resource())
        group_df = df_imagepath[df_imagepath["group"] == group]

        # Apply _load_ims_zyx function to each block
        res = da.map_blocks(_load_tif_zyx, dar, group_df, dtype, dtype=dtype)

        # Convert results to DataArray and save to Zarr with appropriate chunking
        with ProgressBar():
            out = xr.DataArray(res, dims=dims, coords=coords)
            out = out.to_dataset(name="data")
            chunks = {"cycle": 1, "tile_y": 1, "tile_x": 1,
                      "z": n_z, "y": n_y, "x": n_x}
            out = out.chunk(chunks=chunks)
            out.to_zarr(zarr_path, mode="w", group=group + "/0")


def stitched_ims(
        zarr_path, group, image_path, channel, n_tile_y, n_tile_x):
    """
    Processes a stitched image by splitting it into tiles and saving them in a Zarr array.

    Args:
        zarr_path (str): Path to the Zarr file where tiled data will be saved.
        group (str): Group name in the Zarr file for storing the tiled image data.
        image_path (str): Path to the stitched image file in .ims format.
        channel (int): Channel index for selecting specific image data.
        n_tile_y (int): Number of tiles along the y-axis.
        n_tile_x (int): Number of tiles along the x-axis.

    Returns:
        None: The function saves tiled images in the specified Zarr group without returning any value.
    """
    # Load stitched image and select specified channel
    print("Loading stitched image: " + image_path)
    stitched_img = ims(image_path)[0, channel]

    # If image has 3D shape, perform max projection across z-axis
    if len(stitched_img.shape) == 3:
        stitched_img = stitched_img.max(axis=0)

    # Define tile dimensions based on image size and tile count
    n_stitched_y, n_stitched_x = stitched_img.shape
    tile_y_size = n_stitched_y // n_tile_y
    tile_x_size = n_stitched_x // n_tile_x

    # Initialize array for storing individual tiles
    tiled_stitched = np.zeros((n_tile_y, n_tile_x, tile_y_size, tile_x_size))

    # Slice the stitched image into tiles and assign to tiled_stitched array
    for y in range(n_tile_y):
        for x in range(n_tile_x):
            tiled_stitched[y, x, :, :] = stitched_img[y * tile_y_size:(
                y + 1) * tile_y_size, x * tile_x_size:(x + 1) * tile_x_size]

    # Convert tiled image to xarray DataArray and configure coordinates and dimensions
    dims = ("tile_y", "tile_x", "y", "x")
    coords = {"tile_y": np.arange(n_tile_y),
              "tile_x": np.arange(n_tile_x),
              "y": np.arange(tile_y_size),
              "x": np.arange(tile_x_size), }

    # Set chunk sizes for storage
    tiled_stitched = xr.DataArray(tiled_stitched, dims=dims, coords=coords)
    tiled_stitched = tiled_stitched.chunk(
        {"tile_y": 1, "tile_x": 1, "y": tile_y_size, "x": tile_x_size})
    tiled_stitched = tiled_stitched.to_dataset(name="data")

    # Save the tiled DataArray to the specified Zarr group
    tiled_stitched.to_zarr(zarr_path, mode="w", group=group + "/0")


def stitched_tif(
        zarr_path, group, image_path, n_tile_y, n_tile_x, dtype="uint16"):
    """
    Processes a stitched TIFF image by splitting it into tiles and saving them in a Zarr array.

    Args:
        zarr_path (str): Path to the Zarr file where tiled data will be saved.
        group (str): Group name in the Zarr file for storing the tiled image data.
        image_path (str): Path to the stitched image file in TIFF format.
        n_tile_y (int): Number of tiles along the y-axis.
        n_tile_x (int): Number of tiles along the x-axis.
        dtype (str, optional): Data type to cast the image to; defaults to "uint16".

    Returns:
        None: The function saves tiled images in the specified Zarr group without returning any value.
    """
    # Load stitched TIFF image and cast to specified dtype
    print("Loading stitched tif image: " + image_path)
    stitched_img = tifffile.imread(image_path)
    stitched_img = stitched_img.astype(dtype)

    # If image has 3D shape, perform max projection across z-axis
    if len(stitched_img.shape) == 3:
        stitched_img = stitched_img.max(axis=0)

    # Define tile dimensions based on image size and tile count
    n_stitched_y, n_stitched_x = stitched_img.shape
    tile_y_size = n_stitched_y // n_tile_y
    tile_x_size = n_stitched_x // n_tile_x

    # Initialize array for storing individual tiles
    tiled_stitched = np.zeros((n_tile_y, n_tile_x, tile_y_size, tile_x_size))

    # Slice the stitched image into tiles and assign to tiled_stitched array
    for y in range(n_tile_y):
        for x in range(n_tile_x):
            tiled_stitched[y, x, :, :] = stitched_img[y * tile_y_size:(
                y + 1) * tile_y_size, x * tile_x_size:(x + 1) * tile_x_size]

    # Convert tiled image to xarray DataArray and configure coordinates and dimensions
    dims = ("tile_y", "tile_x", "y", "x")
    coords = {"tile_y": np.arange(n_tile_y),
              "tile_x": np.arange(n_tile_x),
              "y": np.arange(tile_y_size),
              "x": np.arange(tile_x_size), }

    tiled_stitched = xr.DataArray(tiled_stitched, dims=dims, coords=coords)

    # Set chunk sizes for storage
    tiled_stitched = tiled_stitched.chunk(
        {"tile_y": 1, "tile_x": 1, "y": tile_y_size, "x": tile_x_size})
    tiled_stitched = tiled_stitched.to_dataset(name="data")

    # Save the tiled DataArray to the specified Zarr group
    tiled_stitched.to_zarr(zarr_path, mode="w", group=group + "/0")
