import os
import shutil

from tqdm import tqdm
import dask.config
import numpy as np
import pandas as pd
import zarr
import xarray as xr
import dask
from dask.diagnostics import ProgressBar
import dask.array as da

from .config import USE_GPU, show_resource
if USE_GPU:
    import cupy as cp
    from cucim.skimage.measure import regionprops_table
else:
    from skimage.measure import regionprops_table


def TCEP_subtraction(zarr_path, group, footer="_sub"):
    """
    Subtracts consecutive cycles in the image data, assuming cycles are organized as TCEP and non-TCEP pairs.
    This process highlights the differences between the cycles and removes background noise.

    Args:
        zarr_path (str): Path to the Zarr file containing the image data.
        group (str): Group name in the Zarr file where the image data is stored.
        footer (str, optional): Footer string to append to the output Zarr group name; defaults to "_sub".

    Returns:
        None: This function saves the result of the TCEP subtraction in a new Zarr group.
    """

    # Open the Zarr dataset and retrieve data and chunk sizes
    root = xr.open_zarr(zarr_path, group=group + "/0")
    xar = root["data"]
    dims = xar.dims
    chunk_sizes = [xar.chunksizes[dim][0] for dim in dims]
    chunk_dict = {dim: size for dim, size in zip(dims, chunk_sizes)}

    xar = xar.astype("float32")

    # Select even and odd cycles separately
    xar_even = xar.sel(cycle=xar.cycle % 2 == 0)
    xar_odd = xar.sel(cycle=xar.cycle % 2 == 1)
    xar_even.coords["cycle"] = xar_even.coords["cycle"] // 2
    xar_odd.coords["cycle"] = xar_odd.coords["cycle"] // 2

    # Subtract odd cycles from even cycles
    xar_sub = xar_even - xar_odd
    xar_sub = xar_sub.fillna(0)
    xar_sub = xar_sub.clip(0)
    xar_sub = xar_sub.astype("int16")

    # Remove subtraction artifacts in blank regions
    xar_sub = xar_sub * (xar_odd > 0)
    xar_sub = xar_sub.fillna(0)
    xar_sub = xar_sub.astype("uint16")

    # Re-chunk the dataset according to the original chunk sizes
    xar_sub = xar_sub.chunk(chunk_dict)

    print("Subtracting TCEP: " + group + show_resource())
    with ProgressBar():
        # Save the processed data back to the Zarr file
        ds = xar_sub.to_dataset(name="data")
        ds.to_zarr(zarr_path, group=group + footer + "/0", mode="w")


def skip_odd_cycle(zarr_path, group, footer="_skc"):
    """
    Selects and retains only the even cycles in the dataset, removing all odd cycles.

    Args:
        zarr_path (str): Path to the Zarr file containing the image data.
        group (str): Group name in the Zarr file where the image data is stored.
        footer (str, optional): Footer string to append to the output Zarr group name; defaults to "_skc".

    Returns:
        None: This function saves the dataset with only even cycles in a new Zarr group.
    """

    # Open the Zarr dataset and retrieve data and chunk sizes
    root = xr.open_zarr(zarr_path, group=group + "/0")
    xar = root["data"]
    dims = xar.dims
    chunk_sizes = [xar.chunksizes[dim][0] for dim in dims]
    chunk_dict = {dim: size for dim, size in zip(dims, chunk_sizes)}

    print("Skipping odd cycle: " + group + show_resource())

    # Select only the even cycles from the data
    with ProgressBar():
        xar_res = xar.sel(cycle=xar.cycle % 2 == 0)

        # Re-chunk the dataset according to the original chunk sizes
        xar_res = xar_res.chunk(chunk_dict)

        # Save the processed data back to the Zarr file
        ds = xar_res.to_dataset(name="data")
        ds.to_zarr(zarr_path, group=group + footer + "/0", mode="w")


def get_intensity(zarr_path, group_int, group_lbl, footer="_int"):
    """
    Calculates the mean intensity of labeled segments in an image dataset and saves the results as a CSV file.

    Args:
        zarr_path (str): Path to the Zarr file containing the image data.
        group_int (str): Group name in the Zarr file for the intensity data.
        group_lbl (str): Group name in the Zarr file for the label data.
        footer (str, optional): Footer string to append to the output CSV file and group name; defaults to "_int".

    Returns:
        None: This function saves the computed intensities and segment information as a CSV file.
    """
    print("Getting intensity: " + group_int + show_resource())

    def _get_intensity(dar_int, dar_lbl, csv_dir, block_info=None):
        """
        Internal function to calculate the mean intensity for each segment in a given chunk.
        """
        cycle = block_info[0]["chunk-location"][0]
        chunk_y = block_info[0]["chunk-location"][1]
        chunk_x = block_info[0]["chunk-location"][2]

        img_seg = dar_lbl[:, :]
        img_int = dar_int[0, :, :]

        if USE_GPU:
            img_seg = cp.array(img_seg)
            img_int = cp.array(img_int)

        seg_ids = np.unique(img_seg)
        seg_ids = seg_ids[seg_ids > 0]
        if len(seg_ids) == 0:
            return np.zeros((1, 1, 1), dtype=np.uint8)

        # Calculate properties of segments
        props = regionprops_table(
            intensity_image=img_int,
            label_image=img_seg,
            properties=["label", "area", "intensity_mean"])

        # Convert properties to DataFrame
        if USE_GPU:
            df = pd.DataFrame({
                "segment_id": props["label"].get(),
                "area_pix2": props["area"].get(),
                "mean_intensity": props["intensity_mean"].get()})
        else:
            df = pd.DataFrame({
                "segment_id": props["label"],
                "area_pix2": props["area"],
                "mean_intensity": props["intensity_mean"]})

        df = df[df["segment_id"] > 0]
        df["total_intensity"] = df["mean_intensity"] * df["area_pix2"]
        df = df[["segment_id", "area_pix2", "total_intensity"]]

        # Save the DataFrame to CSV
        csv_path = os.path.join(
            csv_dir, str(cycle) + "_" + str(chunk_y) + "_" + str(chunk_x) +
            ".csv")

        df.to_csv(csv_path, index=False)

        return np.zeros((1, 1, 1), dtype=np.uint8)

    def _merge_select_csv(
            zarr_path, group, column_names=[
                "segment_id", "area_pix2", "total_intensity"],
            sort_values=["cycle", "segment_id"]):
        """
        Merges individual CSV files from each chunk into a single summary CSV file.
        """
        csv_root = zarr_path.replace(".zarr", "_csv")
        csv_dir = os.path.join(csv_root, group, "0")
        csv_files = os.listdir(csv_dir)
        dfs = []
        for csv_file in tqdm(csv_files):
            cycle, chunk_y, chunk_x = csv_file.replace(
                ".csv", "").split("_")
            cycle = int(cycle)
            chunk_y = int(chunk_y)
            chunk_x = int(chunk_x)

            csv_path = os.path.join(csv_dir, csv_file)
            df = pd.read_csv(csv_path)

            if len(df) == 0:
                continue
            df["cycle"] = cycle
            df["chunk_y"] = chunk_y
            df["chunk_x"] = chunk_x

            dfs.append(df)

        # Combine all dataframes into one
        df = pd.concat(dfs, axis=0)
        df = df.reset_index(drop=True)
        df = df[['cycle', 'chunk_y', 'chunk_x'] + column_names]
        df = df.sort_values(sort_values)

        # Aggregate data by cycle and segment ID
        df = df.groupby(["cycle", "segment_id"]).agg(
            {"area_pix2": "sum", "total_intensity": "sum"}).reset_index()
        df["mean_intensity"] = df["total_intensity"] / df["area_pix2"]
        df = df[["cycle", 'segment_id', 'mean_intensity']]

        # Save the merged DataFrame to a CSV file
        sample_name = os.path.splitext(os.path.basename(zarr_path))[0]
        save_name = sample_name + "_" + group + ".csv"
        save_path = os.path.join(csv_root, save_name)
        df.to_csv(save_path, index=False)

    # Load data arrays from Zarr
    root = zarr.open(zarr_path)
    zar_int = root[group_int + "/0"]["data"]
    dar_int = da.from_zarr(zar_int)

    zar_lbl = root[group_lbl + "/0"]["data"]
    dar_lbl = da.from_zarr(zar_lbl)

    # Setup CSV directory
    csv_root = zarr_path.replace(".zarr", "_csv")
    csv_dir = os.path.join(csv_root, group_int + footer, "0")
    if os.path.exists(csv_dir):
        shutil.rmtree(csv_dir)
    os.makedirs(csv_dir)

    # Process intensity data across chunks
    with ProgressBar():
        da.map_blocks(
            _get_intensity, dar_int, dar_lbl, csv_dir,
            dtype=np.uint8).compute()

    # Merge the CSV files
    _merge_select_csv(zarr_path, group_int + footer)

    # Clean up the temporary directory
    shutil.rmtree(csv_dir)


def intnensity_summary(zarr_path, groups, group_seg, group_out, channels,
                       genename_path=None, skip_odd=True):
    """
    Summarizes intensity data for multiple groups and saves the results as a CSV file. 
    Merges intensity values across cycles, segments, and channels, and optionally associates 
    the results with gene names.

    Args:
        zarr_path (str): Path to the Zarr file containing the image data.
        groups (list): List of group names in the Zarr file for intensity data.
        group_seg (str): Group name in the Zarr file for segmentation data.
        group_out (str): Output group name for the summary CSV file.
        channels (list): List of channel numbers corresponding to each group.
        genename_path (str, optional): Path to a CSV file containing gene names per channel.
        skip_odd (bool, optional): If True, odd cycles will be skipped when using gene names; defaults to True.

    Returns:
        None: This function saves the summarized intensity data as a CSV file.
    """
    print("Summarizing intensity: " + group_out + show_resource())

    root_dir = zarr_path.replace(".zarr", "_csv")
    sample_name = os.path.splitext(os.path.basename(zarr_path))[0]

    # Load the segmentation data CSV
    seg_name = sample_name + "_" + group_seg + ".csv"
    df_seg = pd.read_csv(os.path.join(root_dir, seg_name))

    # Process intensity data for each group
    cols = []
    for group, channel in zip(groups, channels):
        csv_path = os.path.join(root_dir, sample_name + "_" + group + ".csv")
        df = pd.read_csv(csv_path)

        n_cycle = df["cycle"].max() + 1

        df_cycles = []
        for cycle in range(n_cycle):
            df_cycle = df[df["cycle"] == cycle]
            df_cycle = df_cycle[["segment_id", "mean_intensity"]]
            col = "ch" + str(channel) + "_cycle" + str(cycle + 1)
            df_cycle = df_cycle.rename(columns={"mean_intensity": col})
            cols.append(col)
            df_cycles.append(df_cycle)

        df = df_cycles[0]
        for i in range(1, len(df_cycles)):
            df = pd.merge(df, df_cycles[i], on="segment_id", how="outer")

        if group == groups[0]:
            df_all = df
        else:
            df_all = pd.merge(df_all, df, on="segment_id", how="outer")

    # Filter out rows where segment_id is 0
    df_all = df_all[df_all["segment_id"] != 0]
    df_all = pd.merge(df_all, df_seg, on="segment_id", how="left")

    # Reorganize and keep relevant columns
    df_all = df_all[["segment_id", "area_pix2", "area_um2",
                     "centroid_y_pix", "centroid_x_pix",
                     "centroid_y_um", "centroid_x_um"] + cols]

    if genename_path is not None:
        df_gene = pd.read_csv(genename_path)
        names = []
        for ch in channels:
            names.extend(df_gene["ch" + str(ch)].values.tolist())
        if skip_odd:
            names = names[::2]  # Skip odd cycles if flag is set
        names_cols = {}
        for col, name in zip(cols, names):
            names_cols[col] = name
        df_all = df_all.rename(columns=names_cols)

    # Save the merged DataFrame as a CSV file
    save_name = sample_name + "_" + group_out + ".csv"
    save_path = os.path.join(root_dir, save_name)
    df_all.to_csv(save_path, index=False)
