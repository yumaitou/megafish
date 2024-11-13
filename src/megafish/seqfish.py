import os
import shutil
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from dask.diagnostics import ProgressBar
import dask.array as da
from skimage.morphology import disk
from tqdm import tqdm

from .config import USE_GPU, show_resource

if USE_GPU:
    import cupy as cp
    from cucim.skimage.filters import difference_of_gaussians
    from cucim.skimage.morphology import binary_dilation
    from cupyx.scipy.ndimage import maximum_filter
    from cucim.skimage.measure import regionprops_table, label
else:
    from skimage.filters import difference_of_gaussians
    from skimage.morphology import binary_dilation
    from scipy.ndimage import maximum_filter
    from skimage.measure import regionprops_table, label


def dog_sds(NA, wavelength, pitch, psf_size_factor=1, dog_sd_factor=1):
    """
    Calculates the standard deviations for the Difference of Gaussians (DoG) based on 
    the point spread function (PSF) and imaging parameters.

    Args:
        NA (float): Numerical aperture of the imaging system.
        wavelength (float): Wavelength of the fluorescence (in the same units as pitch).
        pitch (float): Pixel size or sampling interval (in the same units as wavelength).
        psf_size_factor (float, optional): Factor to scale the PSF size; defaults to 1.
        dog_sd_factor (float, optional): Factor to scale the second standard deviation for the DoG; defaults to 1.

    Returns:
        tuple: A tuple containing the two standard deviations (dog_sd1, dog_sd2) for the DoG filter.
    """
    # Calculate the point spread function (PSF) size in pixels
    d_psf_pix = (1.22 * wavelength) / (NA * pitch)
    particle_size = psf_size_factor * d_psf_pix

    # Calculate the standard deviations for the DoG filter
    dog_sd1 = particle_size / (1 + np.sqrt(2))
    dog_sd2 = np.sqrt(2) * dog_sd1 * dog_sd_factor

    return dog_sd1, dog_sd2


def DoG_filter(
        zarr_path, group_name, dog_sd1, dog_sd2, axes, mask_radius=None,
        footer="_dog"):
    """
    Applies a Difference of Gaussians (DoG) filter to the image data in a Zarr file.

    Args:
        zarr_path (str): Path to the Zarr file containing the image data.
        group_name (str): Group name in the Zarr file where the image data is stored.
        dog_sd1 (float): The first standard deviation for the DoG filter.
        dog_sd2 (float): The second standard deviation for the DoG filter.
        axes (list of int): List of axes along which the filter will be applied.
        mask_radius (int, optional): Radius for binary dilation mask to exclude specific regions; defaults to None.
        footer (str, optional): Footer string to append to the output group name; defaults to "_dog".

    Returns:
        None: This function saves the filtered data as a new group in the Zarr file.
    """
    def _DoG_filter(img, low_sigmas, high_sigmas, footprint):
        """
        Applies the DoG filter to a single image chunk.

        Args:
            img (xarray.DataArray): The image chunk to process.
            low_sigmas (numpy.ndarray): Array of low standard deviations for the filter.
            high_sigmas (numpy.ndarray): Array of high standard deviations for the filter.
            footprint (tuple or None): Footprint for binary dilation mask; None if not used.

        Returns:
            xarray.DataArray: The filtered image chunk.
        """
        img = img.astype(np.float32)
        frm = img.values

        # If using GPU, convert to CuPy array
        if USE_GPU:
            frm = cp.asarray(frm)

        # Apply the DoG filter
        result = difference_of_gaussians(frm, low_sigmas, high_sigmas)

        # Apply mask if footprint is provided
        if footprint is not None:
            mask = frm == 0
            mask = binary_dilation(mask, footprint=footprint)
            result[mask] = 0

        if USE_GPU:
            result = result.get()

        # Return the result as an xarray.DataArray
        res_array = xr.DataArray(result, dims=img.dims, coords=img.coords)
        return res_array

    # Open the Zarr dataset and extract the data array
    root = xr.open_zarr(zarr_path, group=group_name + "/0")
    xar = root["data"]

    # Create a template for the output based on the input data
    template = xr.DataArray(
        da.empty_like(xar.data, dtype="float32"),
        dims=xar.dims, coords=xar.coords)

    # Set up the sigmas for the DoG filter based on the axes provided
    low_sigmas = np.zeros(len(xar.dims))
    high_sigmas = np.zeros(len(xar.dims))
    for ax in axes:
        low_sigmas[ax] = dog_sd1
        high_sigmas[ax] = dog_sd2

    # Create a mask footprint if a mask radius is provided
    if mask_radius is not None:
        x_shape = np.ones(len(xar.dims), dtype=np.int8)
        x_shape[-2] = 3
        y_shape = np.ones(len(xar.dims), dtype=np.int8)
        y_shape[-1] = 3

        if USE_GPU:
            footprint = [(cp.ones(tuple(x_shape)), mask_radius),
                         (cp.ones(tuple(y_shape)), mask_radius)]
        else:
            footprint = [(np.ones(tuple(x_shape)), mask_radius),
                         (np.ones(tuple(y_shape)), mask_radius)]

    else:
        footprint = None

    # Apply the DoG filter to the image data
    print("Applying DoG filter: " + group_name + show_resource())
    with ProgressBar():
        # Apply the DoG filter to the image data
        dog = xar.map_blocks(
            _DoG_filter, kwargs=dict(
                low_sigmas=low_sigmas, high_sigmas=high_sigmas,
                footprint=footprint),
            template=template)

        # Re-chunk the filtered data using the original chunk sizes
        original_chunks = xar.chunks
        dog = dog.chunk(original_chunks)

        # Save the filtered data as a new group in the Zarr file
        ds = dog.to_dataset(name="data")
        ds.to_zarr(zarr_path, group=group_name + footer + "/0", mode="w")


def local_maxima_footprint(NA, wavelength_um, pitch_um):
    """
    Calculates the footprint for detecting local maxima based on the point spread function (PSF) 
    and imaging parameters.

    Args:
        NA (float): Numerical aperture of the imaging system.
        wavelength_um (float): Wavelength of the fluorescence (in micrometers).
        pitch_um (float): Pixel size or sampling interval (in micrometers).

    Returns:
        numpy.ndarray: A binary footprint (structuring element) for local maxima detection.
    """
    # Calculate the PSF size in pixels
    d_psf_pix = (1.22 * wavelength_um) / (NA * pitch_um)

    # Determine the minimum distance as half of the PSF size, rounded
    mindist = np.round(d_psf_pix * 0.5)

    # Create a disk-shaped footprint with the calculated radius
    footprint = disk(mindist)

    return footprint


def local_maxima(zarr_path, group_name, footprint, axes, footer="_lmx"):
    """
    Detects local maxima in the image data within a Zarr file using a specified footprint.

    Args:
        zarr_path (str): Path to the Zarr file containing the image data.
        group_name (str): Group name in the Zarr file where the image data is stored.
        footprint (numpy.ndarray): The structuring element used for detecting local maxima.
        axes (list of int): List of axes along which the footprint will be expanded.
        footer (str, optional): Footer string to append to the output group name; defaults to "_lmx".

    Returns:
        None: This function saves the detected local maxima as a new group in the Zarr file.
    """
    if USE_GPU:
        def _local_maxima(img, footprint):
            # Convert the image to a CuPy array for GPU processing
            img_cp = cp.asarray(img.copy())
            lmx = maximum_filter(img_cp, footprint=cp.asarray(footprint))
            lmx[cp.logical_not(lmx == img_cp)] = 0
            # Convert the result back to an xarray.DataArray
            res_array = xr.DataArray(
                lmx.get(), dims=img.dims, coords=img.coords)
            return res_array

    else:
        def _local_maxima(img, footprint):
            # Apply maximum filter to find local maxima
            lmx = maximum_filter(img, footprint=footprint)
            lmx[np.logical_not(lmx == img)] = 0
            res_array = xr.DataArray(lmx, dims=img.dims, coords=img.coords)
            return res_array

    # Open the Zarr dataset and extract the data array
    root = xr.open_zarr(zarr_path, group=group_name + "/0")
    xar = root["data"]

    # Create a template for the output based on the input data
    template = xr.DataArray(
        da.empty_like(xar.data, dtype="float32"),
        dims=xar.dims, coords=xar.coords)

    # Expand the footprint based on the specified axes
    for _ in range(axes[0]):
        footprint = np.expand_dims(footprint, axis=0)

    print("Finding local maxima: " + group_name + show_resource())
    with ProgressBar():
        # Apply the local maxima detection to the image data
        lmx = xar.map_blocks(_local_maxima, kwargs=dict(
            footprint=footprint), template=template)

        # Re-chunk the detected maxima using the original chunk sizes
        original_chunks = xar.chunks
        lmx = lmx.chunk(original_chunks)

        # Save the detected maxima as a new group in the Zarr file
        ds = lmx.to_dataset(name="data")
        ds.to_zarr(zarr_path, group=group_name + footer + "/0", mode="w")


def select_by_intensity_sd(zarr_path, group, sd_factor=0, footer="_isd"):
    """
    Selects spots in the image data based on intensity, using a threshold defined by the mean intensity 
    and standard deviation.

    Args:
        zarr_path (str): Path to the Zarr file containing the image data.
        group (str): Group name in the Zarr file where the image data is stored.
        sd_factor (float, optional): Factor to multiply the standard deviation for the threshold; defaults to 0.
        footer (str, optional): Footer string to append to the output group name; defaults to "_isd".

    Returns:
        None: This function saves the filtered data as a new group in the Zarr file.
    """
    print("Selecting spots by intensity sd: " + group + show_resource())
    with ProgressBar():
        # Open the Zarr dataset and extract the data array
        root = xr.open_zarr(zarr_path, group=group + "/0")
        xar = root["data"]

        # Compute the total intensity and the count of non-zero elements
        total = xar.sum().compute()
        count = (xar != 0).sum().compute()

        # Calculate the mean (average) intensity and standard deviation of non-zero values
        ave = total / count
        sd = (xar != 0).std().compute()

        # Determine the threshold based on the mean and standard deviation factor
        threshold = ave + sd_factor * sd

        # Apply the threshold to select spots and set values below it to 0
        res = xar.where(xar > threshold, 0)

        # Save the filtered data as a new group in the Zarr file
        res.to_zarr(zarr_path, group=group + footer + "/0", mode="w")


def select_by_intensity_threshold(zarr_path, group, threshold=0, footer="_ith"):
    """
    Selects spots in the image data based on a specified intensity threshold.

    Args:
        zarr_path (str): Path to the Zarr file containing the image data.
        group (str): Group name in the Zarr file where the image data is stored.
        threshold (float, optional): Intensity threshold for selecting spots; defaults to 0.
        footer (str, optional): Footer string to append to the output group name; defaults to "_ith".

    Returns:
        None: This function saves the filtered data as a new group in the Zarr file.
    """
    print("Selecting spots by intensity: " + group + show_resource())
    with ProgressBar():
        # Open the Zarr dataset and extract the data array
        root = xr.open_zarr(zarr_path, group=group + "/0")
        xar = root["data"]

        # Apply the intensity threshold to select spots and set values below it to 0
        res = xar.where(xar > threshold, 0)

        # Save the filtered data as a new group in the Zarr file
        res.to_zarr(zarr_path, group=group + footer + "/0", mode="w")


def count_spots(zarr_path, group_spot, group_label, footer="_cnt",
                delete_chunk_csv=True):
    """
    Counts spots within labeled segments in the image data stored in a Zarr file and saves the results as CSV files.

    Args:
        zarr_path (str): Path to the Zarr file containing the image data.
        group_spot (str): Group name for the spot data in the Zarr file.
        group_label (str): Group name for the label data in the Zarr file.
        footer (str, optional): Footer string to append to the output CSV group name; defaults to "_cnt".
        delete_chunk_csv (bool, optional): Whether to delete temporary chunk CSV files after merging; defaults to True.

    Returns:
        None: This function saves the spot counts as CSV files in the csv directory.
    """
    def _count_spot(dar_spt, dar_lbl, csv_dir, block_info=None):
        """
        Counts spots in a single image chunk and saves the result as a CSV file.

        Args:
            dar_spt (numpy.ndarray or cupy.ndarray): The spot data for the chunk.
            dar_lbl (numpy.ndarray or cupy.ndarray): The label data for the chunk.
            csv_dir (str): Directory where the chunk-level CSV file will be saved.
            block_info (dict, optional): Dask block information for chunk location.

        Returns:
            numpy.ndarray: A dummy array for compatibility with Dask.
        """
        # Extract chunk information
        cycle = block_info[0]["chunk-location"][0]
        chunk_y = block_info[0]["chunk-location"][1]
        chunk_x = block_info[0]["chunk-location"][2]

        # Convert to GPU arrays if available
        if USE_GPU:
            label = cp.asarray(dar_lbl[:, :])
            spot = cp.asarray(dar_spt[0, :, :])
        else:
            label = dar_lbl[:, :]
            spot = dar_spt[0, :, :]

    # Assign labels to spot pixels and flatten the array
        label = label.astype(np.int32)
        label[label == 0] = -1
        spot_binary = spot > 0

        label_spot = label * spot_binary
        label_spot_flat = label_spot.flatten()
        label_spot_flat = label_spot_flat[label_spot_flat != 0]

        # Count the unique labels
        if USE_GPU:
            unique, counts = cp.unique(label_spot_flat, return_counts=True)
            df = pd.DataFrame(
                {"segment_id": unique.get(), "count": counts.get()})
        else:
            unique, counts = np.unique(label_spot_flat, return_counts=True)
            df = pd.DataFrame({"segment_id": unique, "count": counts})

        df.loc[df["segment_id"] == -1, "segment_id"] = 0

        # Add zero count for missing segment IDs
        if USE_GPU:
            seg_id = cp.unique(label).get()
            missing_seg_id = cp.array([i for i in seg_id if i not in unique])
            missing_count = cp.zeros(len(missing_seg_id))
            df_missing = pd.DataFrame(
                {"segment_id": missing_seg_id.get(),
                 "count": missing_count.get()})
        else:
            seg_id = np.unique(label)
            missing_seg_id = [i for i in seg_id if i not in unique]
            missing_count = np.zeros(len(missing_seg_id))
            df_missing = pd.DataFrame(
                {"segment_id": missing_seg_id, "count": missing_count})
        df_missing = df_missing[df_missing["segment_id"] != -1]

        df = pd.concat([df, df_missing], axis=0)
        df = df.sort_values(by=["segment_id"])

        # Save the DataFrame to a CSV file for the chunk
        csv_path = os.path.join(
            csv_dir, str(cycle) + "_" + str(chunk_y) + "_" + str(chunk_x) +
            ".csv")
        df.to_csv(csv_path, index=False)

        # Return a dummy array
        return np.zeros((1, 1, 1), dtype=np.uint8)

    def _merge_count_csv(zarr_path, group):
        """
        Merges chunk-level CSV files into a single CSV file for the entire dataset.

        Args:
            zarr_path (str): Path to the Zarr file.
            group (str): Group name used for the output CSV file.

        Returns:
            None: This function saves the merged CSV file in the csv directory.
        """
        csv_root_dir = zarr_path.replace(".zarr", "_csv")
        csv_dir = os.path.join(csv_root_dir, group, "0")
        csv_files = os.listdir(csv_dir)
        dfs = []
        for csv_file in tqdm(csv_files):
            cycle, chunk_y, chunk_x = csv_file.replace(".csv", "").split("_")
            cycle = int(cycle)
            chunk_y = int(chunk_y)
            chunk_x = int(chunk_x)

            csv_path = os.path.join(csv_dir, csv_file)
            df = pd.read_csv(csv_path)
            df["cycle"] = cycle
            df["chunk_y"] = chunk_y
            df["chunk_x"] = chunk_x

            dfs.append(df)

        df = pd.concat(dfs, axis=0)
        df = df.reset_index(drop=True)
        df = df[["cycle", "segment_id", "count"]]

        # Aggregate counts by cycle and segment_id
        df = df.groupby(["cycle", "segment_id"]).sum().reset_index()
        df = df.sort_values(["cycle", "segment_id"])
        sample_name = zarr_path.split("/")[-1].replace(".zarr", "")
        csv_name = sample_name + "_" + group + ".csv"
        csv_path = os.path.join(csv_root_dir, csv_name)
        df.to_csv(csv_path, index=False)

    # Open the Zarr file and extract the spot and label data arrays
    root = zarr.open(zarr_path)
    zar_spt = root[group_spot + "/0"]["data"]
    dar_spt = da.from_zarr(zar_spt)
    zar_lbl = root[group_label + "/0"]["data"]
    dar_lbl = da.from_zarr(zar_lbl)

    # Set up the directory for saving chunk-level CSV files
    csv_root = zarr_path.replace(".zarr", "_csv")
    csv_dir = os.path.join(csv_root, group_spot + footer, "0")
    if os.path.exists(csv_dir):
        shutil.rmtree(csv_dir)
    os.makedirs(csv_dir)

    print("Counting spots: " + group_spot + show_resource())
    with ProgressBar():
        # Apply the counting function to each chunk
        da.map_blocks(
            _count_spot, dar_spt, dar_lbl, csv_dir, dtype=np.uint8).compute()

    # Merge the chunk-level CSV files into a single CSV
    _merge_count_csv(zarr_path, group_spot + footer)

    # Remove the temporary chunk-level CSV files if specified
    if delete_chunk_csv:
        shutil.rmtree(csv_dir)


def count_summary(zarr_path, groups, group_seg, group_out, channels,
                  genename_path=None):
    """
    Summarizes spot counts across multiple groups and cycles, merging the results 
    with segment data and optionally gene names.

    Args:
        zarr_path (str): Path to the Zarr file containing the image data.
        groups (list of str): List of group names in the Zarr file for which spot counts are available.
        group_seg (str): Group name for the segment data.
        group_out (str): Output group name for saving the summary CSV.
        channels (list of int): List of channels corresponding to each group.
        genename_path (str, optional): Path to a CSV file containing gene names for each channel; defaults to None.

    Returns:
        None: This function saves the summarized data as a CSV file.
    """
    print("Summarizing spot counts :" + group_out + show_resource())

    root_dir = zarr_path.replace(".zarr", "_csv")
    sample_name = zarr_path.split("/")[-1].replace(".zarr", "")

    # Load the segment data CSV
    seg_name = sample_name + "_" + group_seg + ".csv"
    df_seg = pd.read_csv(os.path.join(root_dir, seg_name))

    cols = []
    for group, channel in zip(groups, channels):
        csv_path = os.path.join(root_dir, sample_name + "_" + group + ".csv")
        df = pd.read_csv(csv_path)

        # Determine the number of cycles in the dataset
        n_cycle = df["cycle"].max() + 1

        df_cycles = []
        # Process each cycle
        for cycle in range(n_cycle):
            df_cycle = df[df["cycle"] == cycle]
            df_cycle = df_cycle[["segment_id", "count"]]
            col = "ch" + str(channel) + "_cycle" + str(cycle + 1)
            df_cycle = df_cycle.rename(columns={"count": col})
            cols.append(col)
            df_cycles.append(df_cycle)

        # Merge all cycles for the current group
        df = df_cycles[0]
        for i in range(1, len(df_cycles)):
            df = pd.merge(df, df_cycles[i], on="segment_id", how="outer")

        # Merge results from different groups
        if group == groups[0]:
            df_all = df
        else:
            df_all = pd.merge(df_all, df, on="segment_id", how="outer")

    # Remove segment_id 0 (background) and merge with segment data
    df_all = df_all[df_all["segment_id"] != 0]
    df_all = pd.merge(df_all, df_seg, on="segment_id", how="left")

    # Reorganize columns for output
    df_all = df_all[["segment_id", "area_pix2", "area_um2",
                     "centroid_y_pix", "centroid_x_pix",
                     "centroid_y_um", "centroid_x_um"] + cols]

    # If a gene name file is provided, map gene names to channels
    if genename_path is not None:
        df_gene = pd.read_csv(genename_path)
        names = []
        for ch in channels:
            names.extend(df_gene["ch" + str(ch)].values.tolist())
        names_cols = {}
        for col, name in zip(cols, names):
            names_cols[col] = name
        df_all = df_all.rename(columns=names_cols)

    # Save the summarized data to a CSV file
    save_name = sample_name + "_" + group_out + ".csv"
    save_path = os.path.join(root_dir, save_name)
    df_all.to_csv(save_path, index=False)


def spot_coordinates(zarr_path, group, footer="_scd"):
    """
    Extracts coordinates of spots from the image data stored in a Zarr file and saves them as CSV files.

    Args:
        zarr_path (str): Path to the Zarr file containing the image data.
        group (str): Group name in the Zarr file where the spot data is stored.
        footer (str, optional): Footer string to append to the output CSV group name; defaults to "_scd".

    Returns:
        None: This function saves the spot coordinates as CSV files.
    """
    # Set up the directory for saving the CSV files
    csv_dir = zarr_path.replace(".zarr", "_csv")
    csv_dir = os.path.join(csv_dir, group + footer, "0")
    if os.path.exists(csv_dir):
        shutil.rmtree(csv_dir)
    os.makedirs(csv_dir)

    # Open the Zarr dataset and extract the data array
    root = zarr.open(zarr_path)
    zar = root[group + "/0"]["data"]
    dar = da.from_zarr(zar)

    def _spot_coords_csv(dar, csv_dir, block_info=None):
        """
        Extracts spot coordinates from a single chunk and saves them as a CSV file.

        Args:
            dar (numpy.ndarray): The spot data for the chunk.
            csv_dir (str): Directory where the chunk-level CSV file will be saved.
            block_info (dict, optional): Dask block information for chunk location.

        Returns:
            numpy.ndarray: A dummy array for compatibility with Dask.
        """
        # Extract chunk information
        cycle = block_info[0]["chunk-location"][0]
        chunk_y = block_info[0]["chunk-location"][1]
        chunk_x = block_info[0]["chunk-location"][2]

        # Find the coordinates of spots in the image
        img = dar[0, :, :]
        y, x = np.where(img > 0)
        z = np.zeros_like(y)
        df = pd.DataFrame({"y": y, "x": x, "z": z})

        # Adjust coordinates to the center of the pixel
        df["y"] = df["y"] + 0.5
        df["x"] = df["x"] + 0.5

        # If no spots are found, return a dummy array
        if len(df) == 0:
            return np.zeros((1, 1, 1), dtype=np.uint8)

        # Save the DataFrame to a CSV file for the chunk
        csv_path = os.path.join(
            csv_dir, str(cycle) + "_" + str(chunk_y) + "_" + str(chunk_x) +
            ".csv")
        df.to_csv(csv_path, index=False)

        # Return a dummy array
        return np.zeros((1, 1, 1), dtype=np.uint8)

    # Apply the function to each chunk in parallel using Dask
    with ProgressBar():
        da.map_blocks(
            _spot_coords_csv, dar, csv_dir, dtype=np.uint8).compute()


def spot_intensity(zarr_path, group_spt, group_seg, group_int, footer="_sit"):
    """
    Computes the intensity of spots within segmented regions for each chunk of the image data and 
    saves the results as CSV files.

    Args:
        zarr_path (str): Path to the Zarr file containing the image data.
        group_spt (str): Group name for the spot data in the Zarr file.
        group_seg (str): Group name for the segmentation data in the Zarr file.
        group_int (str): Group name for the intensity data in the Zarr file.
        footer (str, optional): Footer string to append to the output CSV group name; defaults to "_sit".

    Returns:
        None: This function saves the spot intensity data as CSV files.
    """
    def _spot_intensity(dar_spt, dar_seg, dar_int, csv_dir, block_info=None):
        """
        Computes the intensity of spots in a single image chunk and saves the result as a CSV file.

        Args:
            dar_spt (numpy.ndarray or cupy.ndarray): The spot data for the chunk.
            dar_seg (numpy.ndarray or cupy.ndarray): The segmentation data for the chunk.
            dar_int (numpy.ndarray or cupy.ndarray): The intensity data for the chunk.
            csv_dir (str): Directory where the chunk-level CSV file will be saved.
            block_info (dict, optional): Dask block information for chunk location.

        Returns:
            numpy.ndarray: A dummy array for compatibility with Dask.
        """
        # Extract chunk information
        cycle = block_info[0]["chunk-location"][0]
        chunk_y = block_info[0]["chunk-location"][1]
        chunk_x = block_info[0]["chunk-location"][2]

        # Squeeze dimensions if necessary and use GPU arrays if available
        img_spt = dar_spt[:, :].squeeze()
        img_seg = dar_seg[:, :]
        img_int = dar_int[:, :].squeeze()

        if USE_GPU:
            img_spt = cp.asarray(img_spt)
            img_seg = cp.asarray(img_seg)
            img_int = cp.asarray(img_int)

        seg_ids = np.unique(img_seg)

        dfs = []
        for seg_id in seg_ids:
            if seg_id == 0:
                continue
            seg_mask = img_seg == seg_id
            img_spt_seg = img_spt * seg_mask

            # Label the spots within the segment
            img_label = label(img_spt_seg > 0)
            props = regionprops_table(
                intensity_image=img_int,
                label_image=img_label,
                properties=["label", "mean_intensity"])

            # Store intensity data in a DataFrame
            if USE_GPU:
                df_spt = pd.DataFrame({
                    "segment_id": seg_id.get(),
                    "spot_id": props["label"].get(),
                    "intensity": props["mean_intensity"].get()})
            else:
                df_spt = pd.DataFrame({
                    "segment_id": seg_id,
                    "spot_id": props["label"],
                    "intensity": props["mean_intensity"]})

            dfs.append(df_spt)

        # Return a dummy array if no data is found
        if len(dfs) == 0:
            return np.zeros((1, 1, 1), dtype=np.uint8)

        df = pd.concat(dfs, axis=0)
        csv_path = os.path.join(
            csv_dir, str(cycle) + "_" + str(chunk_y) + "_" + str(chunk_x) +
            ".csv")
        df.to_csv(csv_path, index=False)

        return np.zeros((1, 1, 1), dtype=np.uint8)

    def _merge_select_csv(
            zarr_path, group, column_names=[
                "segment_id", "spot_id",
                "intensity"],
            sort_values=["cycle", 'chunk_y', 'chunk_x', "segment_id", "spot_id"]):
        """
        Merges chunk-level CSV files into a single CSV file for the entire dataset.

        Args:
            zarr_path (str): Path to the Zarr file.
            group (str): Group name used for the output CSV file.
            column_names (list of str, optional): Column names for the merged DataFrame; defaults to spot-related names.
            sort_values (list of str, optional): Columns to sort the DataFrame; defaults to cycle and spot IDs.

        Returns:
            None: This function saves the merged CSV file.
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

        df = pd.concat(dfs, axis=0)
        df = df.reset_index(drop=True)
        df = df[['cycle', 'chunk_y', 'chunk_x'] + column_names]
        df = df.sort_values(sort_values)

        # Save the merged DataFrame to a CSV file
        sample_name = zarr_path.split("/")[-1].replace(".zarr", "")
        save_name = sample_name + "_" + group + ".csv"
        save_path = os.path.join(csv_root, save_name)
        df.to_csv(save_path, index=False)

    # Load the Zarr data arrays for spots, intensity, and segmentation
    root = zarr.open(zarr_path)
    zar_spt = root[group_spt + "/0"]["data"]
    dar_spt = da.from_zarr(zar_spt)

    zar_int = root[group_int + "/0"]["data"]
    dar_int = da.from_zarr(zar_int)

    zar_seg = root[group_seg + "/0"]["data"]
    dar_seg = da.from_zarr(zar_seg)

    # Set up the directory for saving chunk-level CSV files
    csv_root = zarr_path.replace(".zarr", "_csv")
    csv_dir = os.path.join(csv_root, group_spt + footer, "0")
    if os.path.exists(csv_dir):
        shutil.rmtree(csv_dir)
    os.makedirs(csv_dir)

    # Apply the function to each chunk in parallel using Dask
    print("Computing spot intensity: " + group_spt + show_resource())
    with ProgressBar():
        da.map_blocks(_spot_intensity, dar_spt, dar_seg, dar_int, csv_dir,
                      dtype=np.float32).compute()

    # Merge the chunk-level CSV files into a single CSV
    _merge_select_csv(zarr_path, group_spt + footer)

    # Remove the temporary chunk-level CSV files
    shutil.rmtree(csv_dir)
