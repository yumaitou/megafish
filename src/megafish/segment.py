import os
import shutil

import numpy as np
import pandas as pd
from tqdm import tqdm
import zarr
import xarray as xr
from dask.diagnostics import ProgressBar
import dask.array as da

import tifffile
from scipy.spatial import cKDTree
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import centroid as sk_centroid
from .config import USE_GPU, show_resource

if USE_GPU:
    import cupy as cp
    from cucim.skimage.morphology import binary_dilation, binary_erosion
    from cucim.skimage.measure import centroid
    from cupyx.scipy.ndimage import gaussian_filter, maximum_filter, \
        binary_fill_holes
else:
    from skimage.morphology import binary_dilation, binary_erosion
    from skimage.measure import centroid
    from scipy.ndimage import gaussian_filter, maximum_filter, \
        binary_fill_holes


def watershed_label(zarr_path, group, min_distance, footer="_wts"):
    """
        Applies watershed segmentation to the image data.

        Args:
            zarr_path (str): Path to the Zarr file containing the image data.
            group (str): Group name in the Zarr file where the data is stored.
            footer (str, optional): Footer string to append to the output Zarr group name; defaults to "_wts".
            min_distance (int): Minimum distance between peaks in the distance map.

        Returns:
            None: The function saves the segmented result to a new Zarr group.
    """
    def _watershed_label(img, min_distance):
        """
        Applies watershed segmentation to the input image.

        Args:
            img (numpy.ndarray): The input image.

        Returns:
            numpy.ndarray: The segmented image.
        """
        # Apply watershed segmentation

        distance = ndimage.distance_transform_edt(img)
        local_maxi = peak_local_max(
            distance, min_distance=min_distance, exclude_border=False)
        local_max_mask = np.zeros_like(img, dtype=bool)
        local_max_mask[tuple(local_maxi.T)] = True
        markers = ndimage.label(local_max_mask)[0]
        labels = watershed(-distance, markers, mask=img)
        res_array = xr.DataArray(
            labels, dims=img.dims, coords=img.coords)
        return res_array

    # Open the Zarr dataset and extract the data array
    ds = xr.open_zarr(zarr_path, group=group + "/0")
    xar = ds["data"]

    # Retrieve the original chunk sizes for the data
    original_chunks = xar.chunks
    chunk_dict = {dim_name: chunk[0]
                  for dim_name, chunk in zip(xar.dims, original_chunks)}

    print("Watershed: " + group + show_resource())

    with ProgressBar():
        # Apply watershed segmentation across image blocks
        wts = xar.map_blocks(_watershed_label, kwargs={
            "min_distance": min_distance}, template=xar)

        # Set the original chunk sizes and save the result as a new Zarr group
        wts = wts.chunk(chunk_dict)
        ds = wts.to_dataset(name="data")
        ds.to_zarr(zarr_path, group=group + footer + "/0", mode="w")


def dilation(zarr_path, group, mask_radius, footer="_dil"):
    """
    Applies a binary dilation operation to the image data.

    Args:
        zarr_path (str): Path to the Zarr file containing the image data.
        group (str): Group name in the Zarr file where the data is stored.
        mask_radius (int): Radius of the structuring element used for dilation.
        footer (str, optional): Footer string to append to the output Zarr group name; defaults to "_dil".

    Returns:
        None: The function saves the dilated result to a new Zarr group.
    """
    def _dilation(img, footprint):
        """
        Applies binary dilation to the input image using the provided footprint.

        Args:
            img (xarray.DataArray): The input image data array.
            footprint (list of tuple): The structuring element used for dilation, with GPU support if available.

        Returns:
            xarray.DataArray: The dilated image.
        """
        img = img.astype(np.float32)

        # Move the image data to GPU if available
        if USE_GPU:
            frm = cp.asarray(img.values)
        else:
            frm = img.values

        # Apply binary dilation
        result = binary_dilation(frm, footprint=footprint)

        # Return the result as an xarray DataArray, moving back to CPU if necessary
        if USE_GPU:
            res_array = xr.DataArray(
                result.get(), dims=img.dims, coords=img.coords)
        else:
            res_array = xr.DataArray(
                result, dims=img.dims, coords=img.coords)
        return res_array

    # Open the Zarr dataset and extract the data array
    ds = xr.open_zarr(zarr_path, group=group + "/0")
    xar = ds["data"]

    # Create a template array for the output
    template = xr.DataArray(
        da.empty_like(xar.data, dtype="float32"),
        dims=xar.dims, coords=xar.coords)

    # Define the structuring element for dilation
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

    with ProgressBar():
        # Apply dilation across image blocks
        dil = xar.map_blocks(
            _dilation, kwargs=dict(footprint=footprint),
            template=template)

        # Set the original chunk sizes and save the result as a new Zarr group
        original_chunks = xar.chunks
        dil = dil.chunk(original_chunks)
        ds = dil.to_dataset(name="data")
        ds.to_zarr(zarr_path, group=group + footer + "/0", mode="w")


def merge_split_label(zarr_path, group, footer="_msl"):
    """
    Merges split labels in an image dataset by applying offsets and resolving conflicts.

    Args:
        zarr_path (str): Path to the Zarr file containing the label image.
        group (str): Group name in the Zarr file where the label image is stored.
        footer (str, optional): Footer string to append to the output Zarr group name; defaults to "_msl".

    Returns:
        None: The function saves the merged label dataset to a new Zarr group.
    """
    def _apply_offset(img, offset, block_info=None):
        """
        Applies an offset to the label values in the image based on the chunk location.

        Args:
            img (numpy.ndarray): The input label image chunk.
            offset (numpy.ndarray): Offset values for each chunk location.
            block_info (dict, optional): Block information from Dask.

        Returns:
            numpy.ndarray: The label image chunk with applied offsets.
        """
        chunk_y = block_info[0]["chunk-location"][0]
        chunk_x = block_info[0]["chunk-location"][1]

        img_offset = img + offset[chunk_y, chunk_x]
        img_offset = img_offset * (img > 0)  # Retain only positive labels

        return img_offset

    def _change_value(img, src_list, dst_list, block_info=None):
        """
        Reassigns label values based on source and destination lists.

        Args:
            img (numpy.ndarray): The input label image chunk.
            src_list (list): List of source label values to change.
            dst_list (list): List of destination values corresponding to the source values.
            block_info (dict, optional): Block information from Dask.

        Returns:
            numpy.ndarray: The modified label image chunk with updated values.
        """
        img_mod = img.copy()
        img_mod_unique = np.unique(img_mod)

        src_list = list(src_list)
        dst_list = list(dst_list)

        # Filter the source list to include only labels present in the image chunk
        present_src = [src for src in src_list if src in img_mod_unique]
        corresponding_dst = [
            dst_list[src_list.index(src)] for src in present_src]

        # Update label values
        for src, dst in zip(present_src, corresponding_dst):
            img_mod[img_mod == src] = dst

        return img_mod

    # Open the Zarr dataset and extract the label data
    root = xr.open_zarr(zarr_path, group=group + "/0")
    xar = root["data"]

    # Retrieve the original chunk sizes
    orignal_chunks = xar.chunks
    chunk_dict = {dim_name: chunk[0]
                  for dim_name, chunk in zip(xar.dims, orignal_chunks)}

    n_chunks = [len(chunk) for chunk in xar.chunks]

    # Get maximum values from each chunk to calculate offsets
    ref = xar.coarsen({'y': chunk_dict['y'], 'x': chunk_dict['x']}).max()
    ref = ref.stack(z=('y', 'x'))
    ref = ref.shift(z=1).fillna(0).astype(int)
    ref = ref.cumsum('z').unstack('z')
    ref = ref.chunk({'y': n_chunks[0], 'x': n_chunks[1]})
    ref = ref.values

    # Load the label data as a Dask array
    dar_img = da.from_zarr(zarr_path, component=group + "/0/data")
    n_y, n_x = dar_img.shape

    print("Merging split labels: " + group)
    with ProgressBar():
        # Apply offsets to the labels
        dar_res = da.map_blocks(
            _apply_offset, dar_img, ref, dtype=np.float32,
            chunks=(chunk_dict["y"], chunk_dict["x"]))

        dims = ["y", "x"]
        coords = {
            "y": range(n_y),
            "x": range(n_x)}
        chunks = {"y": chunk_dict["y"], "x": chunk_dict["x"]}

        out = xr.DataArray(dar_res, dims=dims, coords=coords)
        out = out.chunk(chunks=chunks)

    # Resolve label conflicts across chunks
    dfs = []
    next_pos_x = np.array(xar.chunks[1]).cumsum()[:-1]
    curr_pos_x = next_pos_x - 1
    for c, n in tqdm(zip(curr_pos_x, next_pos_x), total=len(next_pos_x)):
        xar_chunk = out.sel(x=slice(c, n)).values
        # remove rows where at least one of the two columns of xar_boder is 0
        xar_chunk = xar_chunk[xar_chunk[:, 0] * xar_chunk[:, 1] != 0]
        df = pd.DataFrame(xar_chunk)
        df["count"] = 1
        df = df.groupby([0, 1]).count().reset_index()
        # If the values in the second column are the same,
        # keep only the one with the larger count.
        df = df.sort_values(by=[1, "count"], ascending=[True, False])
        df = df.drop_duplicates(subset=[1], keep='first')
        dfs.append(df)

    next_pos_y = np.array(xar.chunks[0]).cumsum()[:-1]
    curr_pos_y = next_pos_y - 1
    for c, n in tqdm(zip(curr_pos_y, next_pos_y), total=len(next_pos_y)):
        xar_chunk = out.sel(y=slice(c, n)).values
        xar_chunk = xar_chunk.T
        xar_chunk = xar_chunk[xar_chunk[:, 0] * xar_chunk[:, 1] != 0]
        df = pd.DataFrame(xar_chunk)
        df["count"] = 1
        df = df.groupby([0, 1]).count().reset_index()
        df = df.sort_values(by=[1, "count"], ascending=[True, False])
        df = df.drop_duplicates(subset=[1], keep='first')
        dfs.append(df)

    df = pd.concat(dfs)

    df = df.sort_values(by=[1, "count"], ascending=[True, False])
    df = df.drop_duplicates(subset=[1], keep='first')

    df.columns = ["dst", "src", "count"]
    df = df[["src", "dst"]].reset_index(drop=True)

    src = df["src"].values
    dst = df["dst"].values

    # Apply the resolved label values to the image
    with ProgressBar():
        dar_res = da.map_blocks(
            _change_value, out.data, src, dst, dtype=np.uint32,
            chunks=(chunk_dict["y"], chunk_dict["x"]))

        dimps = ["y", "x"]
        coords = {
            "y": out.y.values,
            "x": out.x.values}
        chunks = {"y": chunk_dict["y"], "x": chunk_dict["x"]}

        out2 = xr.DataArray(dar_res, dims=dimps, coords=coords)
        out2 = out2.to_dataset(name="data")
        out2 = out2.chunk(chunks=chunks)
        out2.to_zarr(zarr_path, mode="w", group=group + footer + "/0")


def grow_voronoi(zarr_path, group, depth, max_distance, footer="_vor"):
    """
    Expands labeled regions in an image using a Voronoi-like approach, filling the image
    based on the nearest labeled pixel within a specified maximum distance.

    Args:
        zarr_path (str): Path to the Zarr file containing the image data.
        group (str): Group name in the Zarr file where the image data is stored.
        depth (int): The depth of overlap used when applying the Voronoi expansion across chunks.
        max_distance (float): The maximum allowable distance for a pixel to be filled based on the nearest label.
        footer (str, optional): Footer string to append to the output Zarr group name; defaults to "_vor".

    Returns:
        None: This function saves the expanded Voronoi regions as a new Zarr group.
    """
    def _grow_voronoi_tree(img, max_distance):
        """
        Expands labeled regions using a Voronoi-like method within the given maximum distance.

        Args:
            img (numpy.ndarray): The input image with labeled regions.
            max_distance (float): The maximum allowable distance for expansion.

        Returns:
            numpy.ndarray: The expanded image with Voronoi-like regions.
        """
        labels = np.unique(img[img > 0])

        if labels.size == 0:
            return img

        # Get the coordinates of labeled pixels and their corresponding labels
        points = []
        point_labels = []
        for label in labels:
            coords = np.array(np.where(img == label)).T
            points.extend(coords)
            point_labels.extend([label] * len(coords))

        points = np.array(points)
        point_labels = np.array(point_labels)

        # Create a KDTree for fast nearest-neighbor lookup
        tree = cKDTree(points)

        # Generate a grid of coordinates covering the entire image
        xx, yy = np.meshgrid(
            np.arange(img.shape[1]), np.arange(img.shape[0]))
        coords_grid = np.column_stack([yy.ravel(), xx.ravel()])

       # Find the nearest labeled point for each pixel in the grid
        dists, idx = tree.query(coords_grid)

        # Mask out pixels where the distance exceeds the maximum distance
        within_distance = dists <= max_distance
        voronoi_labels = np.zeros(img.shape, dtype=img.dtype)
        voronoi_labels.ravel()[
            within_distance] = point_labels[idx[within_distance]]

        return voronoi_labels

    # Load the image data from Zarr as a Dask array
    dar = da.from_zarr(zarr_path, component=group + "/0/data")

    # Apply the Voronoi expansion function across chunks with overlap
    res = dar.map_overlap(_grow_voronoi_tree, depth=depth,
                          max_distance=max_distance, dtype=dar.dtype)

    print("Growing Voronoi: " + group + show_resource())
    with ProgressBar():
        # Set up dimensions, coordinates, and chunks for the output DataArray
        dims = ["y", "x"]
        coords = {"y": range(dar.shape[0]),
                  "x": range(dar.shape[1])}
        chunks = {"y": dar.chunks[0][0], "x": dar.chunks[1][0]}

        # Create an xarray DataArray from the result and save it as a new Zarr group
        out = xr.DataArray(res, dims=dims, coords=coords)
        out = out.to_dataset(name="data")
        out = out.chunk(chunks=chunks)
        out.to_zarr(zarr_path, mode="w", group=group + footer + "/0")


def masking(zarr_path, group_target, group_mask, reverse=False, footer="_msk"):
    """
    Applies a mask to the target image data, setting values outside the mask to zero.

    Args:
        zarr_path (str): Path to the Zarr file containing the image data.
        group_target (str): Group name in the Zarr file for the target data to be masked.
        group_mask (str): Group name in the Zarr file for the mask data.
        reverse (bool, optional): If True, inverts the mask, setting values within the mask to zero; defaults to False.
        footer (str, optional): Footer string to append to the output Zarr group name; defaults to "_msk".

    Returns:
        None: This function saves the masked result as a new Zarr group.
    """
    # Open the Zarr datasets for the target and mask data
    ds = xr.open_zarr(zarr_path, group=group_target + "/0")
    xar_tgt = ds["data"]

    ds = xr.open_zarr(zarr_path, group=group_mask + "/0")
    xar_msk = ds["data"]

    # Retrieve the original chunk sizes for the target data
    original_chunks = xar_tgt.chunks
    chunk_dict = {dim_name: chunk[0]
                  for dim_name, chunk in zip(xar_tgt.dims, original_chunks)}

    print("Masking: " + group_target + show_resource())
    with ProgressBar():
        # Apply the mask; if reverse is True, invert the mask
        if reverse:
            res = xar_tgt * (1 - (xar_msk > 0))
        else:
            res = xar_tgt * (xar_msk > 0)

        # Re-chunk the result to match the original chunk sizes
        res = res.chunk(chunks=chunk_dict)
        ds = res.to_dataset(name="data")

        # Save the masked data as a new Zarr group
        ds.to_zarr(zarr_path, group=group_target + footer + "/0", mode="w")


def fill_holes(zarr_path, group, footer="_fil"):
    """
    Fills holes in labeled regions of an image dataset.

    Args:
        zarr_path (str): Path to the Zarr file containing the image data.
        group (str): Group name in the Zarr file where the image data is stored.
        footer (str, optional): Footer string to append to the output Zarr group name; defaults to "_fil".

    Returns:
        None: This function saves the image with filled holes as a new Zarr group.
    """
    def _fill_holes(img):
        """
        Fills holes in the labeled regions of the input image.

        Args:
            img (xarray.DataArray): The input image data array.

        Returns:
            xarray.DataArray: The image with holes filled in each labeled region.
        """
        img = img.astype(np.float32)

        # Handle GPU or CPU processing
        if USE_GPU:
            frm = cp.asarray(img.values)
            out = cp.zeros(frm.shape, dtype=frm.dtype)
            ids = cp.unique(frm)
        else:
            frm = img.values
            out = np.zeros(frm.shape, dtype=frm.dtype)
            ids = np.unique(frm)

        # Fill holes for each unique label in the image
        for i in ids:
            mask = frm == i
            result = binary_fill_holes(mask).astype(frm.dtype)
            out += result * i

        # Convert back to an xarray DataArray, moving data back to CPU if needed
        if USE_GPU:
            res_array = xr.DataArray(
                out.get(), dims=img.dims, coords=img.coords)
        else:
            res_array = xr.DataArray(
                out, dims=img.dims, coords=img.coords)

        return res_array

    # Open the Zarr dataset and extract the data array
    ds = xr.open_zarr(zarr_path, group=group + "/0")
    xar = ds["data"]

    # Create a template array for the output with the same dimensions and coordinates
    template = xr.DataArray(
        da.empty_like(xar.data, dtype=xar.dtype),
        dims=xar.dims, coords=xar.coords)

    print("Filling holes: " + group + show_resource())
    with ProgressBar():
        # Apply the hole-filling function across the dataset blocks
        dil = xar.map_blocks(_fill_holes, template=template)

        # Set the original chunk sizes and save the result as a new Zarr group
        original_chunks = xar.chunks
        dil = dil.chunk(original_chunks)
        ds = dil.to_dataset(name="data")
        ds.to_zarr(zarr_path, group=group + footer + "/0", mode="w")


def remove_edge_mask(zarr_path, group, footer="_egr"):
    """
    Removes labeled regions touching the edges of the image.

    Args:
        zarr_path (str): Path to the Zarr file containing the image data.
        group (str): Group name in the Zarr file where the image data is stored.
        footer (str, optional): Footer string to append to the output Zarr group name; defaults to "_egr".

    Returns:
        None: This function saves the image with edge labels removed as a new Zarr group.
    """
    def _remove_edge_mask(img):
        """
        Identifies and removes labeled regions that touch the edges of the image.

        Args:
            img (xarray.DataArray): The input image data array.

        Returns:
            xarray.DataArray: The image with labels touching the edges removed.
        """
        img = img.astype(np.float32)

        # Handle GPU or CPU processing
        if USE_GPU:
            frm = cp.asarray(img.values)
            edge_mask = cp.zeros(frm.shape, dtype=frm.dtype)
        else:
            frm = img.values
            edge_mask = np.zeros(frm.shape, dtype=frm.dtype)

        # Create a mask for the edges of the image
        edge_mask[0, :] = 1
        edge_mask[-1, :] = 1
        edge_mask[:, 0] = 1
        edge_mask[:, -1] = 1

        # Identify labeled regions that touch the edges and remove them
        edges = frm * edge_mask
        ids_edge = cp.unique(edges) if USE_GPU else np.unique(edges)
        for i in ids_edge:
            mask = frm == i
            frm[mask] = 0

        # Convert back to an xarray DataArray, moving data back to CPU if needed
        if USE_GPU:
            res_array = xr.DataArray(
                frm.get(), dims=img.dims, coords=img.coords)
        else:
            res_array = xr.DataArray(
                frm, dims=img.dims, coords=img.coords)

        return res_array

    # Open the Zarr dataset and extract the data array
    ds = xr.open_zarr(zarr_path, group=group + "/0")
    xar = ds["data"]

    # Create a template array for the output with the same dimensions and coordinates
    template = xr.DataArray(
        da.empty_like(xar.data, dtype=xar.dtype),
        dims=xar.dims, coords=xar.coords)

    print("Removing edge mask: " + group + show_resource())
    with ProgressBar():
        # Apply the edge mask removal function across the dataset blocks
        dil = xar.map_blocks(
            _remove_edge_mask, template=template)

        # Set the original chunk sizes and save the result as a new Zarr group
        original_chunks = xar.chunks
        dil = dil.chunk(original_chunks)
        ds = dil.to_dataset(name="data")
        ds.to_zarr(zarr_path, group=group + footer + "/0", mode="w")


def label_edge(zarr_path, group, thickness, footer="_edg"):
    """
    Identifies and labels the edges of labeled regions in an image dataset, with adjustable thickness.

    Args:
        zarr_path (str): Path to the Zarr file containing the image data.
        group (str): Group name in the Zarr file where the image data is stored.
        thickness (int): Thickness of the labeled edges.
        footer (str, optional): Footer string to append to the output Zarr group name; defaults to "_edg".

    Returns:
        None: This function saves the image with labeled edges as a new Zarr group.
    """
    def _label_edge(img, thickness, dtype=np.uint8):
        """
        Detects and labels the edges of labeled regions in the image, applying a specified thickness.

        Args:
            img (numpy.ndarray): The input image with labeled regions.
            thickness (int): Thickness of the edge lines.
            dtype (type, optional): Data type for the output image; defaults to np.uint8.

        Returns:
            numpy.ndarray: The image with labeled edges of specified thickness.
        """
        labels = np.unique(img[img > 0])

        if labels.size == 0:
            return np.zeros(img.shape, dtype=np.uint8)

        # Initialize an empty array for the edges
        edges = np.zeros_like(img, dtype=np.uint8)
        for label_ in labels:
            mask = img == label_
            # Apply binary erosion to identify edges
            ero = binary_erosion(mask)
            edges += mask - ero.astype(np.uint8)

        # Apply a maximum filter to thicken the edges
        edges = edges > 0
        thicken_edges = maximum_filter(edges, size=thickness)

        return thicken_edges

    # Load the image data from Zarr as a Dask array
    dar = da.from_zarr(zarr_path, component=group + "/0/data")

    # Apply the edge detection function across chunks with overlap
    res = dar.map_overlap(_label_edge, depth=10,
                          thickness=thickness, dtype=np.uint8)

    print("Making edge line: " + group + show_resource())
    with ProgressBar():
        # Set up dimensions, coordinates, and chunks for the output DataArray
        dims = ["y", "x"]
        coords = {"y": range(dar.shape[0]),
                  "x": range(dar.shape[1])}
        chunks = {"y": dar.chunks[0][0], "x": dar.chunks[1][0]}

        # Create an xarray DataArray from the result and save it as a new Zarr group
        out = xr.DataArray(res, dims=dims, coords=coords)
        out = out.to_dataset(name="data")
        out = out.chunk(chunks=chunks)
        out.to_zarr(zarr_path, mode="w", group=group + footer + "/0")


def repeat_cycle(zarr_path, group, n_cycle, footer="_rep"):
    """
    Repeats an image dataset over multiple cycles.

    Args:
        zarr_path (str): Path to the Zarr file containing the image data.
        group (str): Group name in the Zarr file where the image data is stored.
        n_cycle (int): Number of times to repeat the dataset over cycles.
        footer (str, optional): Footer string to append to the output Zarr group name; defaults to "_rep".

    Returns:
        None: This function saves the repeated dataset as a new Zarr group.
    """
    # TODO: TOO SLOW
    print("Repeating cycle: " + group + show_resource())
    with ProgressBar():
        # Open the Zarr dataset and extract the data array
        root = xr.open_zarr(zarr_path, group=group + "/0")
        xar = root["data"]
        if len(xar.dims) == 2:
            n_y, n_x = xar.shape
        elif len(xar.dims) == 3:
            _, n_y, n_x = xar.shape

        # Load the data as a Dask array
        dar = da.from_zarr(zarr_path, component=group + "/0/data")
        chunks = dar.chunksize

        # Expand the dataset to all cycles
        if len(dar.shape) == 2:
            res = da.expand_dims(dar, axis=0)
            res = da.repeat(res, n_cycle, 0)
        elif len(dar.shape) == 3:
            res = da.repeat(dar, n_cycle, 0)

        # Create an xarray DataArray with the repeated data
        xar = xr.DataArray(
            res, dims=["cycle", "y", "x"],
            coords={"cycle": range(n_cycle),
                    "y": range(n_y), "x": range(n_x)})
        xar = xar.chunk({"cycle": 1, "y": chunks[-2], "x": chunks[-1]})

        # Convert the DataArray to a dataset and save it as a new Zarr group
        xar = xar.to_dataset(name="data")
        xar.to_zarr(zarr_path, group=group + footer + "/0", mode="w")


def info_csv(zarr_path, group, pitch, footer="_seg"):
    """
    Generates segment information CSV files from image data stored in a Zarr file, summarizing
    properties such as area and centroid for each segment. Merges the CSV files into a single summary.

    Args:
        zarr_path (str): Path to the Zarr file containing the image data.
        group (str): Group name in the Zarr file where the image data is stored.
        pitch (tuple of float): Pixel size in (y, x) directions, used to convert pixel measurements to micrometers.
        footer (str, optional): Footer string to append to the output CSV directory name; defaults to "_seg".

    Returns:
        None: This function saves the segment information as CSV files.
    """
    if len(pitch) != 2:
        raise ValueError("pitch must be a tuple of two floats with (y, x).")

    # Set up the directory for saving CSV files
    csv_dir = zarr_path.replace(".zarr", "_csv")
    csv_dir = os.path.join(csv_dir, group + footer, "0")
    if os.path.exists(csv_dir):
        shutil.rmtree(csv_dir)
    os.makedirs(csv_dir)

    # Load the image data from Zarr as a Dask array
    root = zarr.open(zarr_path)
    zar = root[group + "/0"]["data"]
    dar = da.from_zarr(zar)
    chunk_size_y, chunk_size_x = dar.chunksize

    def _segment_info_csv(dar, csv_dir, block_info=None):
        """
        Extracts segment information such as area and centroid for each label in a chunk
        and saves the information as a CSV file.

        Args:
            dar (numpy.ndarray): The input image chunk.
            csv_dir (str): The directory to save the CSV files.
            block_info (dict, optional): Block information from Dask.

        Returns:
            numpy.ndarray: An empty array as a placeholder.
        """
        chunk_y = block_info[0]["chunk-location"][0]
        chunk_x = block_info[0]["chunk-location"][1]

        label = dar[:, :]
        label_ids = np.unique(label)
        label_ids = label_ids[label_ids > 0]
        if len(label_ids) == 0:
            return np.zeros((1, 1), dtype=np.uint8)

        area = np.zeros(len(label_ids))
        centers = np.zeros((len(label_ids), 2))
        for i, label_id in enumerate(label_ids):
            area[i] = (label == label_id).sum()
            centers[i] = sk_centroid(label == label_id)

        # Create a DataFrame to store segment information
        df = pd.DataFrame({
            "segment_id": label_ids,
            "area": area,
            "centroid_y": centers[:, 0],
            "centroid_x": centers[:, 1]})

        # Save the DataFrame as a CSV file
        csv_path = os.path.join(
            csv_dir, str(chunk_y) + "_" + str(chunk_x) + ".csv")
        df.to_csv(csv_path, index=False)

        return np.zeros((1, 1), dtype=np.uint8)

    def _merge_segment_info_csv(
            zarr_path, group, pitch, chunk_size_y, chunk_size_x,
            column_names=["segment_id", "area", "centroid_y", "centroid_x"],
            sort_values=["segment_id"]):
        """
        Merges individual segment CSV files into a single summary CSV file.

        Args:
            zarr_path (str): Path to the Zarr file.
            group (str): Group name in the Zarr file.
            pitch (tuple of float): Pixel size in (y, x) directions.
            chunk_size_y (int): Chunk size in the y direction.
            chunk_size_x (int): Chunk size in the x direction.
            column_names (list, optional): Column names for the merged DataFrame; defaults to segment properties.
            sort_values (list, optional): Columns to sort by; defaults to ["segment_id"].

        Returns:
            None
        """
        csv_root_dir = zarr_path.replace(".zarr", "_csv")
        csv_dir = os.path.join(csv_root_dir, group, "0")
        csv_files = os.listdir(csv_dir)
        dfs = []
        for csv_file in tqdm(csv_files):
            chunk_y, chunk_x = csv_file.replace(".csv", "").split("_")
            chunk_y = int(chunk_y)
            chunk_x = int(chunk_x)

            csv_path = os.path.join(csv_dir, csv_file)
            df = pd.read_csv(csv_path)
            df["chunk_y"] = chunk_y
            df["chunk_x"] = chunk_x
            dfs.append(df)

        # Concatenate and sort the data
        df = pd.concat(dfs, axis=0)
        df = df.reset_index(drop=True)
        df = df[['chunk_y', 'chunk_x'] + column_names]
        df = df.sort_values(sort_values)

        # Calculate the pixel and micrometer coordinates of centroids
        df["centroid_y_pix"] = \
            df["chunk_y"] * chunk_size_y + df["centroid_y"]
        df["centroid_x_pix"] = \
            df["chunk_x"] * chunk_size_x + df["centroid_x"]

        # Aggregate the data by segment_id
        df = df.groupby("segment_id").agg(
            {"area": "sum", "centroid_y_pix": "mean",
                "centroid_x_pix": "mean"}).reset_index()

        # Convert pixel measurements to micrometers
        df["centroid_y_um"] = df["centroid_y_pix"] * pitch[0]
        df["centroid_x_um"] = df["centroid_x_pix"] * pitch[1]

        df = df.rename(columns={"area": "area_pix2"})
        df["area_um2"] = df["area_pix2"] * pitch[0] * pitch[1]
        df = df[["segment_id", "area_pix2", "area_um2",
                 "centroid_y_pix", "centroid_x_pix",
                 "centroid_y_um", "centroid_x_um"]]

        # Save the merged DataFrame as a CSV file
        sample_name = os.path.splitext(os.path.basename(zarr_path))[0]
        csv_name = sample_name + "_" + group + ".csv"
        csv_path = os.path.join(csv_root_dir, csv_name)
        df.to_csv(csv_path, index=False)

    print("Creating segment info csv: " + group + show_resource())
    with ProgressBar():
        # Apply the segment information extraction function to each block of the dataset
        da.map_blocks(
            _segment_info_csv, dar, csv_dir, dtype=np.uint8).compute()

    # Merge the generated CSV files into a single summary CSV
    _merge_segment_info_csv(zarr_path, group + footer,
                            pitch, chunk_size_y, chunk_size_x)


def merge_groups(zarr_path, groups, output_group):
    """
    Merges multiple groups of image data from a Zarr file into a single output group.

    Args:
        zarr_path (str): Path to the Zarr file containing the image groups.
        groups (list of str): List of group names in the Zarr file to merge.
        output_group (str): The name of the output group where the merged data will be saved.

    Returns:
        None: This function saves the merged dataset as a new group in the Zarr file.
    """
    print("Merging groups: " + str(groups) + show_resource())
    xars = []
    for group in groups:
        # Open each group in the Zarr file and extract the data array
        root = xr.open_zarr(zarr_path, group=group + "/0")
        xar = root["data"]

        # Expand dimensions to include the group as a coordinate for merging
        xar = xar.expand_dims({"group": [group]})
        xars.append(xar)

    # Concatenate the extracted data arrays along the 'group' dimension
    res = xr.concat(xars, dim="group")

    # Save the concatenated result as a new group in the Zarr file
    res.to_zarr(zarr_path, group=output_group + "/0", mode="w")


def normalize_groups(zarr_path, group, footer="_nrm"):
    """
    Normalizes intensity values across groups within an image dataset and computes the maximum intensity 
    projection (MIP).

    Args:
        zarr_path (str): Path to the Zarr file containing the image data.
        group (str): Group name in the Zarr file where the image data is stored.
        footer (str, optional): Footer string to append to the output Zarr group name; defaults to "_nrm".

    Returns:
        None: This function saves the normalized and MIP result as a new Zarr group.
    """
    print("Normalizing groups: " + group + show_resource())

    # Open the Zarr dataset and extract the data array
    ds = xr.open_zarr(zarr_path, group=group + "/0")
    xar = ds["data"]

    # Retrieve the original chunk sizes and prepare a chunking configuration
    original_chunks = xar.chunks
    chunk_dict = {dim_name: chunk[0]
                  for dim_name, chunk in zip(xar.dims, original_chunks)}
    chunk_dict.pop("group")

    # Normalize intensity values for each group
    xar = xar / xar.max(dim=["group", "y", "x"])

    # Compute the maximum intensity projection (MIP) across groups
    res = xar.max(dim="group")

    # Re-chunk the result based on the modified chunk configuration
    res = res.chunk(chunks=chunk_dict)

    # Save the normalized and MIP result as a new Zarr group
    res.to_zarr(zarr_path, group=group + footer + "/0", mode="w")


def select_slice(zarr_path, group, dim, position, chunk_dict=None,
                 footer="_sel"):
    """
    Selects a slice from an image dataset along a specified dimension.

    Args:
        zarr_path (str): Path to the Zarr file containing the image data.
        group (str): Group name in the Zarr file where the image data is stored.
        dim (str): The dimension along which to select the slice (e.g., "z", "cycle").
        position (int): The index position of the slice to select along the specified dimension.
        chunk_dict (dict or None, optional): A dictionary specifying chunk sizes for each dimension.
                                   If None, the original chunk sizes (excluding the selected dimension) are used.
        footer (str, optional): Footer string to append to the output Zarr group name; defaults to "sel".

    Returns:
        None: This function saves the selected slice as a new Zarr group.
    """
    print("Selecting slice: " + group + show_resource())

    # Open the Zarr dataset and extract the data array
    ds = xr.open_zarr(zarr_path, group=group + "/0")

    # If chunk_dict is not provided, use the original chunk sizes (excluding the selected dimension)
    if chunk_dict is None:
        original_chunks = ds["data"].chunks
        chunk_dict = {dim_name: chunk[0]
                      for dim_name, chunk in zip(ds.dims, original_chunks)}
        chunk_dict.pop(dim)

    with ProgressBar():
        # Select the specified slice along the given dimension
        res = ds["data"].isel({dim: position})
        # Drop the dimension from the dataset after slicing
        res = res.drop_vars(dim)

        # Re-chunk the result and save it as a new Zarr group
        res = res.chunk(chunks=chunk_dict)
        res.to_zarr(zarr_path, group=group + footer + "/0", mode="w")


def merge_to_one_group(zarr_path, groups, output_group):
    """
    Merges multiple groups of image data from a Zarr file into a single output group.

    Args:
        zarr_path (str): Path to the Zarr file containing the image groups.
        groups (list of str): List of group names in the Zarr file to merge.
        output_group (str): The name of the output group where the merged data will be saved.

    Returns:
        None: This function saves the merged dataset as a new group in the Zarr file.
    """
    print("Merging groups: " + str(groups) + show_resource())
    xars = []
    for group in groups:
        # Open each group in the Zarr file and extract the data array
        root = xr.open_zarr(zarr_path, group=group + "/0")
        xar = root["data"]

        # Expand dimensions to include the group as a coordinate for merging
        xar = xar.expand_dims({"group": [group]})
        xars.append(xar)

    # Concatenate the extracted data arrays along the 'group' dimension
    res = xr.concat(xars, dim="group")

    # Save the concatenated result as a new group in the Zarr file
    res.to_zarr(zarr_path, group=output_group + "/0", mode="w")


def scaled_mip(zarr_path, group, dim="cycle", footer="_nrm"):
    """
    Normalizes intensity values across a specified dimension and computes the maximum intensity 
    projection (MIP).

    Args:
        zarr_path (str): Path to the Zarr file containing the image data.
        group (str): Group name in the Zarr file where the image data is stored.
        dim (str, optional): The dimension along which to compute the MIP; defaults to "cycle".
        footer (str, optional): Footer string to append to the output Zarr group name; defaults to "_nrm".

    Returns:
        None: This function saves the normalized and MIP result as a new Zarr group.
    """
    print("Normalizing and making mip: " + group + show_resource())

    # Open the Zarr dataset and extract the data array
    ds = xr.open_zarr(zarr_path, group=group + "/0")
    xar = ds["data"]

    # Retrieve the original chunk sizes and prepare a chunking configuration
    original_chunks = xar.chunks
    chunk_dict = {dim_name: chunk[0]
                  for dim_name, chunk in zip(xar.dims, original_chunks)}

    # Normalize intensity values for each group along the specified dimension
    xar = xar / xar.max(dim=[dim, "y", "x"])

    # Compute the maximum intensity projection (MIP) along the specified dimension
    res = xar.max(dim=dim)

    # Remove the specified dimension from the chunking configuration
    chunk_dict.pop(dim)

    # Re-chunk the result based on the modified chunk configuration and save it as a new Zarr group
    res = res.chunk(chunks=chunk_dict)
    res.to_zarr(zarr_path, group=group + footer + "/0", mode="w")
