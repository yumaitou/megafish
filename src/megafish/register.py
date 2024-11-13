import numpy as np
import pandas as pd
import zarr
import xarray as xr
from dask.diagnostics import ProgressBar
import dask.array as da

from skimage.feature import SIFT, match_descriptors
from skimage.measure import ransac

from .config import USE_GPU, show_resource

if USE_GPU:
    import cupy as cp
    from cucim.skimage.registration import phase_cross_correlation
    from cucim.skimage.transform import ProjectiveTransform, AffineTransform, \
        warp
    from cupyx.scipy.ndimage import shift as nd_shift
else:
    from skimage.registration import phase_cross_correlation
    from skimage.transform import ProjectiveTransform, AffineTransform, warp
    from scipy.ndimage import shift as nd_shift


def _get_yx_dims(arr):
    shape = arr.shape
    keep_dims = len(shape) - 2
    slices = [0] * keep_dims + [slice(None), slice(None)]
    return arr[tuple(slices)]


def shift_cycle_cYXyx(
        zarr_path, group, sift_kwargs=None, match_kwargs=None,
        ransac_kwargs=None, subfooter="", footer="_shift_cycle"):
    """
    Calculates and stores cycle shifts for aligning image tiles based on phase correlation and feature matching.
    If the SIFT detector parameters are not provided, only phase correlation is used to calculate shifts.

    Args:
        zarr_path (str): Path to the Zarr file containing image data.
        group (str): Group name in the Zarr file where image data is stored.
        sift_kwargs (dict, optional): Parameters for the SIFT detector; defaults to None.
        match_kwargs (dict, optional): Parameters for matching descriptors; defaults to None.
        ransac_kwargs (dict, optional): Parameters for RANSAC transformation; defaults to None.
        subfooter (str, optional): String to append before the footer in the output CSV filename; defaults to an empty string.
        footer (str, optional): String appended to the output CSV filename; defaults to "_shift_cycle".

    Returns:
        None: The function saves shift information as a CSV file.
    """
    def _shift_cycle(
            mov_tiles, ref_tiles, sift_kwargs=None, match_kwargs=None,
            ransac_kwargs=None):
        """
        Calculates the shift matrix for a single cycle using phase correlation and SIFT feature matching.

        Args:
            mov_tiles (ndarray): Moving image tiles (to be aligned).
            ref_tiles (ndarray): Reference image tiles (alignment target).
            sift_kwargs (dict, optional): Parameters for the SIFT detector.
            match_kwargs (dict, optional): Parameters for matching descriptors.
            ransac_kwargs (dict, optional): Parameters for RANSAC transformation.

        Returns:
            ndarray: A shift matrix for the current cycle.
        """
        # Extract image data and replace NaNs with zeros
        ref_img = _get_yx_dims(ref_tiles)
        ref_img = np.nan_to_num(ref_img)
        mov_img = _get_yx_dims(mov_tiles)
        mov_img = np.nan_to_num(mov_img)

        # Move data to GPU if available
        if USE_GPU:
            ref_img = cp.asarray(ref_img)
            mov_img = cp.asarray(mov_img)

        # Normalize images for phase correlation
        ref_img = (ref_img - ref_img.min()) / (ref_img.max() - ref_img.min())
        mov_img = (mov_img - mov_img.min()) / (mov_img.max() - mov_img.min())

        # Replace NaNs after normalization
        if USE_GPU:
            ref_img = cp.nan_to_num(ref_img)
            mov_img = cp.nan_to_num(mov_img)
        else:
            ref_img = np.nan_to_num(ref_img)
            mov_img = np.nan_to_num(mov_img)

        # Apply phase correlation to estimate shift
        shift, error, diffphase = phase_cross_correlation(
            ref_img, mov_img, normalization=None)  # (y, x)
        H_shift = AffineTransform(translation=(-shift[1], -shift[0]))

        # Initialize the shift matrix with the same shape
        keep_dims = len(mov_tiles.shape) - 2
        shift_matrix_shape = (1,) * keep_dims + (2, 9)
        shift_matrix = np.zeros(shift_matrix_shape, dtype=np.float32)
        append_slices = tuple([0] * (keep_dims + 1) + [slice(None)])

        # If SIFT parameters are not provided, return the basic shift matrix
        if sift_kwargs is None:
            if USE_GPU:
                shift_matrix[append_slices] = H_shift.params.flatten().get()
            else:
                shift_matrix[append_slices] = H_shift.params.flatten()
            return shift_matrix

        # Shift moving image based on phase correlation result
        if USE_GPU:
            mov_img = nd_shift(mov_img, shift).get()
            ref_img = ref_img.get()
        else:
            mov_img = nd_shift(mov_img, shift)

        # Detect keypoints using SIFT
        detector_extractor_ref = SIFT(**sift_kwargs)
        try:
            detector_extractor_ref.detect_and_extract(ref_img)
        except RuntimeError:
            keypoints_ref = np.zeros((0, 2), dtype=np.float32)
        else:
            keypoints_ref = detector_extractor_ref.keypoints  # (x, y)

        detector_extractor_mov = SIFT(**sift_kwargs)
        try:
            detector_extractor_mov.detect_and_extract(mov_img)
        except RuntimeError:
            keypoints_mov = np.zeros((0, 2), dtype=np.float32)
        else:
            keypoints_mov = detector_extractor_mov.keypoints  # (x, y)

        # Match descriptors and compute transformation
        if len(keypoints_ref) == 0 or len(keypoints_mov) == 0:
            H = AffineTransform(translation=(0, 0))
            match_keys_ref = np.zeros((0, 2), dtype=np.float32)
            match_keys_mov = np.zeros((0, 2), dtype=np.float32)
            inliers = np.zeros(0, dtype=bool)
        else:
            matches = match_descriptors(
                keypoints_ref, keypoints_mov, **match_kwargs)

            match_keys_ref = keypoints_ref[matches[:, 0]]  # (x, y)
            match_keys_mov = keypoints_mov[matches[:, 1]]  # (x, y)

            # Apply RANSAC if sufficient matches are found
            if len(match_keys_ref) < 4:
                H = AffineTransform(translation=(0, 0))
            else:
                H, inliers = ransac(
                    (np.flip(match_keys_mov, axis=-1),
                        np.flip(match_keys_ref, axis=-1)),
                    ProjectiveTransform, **ransac_kwargs)

        # Compute the final transformation and update the shift matrix
        if USE_GPU:
            H_inv = cp.linalg.inv(cp.asarray(H.params))
            H = ProjectiveTransform(H_shift.params @ H_inv)
            shift_matrix[append_slices] = H.params.flatten().get()
        else:
            H_inv = np.linalg.inv(H.params)
            H = ProjectiveTransform(H_shift.params @ H_inv)
            shift_matrix[append_slices] = H.params.flatten()

        return shift_matrix

    # Open Zarr file and get data array
    root = zarr.open(zarr_path)
    zr = root[group + "/0"]["data"]
    da_zr = da.from_zarr(zr)
    n_cycle, n_tile_y, n_tile_x, n_y, n_x = da_zr.shape

    # Reference tiles for alignment
    ref_tiles = da_zr[0, :, :, :, :]

    # Map shifts across cycles
    shifts = da.map_blocks(
        _shift_cycle, da_zr, ref_tiles, sift_kwargs, match_kwargs,
        ransac_kwargs,
        dtype=np.float32, chunks=(1, 1, 1, 2, 9))

    print("Calculating cycle shifts: " + group + show_resource())
    with ProgressBar():
        shift_matrix = shifts.compute()

    # Reshape shift matrix for saving as CSV
    n_rows = n_cycle * n_tile_y * n_tile_x
    n_cols = 9 * 2
    n_indices = 3
    shift_matrix_reshape = shift_matrix.reshape(n_rows, n_cols)[:, :9]
    index_matrix = np.indices(
        (n_cycle, n_tile_y, n_tile_x)).reshape(n_indices, n_rows).T

    shift_cols = ["shift_" + str(i) for i in range(9)]

    index_shift_matrix = np.concatenate(
        (index_matrix, shift_matrix_reshape), axis=1)
    shifts_df = pd.DataFrame(
        index_shift_matrix,
        columns=["cycle", "tile_y", "tile_x"] + shift_cols)

    # Save the shift matrix as a CSV file
    shifts_df.to_csv(zarr_path.replace(
        ".zarr", subfooter + footer + ".csv"), index=False)


def shift_tile_cYXyx(
        zarr_path, group_mov, group_stitched, max_shift=100, sift_kwargs=None,
        match_kwargs=None, ransac_kwargs=None, subfooter="",
        footer="_shift_tile"):
    """
    Calculates and stores tile shifts for aligning image tiles based on phase correlation and feature matching.
    If the SIFT detector parameters are not provided, only phase correlation is used to calculate shifts.

    Args:
        zarr_path (str): Path to the Zarr file containing image data.
        group_mov (str): Group name in the Zarr file for the moving image data.
        group_stitched (str): Group name in the Zarr file for the reference stitched image data.
        max_shift (int, optional): Maximum allowed shift (in pixels); defaults to 100.
        sift_kwargs (dict, optional): Parameters for the SIFT detector; defaults to None.
        match_kwargs (dict, optional): Parameters for matching descriptors; defaults to None.
        ransac_kwargs (dict, optional): Parameters for RANSAC transformation; defaults to None.
        subfooter (str, optional): String to append before the footer in the output CSV filename; defaults to an empty string.
        footer (str, optional): String appended to the output CSV filename; defaults to "_shift_tile".

    Returns:
        None: The function saves the shift matrix as a CSV file without returning any value.
    """
    def _shift_tile(img, max_shift, sift_kwargs, match_kwargs, ransac_kwargs):
        """
        Calculates the shift matrix for a single tile using phase correlation and SIFT feature matching.

        Args:
            img (xarray.DataArray): Image tiles with dimensions for reference and moving images.
            max_shift (int): Maximum allowed shift (in pixels).
            sift_kwargs (dict, optional): Parameters for the SIFT detector.
            match_kwargs (dict, optional): Parameters for matching descriptors.
            ransac_kwargs (dict, optional): Parameters for RANSAC transformation.

        Returns:
            xarray.DataArray: A shift matrix for the current tile.
        """
        # Extract and preprocess reference and moving images
        ref_tiles = img.sel(refmov="ref").values
        ref_img = _get_yx_dims(ref_tiles)
        ref_img = np.nan_to_num(ref_img)

        mov_tiles = img.sel(refmov="mov").values
        mov_img = _get_yx_dims(mov_tiles)
        mov_img = np.nan_to_num(mov_img)

        # Move to GPU if available
        if USE_GPU:
            ref_img = cp.asarray(ref_img)
            mov_img = cp.asarray(mov_img)

        # Normalize images for phase correlation
        ref_img = (ref_img - ref_img.min()) / (ref_img.max() - ref_img.min())
        mov_img = (mov_img - mov_img.min()) / (mov_img.max() - mov_img.min())

        # Replace NaNs after normalization
        if USE_GPU:
            ref_img = cp.nan_to_num(ref_img)
            mov_img = cp.nan_to_num(mov_img)
        else:
            ref_img = np.nan_to_num(ref_img)
            mov_img = np.nan_to_num(mov_img)

        # Apply phase correlation to estimate shift
        shift, error, diffphase = phase_cross_correlation(
            ref_img, mov_img, normalization=None)  # (y, x)
        H_shift = AffineTransform(translation=(-shift[1], -shift[0]))

        # Initialize the shift matrix
        shift_matrix = np.zeros((1, 1, 9), dtype=np.float32)
        if sift_kwargs is None:
            if USE_GPU:
                shift_matrix[0, 0, :] = H_shift.params.flatten().get()
            else:
                shift_matrix[0, 0, :] = H_shift.params.flatten()
        else:
            # Shift moving image based on phase correlation result
            if USE_GPU:
                mov_img = nd_shift(mov_img, shift).get()
                ref_img = ref_img.get()
            else:
                mov_img = nd_shift(mov_img, shift)

            # Detect keypoints using SIFT
            detector_extractor_ref = SIFT(**sift_kwargs)
            try:
                detector_extractor_ref.detect_and_extract(ref_img)
            except RuntimeError:
                keypoints_ref = np.zeros((0, 2), dtype=np.float32)
            else:
                keypoints_ref = detector_extractor_ref.keypoints  # (x, y)

            detector_extractor_mov = SIFT(**sift_kwargs)
            try:
                detector_extractor_mov.detect_and_extract(mov_img)
            except RuntimeError:
                keypoints_mov = np.zeros((0, 2), dtype=np.float32)
            else:
                keypoints_mov = detector_extractor_mov.keypoints  # (x, y)

            # Match keypoints and estimate transformation
            if len(keypoints_ref) == 0 or len(keypoints_mov) == 0:
                H = AffineTransform(translation=(0, 0))
                match_keys_ref = np.zeros((0, 2), dtype=np.float32)
                match_keys_mov = np.zeros((0, 2), dtype=np.float32)
                inliers = np.zeros(0, dtype=bool)
            else:
                matches = match_descriptors(
                    keypoints_ref, keypoints_mov, **match_kwargs)

                match_keys_ref = keypoints_ref[matches[:, 0]]  # (x, y)
                match_keys_mov = keypoints_mov[matches[:, 1]]  # (x, y)

                if len(match_keys_ref) < 4:
                    H = AffineTransform(translation=(0, 0))
                else:
                    H, inliers = ransac(
                        (np.flip(match_keys_mov, axis=-1),
                         np.flip(match_keys_ref, axis=-1)),
                        ProjectiveTransform, **ransac_kwargs)

            # Apply the transformation and enforce max shift constraint
            if USE_GPU:
                H_inv = cp.linalg.inv(cp.asarray(H.params))
                H = ProjectiveTransform(H_shift.params @ H_inv)

                if cp.linalg.norm(H.params[0:2, 2]) > max_shift:
                    H = AffineTransform(translation=(0, 0))
                shift_matrix[0, 0, :] = H.params.flatten().get()
            else:
                H_inv = np.linalg.inv(H.params)
                H = ProjectiveTransform(H_shift.params @ H_inv)

                if np.linalg.norm(H.params[0:2, 2]) > max_shift:
                    H = AffineTransform(translation=(0, 0))
                shift_matrix[0, 0, :] = H.params.flatten()

        # Return the shift matrix as an xarray DataArray
        res = xr.DataArray(
            shift_matrix,
            dims=["tile_y", "tile_x", "shift"],
            coords={
                "tile_y": img.coords["tile_y"],
                "tile_x": img.coords["tile_x"],
                "shift": np.arange(9)})
        return res

    # Open Zarr groups for reference and moving images
    group_ref = group_stitched + "/0"
    root = xr.open_zarr(zarr_path, group=group_ref)
    xar_ref = root["data"]

    xar_ref = xar_ref.expand_dims(dim={"refmov": ["ref"]}, axis=[0])

    group_mov = group_mov + "/0"
    root = xr.open_zarr(zarr_path, group=group_mov)
    xar_mov = root["data"]
    xar_mov = xar_mov.isel(cycle=0)  # TODO
    n_tile_y, n_tile_x, n_y, n_x = xar_mov.shape
    xar_mov = xar_mov.expand_dims(dim={"refmov": ["mov"]}, axis=[0])

    # Concatenate reference and moving images
    xar_in = xr.concat([xar_ref, xar_mov], dim="refmov")
    xar_in = xar_in.chunk({
        "refmov": 2, "tile_y": 1, "tile_x": 1, "y": n_y, "x": n_x})

    # Set up a template for the output shifts
    new_dims = ["tile_y", "tile_x", "shift"]
    new_coords = {
        "tile_y": np.arange(n_tile_y),
        "tile_x": np.arange(n_tile_x),
        "shift": np.arange(9)}
    template = xr.DataArray(
        da.empty((n_tile_y, n_tile_x, 9), dtype=np.float32, chunks=(1, 1, 9)),
        dims=new_dims, coords=new_coords)

    # Map shifts across tiles
    res = xar_in.map_blocks(
        _shift_tile,
        kwargs={
            "max_shift": max_shift,
            "sift_kwargs": sift_kwargs,
            "match_kwargs": match_kwargs,
            "ransac_kwargs": ransac_kwargs},
        template=template)

    print("Calculating tile shifts: " + group_mov + show_resource())
    with ProgressBar():
        res = res.compute()

    # Reshape and save the shift matrix as a CSV
    n_tile_y, n_tile_x, n_shift = res.shape
    n_rows = n_tile_y * n_tile_x
    n_cols = 9
    shift_matrix = res.values.reshape(n_rows, n_cols)

    index_matrix = np.indices((n_tile_y, n_tile_x)).reshape(
        2, n_rows).T

    sift_cols = ["shift_" + str(i) for i in range(9)]
    index_shift_matrix = np.concatenate(
        (index_matrix, shift_matrix), axis=1)
    shifts_df = pd.DataFrame(
        index_shift_matrix,
        columns=["tile_y", "tile_x"] + sift_cols)

    shifts_df.to_csv(zarr_path.replace(
        ".zarr", subfooter + footer + ".csv"), index=False)


def dummy_shift_tile(zarr_path, shift_cycle_footer, subfooter="",
                     footer="_shift_tile"):
    """
    Creates a dummy tile shifts CSV file with identity transformation values.

    Args:
        zarr_path (str): Path to the Zarr file associated with the image data.
        shift_cycle_footer (str): Footer of the shift cycle CSV file to read and modify.
        subfooter (str, optional): String to append before the footer in the output CSV filename; defaults to an empty string.
        footer (str, optional): String appended to the output CSV filename; defaults to "_shift_tile".

    Returns:
        None: The function saves a dummy shifts CSV file.
    """
    print("Creating dummy tile shifts csv")

    # Read the shift cycle CSV file
    shift_cycle_path = zarr_path.replace(".zarr", shift_cycle_footer + ".csv")
    df = pd.read_csv(shift_cycle_path)

    # Remove the 'cycle' column
    df = df[df["cycle"] == 0]
    df = df.drop(columns=["cycle"])

    # Set all shift values to create an identity transformation
    df["shift_0"] = 1
    df["shift_1"] = 0
    df["shift_2"] = 0
    df["shift_3"] = 0
    df["shift_4"] = 1
    df["shift_5"] = 0
    df["shift_6"] = 0
    df["shift_7"] = 0
    df["shift_8"] = 1

    # Save the modified DataFrame as a new CSV file
    df.to_csv(zarr_path.replace(
        ".zarr", subfooter + footer + ".csv"), index=False)


def merge_shift_cYXyx(
        zarr_path, group, subfooter="", cycle_footer="_shift_cycle",
        tile_footer="_shift_tile", footer="_shift_tile_cycle"):
    """
    Merges cycle and tile shift transformations and saves the combined shifts as a CSV file.

    Args:
        zarr_path (str): Path to the Zarr file associated with the image data.
        group (str): Group name in the Zarr file where image data is stored.
        subfooter (str, optional): String to append before the footer in the output CSV filename; defaults to an empty string.
        cycle_footer (str, optional): Footer of the cycle shift CSV file; defaults to "_shift_cycle".
        tile_footer (str, optional): Footer of the tile shift CSV file; defaults to "_shift_tile".
        footer (str, optional): String appended to the output CSV filename; defaults to "_shift_tile_cycle".

    Returns:
        None: The function saves the merged shifts as a CSV file.
    """
    # Paths for the shift files
    shift_tile_path = zarr_path.replace(
        ".zarr", subfooter + tile_footer + ".csv")
    shift_cycle_path = zarr_path.replace(
        ".zarr", subfooter + cycle_footer + ".csv")

    # Load cycle and tile shift data
    shifts_tile_df = pd.read_csv(shift_tile_path)
    shifts_cycle_df = pd.read_csv(shift_cycle_path)

    # Load image data from the Zarr file
    ds = xr.open_zarr(zarr_path, group=group + "/0")
    xar = ds["data"]
    n_cycle, n_tile_y, n_tile_x, n_y, n_x = xar.shape

    shift_cols = ["shift_" + str(i) for i in range(9)]

    cycles = []
    tiles_y = []
    tiles_x = []
    shifts = []

    # Iterate through all cycles, tiles in y, and tiles in x dimensions
    for cycle in range(n_cycle):
        for tile_y in range(n_tile_y):
            for tile_x in range(n_tile_x):
                # Filter the tile and cycle shift data for the current cycle, tile_y, and tile_x
                shift_tile = shifts_tile_df[
                    (shifts_tile_df["tile_y"] == tile_y) &
                    (shifts_tile_df["tile_x"] == tile_x)
                ]
                shift_cycle = shifts_cycle_df[
                    (shifts_cycle_df["cycle"] == cycle) &
                    (shifts_cycle_df["tile_y"] == tile_y) &
                    (shifts_cycle_df["tile_x"] == tile_x)
                ]

                # Extract the transformation matrices from the filtered data
                shift_tile = shift_tile[shift_cols].values
                shift_cycle = shift_cycle[shift_cols].values

                # Reshape matrices into 3x3 format
                H_tile = shift_tile.reshape(3, 3)
                H_cycle = shift_cycle.reshape(3, 3)

                # Apply the combined transformation (cycle shift followed by tile shift)
                H = H_cycle @ H_tile

                # Store the combined transformation and indices
                shifts.append(H.flatten())
                cycles.append(cycle)
                tiles_y.append(tile_y)
                tiles_x.append(tile_x)

    # Combine indices and shifts into a DataFrame
    index_array = np.array([cycles, tiles_y, tiles_x]).T
    shifts = np.array(shifts)
    index_shifts = np.concatenate((index_array, shifts), axis=1)
    shifts_df = pd.DataFrame(
        index_shifts,
        columns=["cycle", "tile_y", "tile_x"] + shift_cols)

    # Save the merged shift data as a CSV file
    shifts_df.to_csv(zarr_path.replace(
        ".zarr", subfooter + footer + ".csv"), index=False)


def get_edges(n_cycle, n_tile_y, n_tile_x, df_shift, n_y_stitched,
              n_x_stitched, n_y, n_x, margin=500):
    """
    Calculates the edges of tiles after applying transformations, taking into account tile positions and transformations.

    Args:
        n_cycle (int): Number of cycles.
        n_tile_y (int): Number of tiles along the y-axis.
        n_tile_x (int): Number of tiles along the x-axis.
        df_shift (pandas.DataFrame): DataFrame containing shift transformation matrices for each cycle and tile.
        n_y_stitched (int): Height of each stitched tile.
        n_x_stitched (int): Width of each stitched tile.
        n_y (int): Height of each individual tile.
        n_x (int): Width of each individual tile.
        margin (int, optional): Margin added to the edges for buffer space; defaults to 500.

    Returns:
        pandas.DataFrame: A DataFrame containing the min and max x and y coordinates of each tile after transformation, including the cycle, tile_y, and tile_x.
    """
    edges = []

    # Iterate through all cycles, tiles in y, and tiles in x dimensions
    for cycle in range(n_cycle):
        for tile_y in range(n_tile_y):
            for tile_x in range(n_tile_x):
                # Define the original tile edges (corners)
                edge = ((0, 0), (n_x, 0), (0, n_y), (n_x, n_y))

                # Calculate the offset based on tile position
                offset = (tile_x * n_x_stitched, tile_y * n_y_stitched)

                # Apply the offset to each corner
                edge_offset = [
                    (edge[0][0] + offset[0], edge[0][1] + offset[1]),
                    (edge[1][0] + offset[0], edge[1][1] + offset[1]),
                    (edge[2][0] + offset[0], edge[2][1] + offset[1]),
                    (edge[3][0] + offset[0], edge[3][1] + offset[1])]

                # Get the transformation matrix H for the current tile
                shift = df_shift[
                    (df_shift["cycle"] == cycle) &
                    (df_shift["tile_y"] == tile_y) &
                    (df_shift["tile_x"] == tile_x)]
                shift_cols = ["shift_" + str(i) for i in range(9)]
                H_mat = shift[shift_cols].values[0].reshape(3, 3)
                H_inv = np.linalg.inv(H_mat)

                if USE_GPU:
                    H_inv = cp.asarray(H_inv)

                # Apply the inverse transformation to the edge coordinates
                H = ProjectiveTransform(matrix=H_inv)

                if USE_GPU:
                    edge_offset = cp.array(edge_offset).T
                    edge_offset = cp.vstack((edge_offset, cp.ones(
                        edge_offset.shape[1])))  # make (x, y, 1)
                else:
                    edge_offset = np.array(edge_offset).T
                    edge_offset = np.vstack((edge_offset, np.ones(
                        edge_offset.shape[1])))  # make (x, y, 1)

                # Apply the transformation matrix
                edge_offset = H.params @ edge_offset  # apply H
                # remove the homogeneous coordinate
                edge_offset = edge_offset[:2, :].T

                # Calculate the bounding box with margin
                max_x = edge_offset[:, 0].max() + margin
                min_x = edge_offset[:, 0].min() - margin
                max_y = edge_offset[:, 1].max() + margin
                min_y = edge_offset[:, 1].min() - margin

                # Append the edge information
                edges.append(
                    [cycle, tile_y, tile_x, min_y, max_y, min_x, max_x])

    # Create a DataFrame to store all edges
    edges = pd.DataFrame(
        edges, columns=["cycle", "tile_y", "tile_x",
                        "min_y", "max_y", "min_x", "max_x"])
    return edges


def create_chunk_dataframe(shape, chunk_size):
    """
    Creates a DataFrame representing the coordinates of chunks within a grid
    based on the specified shape and chunk size.

    Args:
        shape (tuple of int): The dimensions of the grid (height, width).
        chunk_size (tuple of int): The size of each chunk (chunk_height, chunk_width).

    Returns:
        pandas.DataFrame: A DataFrame containing the chunk indices and coordinates, with columns:
                      'chunk_y', 'chunk_x', 'upper_y', 'lower_y', 'left_x', 'right_x'.
    """
    def _normalize_chunks(chunks, shape):
        """
        Calculates the number and size of chunks needed to cover the entire grid.

        Args:
            chunks (tuple of int): The size of each chunk (chunk_height, chunk_width).
            shape (tuple of int): The dimensions of the grid (height, width).

        Returns:
            list of list of int: A list where each element is a list containing the sizes of each chunk dimension.
        """
        num_chunks = [(shape[i] + chunks[i] - 1) // chunks[i]
                      for i in range(len(shape))]
        chunk_sizes = [
            [chunks[i]] * (num_chunks[i] - 1) + [
                shape[i] - chunks[i] * (num_chunks[i] - 1)]
            for i in range(len(shape))]

        return chunk_sizes

    def _get_chunk_coordinates(shape, chunk_size):
        """
        Generates the coordinates of each chunk within the grid.

        Args:
            shape (tuple of int): The dimensions of the grid (height, width).
            chunk_size (tuple of int): The size of each chunk (chunk_height, chunk_width).

        Yields:
            tuple: Chunk indices and coordinates (chunk_y, chunk_x, upper_y, lower_y, left_x, right_x).
        """
        chunk_dims = _normalize_chunks(chunk_size, shape)
        for y, _ in enumerate(chunk_dims[0]):
            for x, _ in enumerate(chunk_dims[1]):
                yield y, x, sum(chunk_dims[0][:y]), \
                    sum(chunk_dims[0][:y + 1]), sum(chunk_dims[1][:x]), \
                    sum(chunk_dims[1][:x + 1])

    data = list(_get_chunk_coordinates(shape, chunk_size))
    return pd.DataFrame(data, columns=[
        'chunk_y', 'chunk_x', 'upper_y', 'lower_y', 'left_x', 'right_x'])


def get_overlap(chunk_sel, tiles_df, cycle):
    """
    Finds the overlapping tiles within a specified chunk for a given cycle.

    Args:
        chunk_sel (dict): A dictionary containing the coordinates of the chunk
                          with keys: 'left_x', 'right_x', 'upper_y', 'lower_y'.
        tiles_df (pandas.DataFrame): A DataFrame containing tile information with columns:
                                 'cycle', 'min_x', 'max_x', 'min_y', 'max_y'.
        cycle (int): The cycle number for which to find overlapping tiles.

    Returns:
        pandas.DataFrame: A DataFrame of tiles that overlap with the specified chunk in the given cycle.
    """
    # Filter tiles for the specified cycle
    tiles_df_cycle = tiles_df[
        (tiles_df["cycle"] == cycle)]

    # Determine if any part of the tile's width overlaps with the chunk's width
    right_in = (chunk_sel["left_x"] <= tiles_df_cycle["max_x"]) & (
        tiles_df_cycle["max_x"] <= chunk_sel["right_x"])
    left_in = (chunk_sel["left_x"] <= tiles_df_cycle["min_x"]) & (
        tiles_df_cycle["min_x"] <= chunk_sel["right_x"])
    width_in = (tiles_df_cycle["min_x"] <= chunk_sel["left_x"]) & (
        chunk_sel["right_x"] <= tiles_df_cycle["max_x"])
    or_width_in = right_in | left_in | width_in

    # Determine if any part of the tile's height overlaps with the chunk's height
    upper_in = (chunk_sel["upper_y"] <= tiles_df_cycle["max_y"]) & (
        tiles_df_cycle["max_y"] <= chunk_sel["lower_y"])
    lower_in = (chunk_sel["upper_y"] <= tiles_df_cycle["min_y"]) & (
        tiles_df_cycle["min_y"] <= chunk_sel["lower_y"])
    height_in = (tiles_df_cycle["min_y"] <= chunk_sel["upper_y"]) & (
        chunk_sel["lower_y"] <= tiles_df_cycle["max_y"])
    or_height_in = upper_in | lower_in | height_in

    # Return tiles that overlap in both width and height
    return tiles_df_cycle[or_width_in & or_height_in]


def _register_chunk(input_img, zarr_path, group_name, df_chunk, df_tile,
                    df_H, n_y, n_x, chunk_size, block_info=None):
    """
    Registers and stitches image chunks based on transformation matrices and tile offsets.

    Args:
        input_img (numpy.ndarray): The input image data.
        zarr_path (str): Path to the Zarr file containing the image data.
        group_name (str): Name of the group in the Zarr file where data is stored.
        df_chunk (pandas.DataFrame): DataFrame containing chunk information (coordinates and dimensions).
        df_tile (pandas.DataFrame): DataFrame containing tile information for each cycle.
        df_H (pandas.DataFrame): DataFrame containing the transformation matrices for each tile.
        n_y (int): Height of each individual tile.
        n_x (int): Width of each individual tile.
        chunk_size (tuple of int): The size of each chunk (chunk_height, chunk_width).
        block_info (dict, optional): Information about the current block being processed in Dask.

    Returns:
        numpy.ndarray: The registered image chunk after stitching and applying transformations.
    """
    # Extract cycle and chunk coordinates from block_info
    cycle = block_info[0]["chunk-location"][0]
    chunk_y = block_info[0]["chunk-location"][1]
    chunk_x = block_info[0]["chunk-location"][2]

    # Load image data from Zarr
    dar_img = da.from_zarr(zarr_path, component=group_name + "/0/data")

    # Select the current chunk based on chunk coordinates
    chunk_sel = df_chunk[
        (df_chunk["chunk_y"] == chunk_y) &
        (df_chunk["chunk_x"] == chunk_x)].iloc[0]

    # Find overlapping tiles for the current chunk and merge with transformation matrices
    overlap = get_overlap(chunk_sel, df_tile, cycle)
    overlap = overlap.merge(df_H, on=["cycle", "tile_y", "tile_x"])

    # Calculate offsets for each overlapping tile
    overlap["offset_y"] = overlap["tile_y"] * n_y - chunk_sel["upper_y"]
    overlap["offset_x"] = overlap["tile_x"] * n_x - chunk_sel["left_x"]

    # Initialize an empty array for the registered tile image
    tile_img = cp.zeros(chunk_size) if USE_GPU else np.zeros(chunk_size)

    # Iterate through each overlapping tile and apply transformations
    for i, row in overlap.iterrows():
        shift_cols = ["shift_" + str(i) for i in range(9)]
        H_mat = row[shift_cols].values.reshape(3, 3).astype(np.float32)

        if USE_GPU:
            H_mat = cp.array(H_mat)
            offset = cp.array([-row["offset_x"], -row["offset_y"]])
            H_offset = cp.eye(3)
        else:
            offset = np.array([-row["offset_x"], -row["offset_y"]])
            H_offset = np.eye(3)

        # Apply the offset to the transformation matrix
        H_offset[:2, 2] = offset
        H_mat = H_mat @ H_offset
        H = AffineTransform(matrix=H_mat)

        # Extract the image data for the current tile
        tile_img_add = dar_img[cycle,
                               int(row["tile_y"]), int(row["tile_x"])]
        if USE_GPU:
            tile_img_add = cp.asarray(tile_img_add.compute())
        else:
            tile_img_add = tile_img_add.compute()

        # Apply the transformation and warp the image
        tile_img_add = warp(tile_img_add, H, output_shape=chunk_size,
                            preserve_range=True, order=0)

        # Take the maximum pixel values to merge the tile image
        tile_img = np.maximum(tile_img, tile_img_add)

    # Adjust the image dimensions based on chunk boundaries
    upper_y = chunk_sel["upper_y"]
    lower_y = chunk_sel["lower_y"]
    left_x = chunk_sel["left_x"]
    right_x = chunk_sel["right_x"]

    if lower_y - upper_y != chunk_size[0] or \
            right_x - left_x != chunk_size[1]:
        tile_img = tile_img[:lower_y - upper_y, :right_x - left_x]

    # Prepare the output image array
    out_img = np.zeros((1, chunk_size[0], chunk_size[1]))
    if USE_GPU:
        tile_img = tile_img.get()
    out_img[0, :tile_img.shape[0], :tile_img.shape[1]] = tile_img

    return out_img


def registration_cYXyx(zarr_path, group_tile, group_ref, chunk_size,
                       subfooter="", shift_footer="_shift_tile_cycle",
                       footer="_reg"):
    """
    Registers and stitches image tiles based on transformation matrices, creating a registered dataset in Zarr format.

    Args:
        zarr_path (str): Path to the Zarr file containing the image data.
        group_tile (str): Group name in the Zarr file for the tiles to be registered.
        group_ref (str): Group name in the Zarr file for the reference stitched images.
        chunk_size (tuple of int): Size of each chunk (chunk_height, chunk_width).
        subfooter (str, optional): String to append before the shift footer in the output filename; defaults to an empty string.
        shift_footer (str, optional): Footer of the shift CSV file; defaults to "_shift_tile_cycle".
        footer (str, optional): String appended to the output Zarr group name; defaults to "_reg".

    Returns:
        None: The function saves the registered and stitched images to a new Zarr group.
    """
    # Load transformation matrices from the shift file
    shift_path = zarr_path.replace(
        ".zarr", subfooter + shift_footer + ".csv")
    df_H = pd.read_csv(shift_path)

    # Load the image data from the Zarr file
    dar_img = da.from_zarr(zarr_path, component=group_tile + "/0/data")
    n_cycle, n_tile_y, n_tile_x, n_y, n_x = dar_img.shape

    # Load the reference stitched images from the Zarr file
    dar_stitched = da.from_zarr(zarr_path, component=group_ref + "/0/data")
    n_tile_stiched_y, n_tile_stiched_x, n_y_stitched, n_x_stitched = \
        dar_stitched.shape

    # Calculate tile edges based on transformations
    df_tile = get_edges(n_cycle, n_tile_y, n_tile_x, df_H,
                        n_y_stitched, n_x_stitched, n_y, n_x)

    # Create a DataFrame representing chunk coordinates
    shape = (n_y_stitched * n_tile_stiched_y,
             n_x_stitched * n_tile_stiched_x)
    df_chunk = create_chunk_dataframe(shape, chunk_size)

    # Determine the number of chunks in y and x directions
    n_chunk_y = df_chunk["chunk_y"].max() + 1
    n_chunk_x = df_chunk["chunk_x"].max() + 1
    chunk_w = n_chunk_x * chunk_size[1]
    chunk_h = n_chunk_y * chunk_size[0]

    # Create an empty array for storing the registered image chunks
    dar_chunk = da.zeros((n_cycle, chunk_h, chunk_w),
                         dtype=dar_img.dtype,
                         chunks=(1, chunk_size[0], chunk_size[1]))

    # Register the chunks by applying the transformations using _register_chunk function
    dar_res = da.map_blocks(
        _register_chunk, dar_chunk, zarr_path, group_tile, df_chunk, df_tile,
        df_H, n_y, n_x, chunk_size, dtype=dar_img.dtype,
        chunks=(1, chunk_size[0], chunk_size[1]))

    print("Registering: " + group_tile + show_resource())
    with ProgressBar():
        # Define the dimensions and coordinates for the output DataArray
        dims = ["cycle", "y", "x"]
        coords = {
            "cycle": range(n_cycle),
            "y": range(chunk_h),
            "x": range(chunk_w)}
        chunks = {"cycle": 1,
                  "y": chunk_size[0], "x": chunk_size[1]}

        # Create the output dataset, chunk it, and save it to Zarr
        out = xr.DataArray(dar_res, dims=dims, coords=coords)
        out = out.to_dataset(name="data")
        out = out.chunk(chunks=chunks)
        out.to_zarr(zarr_path, group=group_tile + footer + "/0", mode="w")


def registration_cYXyx_noref(zarr_path, group_tile, stitched_shape, chunk_size,
                             subfooter="", shift_footer="_shift_tile_cycle",
                             footer="_reg"):
    """
    Registers and stitches image tiles based on transformation matrices, creating a registered dataset in Zarr format.
    This version does not use a reference stitched image group, but instead takes the stitched shape directly as input.

    Differences from `registration_cYXyx`:
        - This function does not rely on a reference stitched image group (`group_ref`).
        - The stitched shape (`stitched_shape`) is provided directly as an argument, specifying the dimensions of the 
          stitched images (number of tiles in y and x directions and the dimensions of each tile).

    Args:
        zarr_path (str): Path to the Zarr file containing the image data.
        group_tile (str): Group name in the Zarr file for the tiles to be registered.
        stitched_shape (tuple of int): Shape of the stitched image (n_tile_stitched_y, n_tile_stitched_x, n_y_stitched, n_x_stitched).
        chunk_size (tuple of int): Size of each chunk (chunk_height, chunk_width).
        subfooter (str, optional): String to append before the shift footer in the output filename; defaults to an empty string.
        shift_footer (str, optional): Footer of the shift CSV file; defaults to "_shift_tile_cycle".
        footer (str, optional): String appended to the output Zarr group name; defaults to "_reg".

    Returns:
        None: The function saves the registered and stitched images to a new Zarr group.
    """
    # Load transformation matrices from the shift file
    print("Registering: " + group_tile + show_resource())
    shift_path = zarr_path.replace(
        ".zarr", subfooter + shift_footer + ".csv")
    df_H = pd.read_csv(shift_path)

    # Load the image data from the Zarr file
    dar_img = da.from_zarr(zarr_path, component=group_tile + "/0/data")
    n_cycle, n_tile_y, n_tile_x, n_y, n_x = dar_img.shape

    # Unpack the stitched shape provided as input
    n_tile_stiched_y, n_tile_stiched_x, n_y_stitched, n_x_stitched = \
        stitched_shape

    # Calculate tile edges based on transformations
    df_tile = get_edges(n_cycle, n_tile_y, n_tile_x, df_H,
                        n_y_stitched, n_x_stitched, n_y, n_x)

    # Create a DataFrame representing chunk coordinates
    shape = (n_y_stitched * n_tile_stiched_y,
             n_x_stitched * n_tile_stiched_x)
    df_chunk = create_chunk_dataframe(shape, chunk_size)

    # Determine the number of chunks in y and x directions
    n_chunk_y = df_chunk["chunk_y"].max() + 1
    n_chunk_x = df_chunk["chunk_x"].max() + 1
    chunk_w = n_chunk_x * chunk_size[1]
    chunk_h = n_chunk_y * chunk_size[0]

    # Create an empty array for storing the registered image chunks
    dar_chunk = da.zeros((n_cycle, chunk_h, chunk_w),
                         dtype=dar_img.dtype,
                         chunks=(1, chunk_size[0], chunk_size[1]))

    # Register the chunks by applying the transformations using _register_chunk function
    dar_res = da.map_blocks(
        _register_chunk, dar_chunk, zarr_path, group_tile, df_chunk, df_tile,
        df_H, n_y, n_x, chunk_size, dtype=dar_img.dtype,
        chunks=(1, chunk_size[0], chunk_size[1]))

    with ProgressBar():
        # Define the dimensions and coordinates for the output DataArray
        dims = ["cycle", "y", "x"]
        coords = {
            "cycle": range(n_cycle),
            "y": range(chunk_h),
            "x": range(chunk_w)}
        chunks = {"cycle": 1,
                  "y": chunk_size[0], "x": chunk_size[1]}

        # Create the output dataset, chunk it, and save it to Zarr
        out = xr.DataArray(dar_res, dims=dims, coords=coords)
        out = out.to_dataset(name="data")
        out = out.chunk(chunks=chunks)
        out.to_zarr(zarr_path, group=group_tile + footer + "/0", mode="w")
