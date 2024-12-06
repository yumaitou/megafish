import os
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
import zarr
import xarray as xr
from dask.diagnostics import ProgressBar
import dask.array as da

from .utils import natural_sort
from .config import USE_GPU, show_resource

if USE_GPU:
    import cupy as cp
    from cucim.skimage.morphology import binary_dilation
    from cucim.skimage.filters import gaussian
    from cupyx.scipy.signal import fftconvolve
    from cucim.skimage.measure import regionprops_table, label
    from cuml.neighbors import NearestNeighbors
else:
    from skimage.morphology import binary_dilation
    from skimage.filters import gaussian
    from scipy.signal import fftconvolve
    from skimage.measure import regionprops_table, label
    from sklearn.neighbors import NearestNeighbors


def gaussian_kernel(shape, sigma):
    """
    Generates a Gaussian kernel for spatial filtering.
    This kernel is used for spatial filtering in the MERFISH prefiltering step.

    Args:
        shape (tuple of int): The dimensions (height, width) of the kernel.
        sigma (float): The standard deviation of the Gaussian distribution, 
                       which controls the spread of the kernel.

    Returns:
        numpy.ndarray: A 2D array representing the Gaussian kernel, normalized 
        so that the sum of all elements equals 1.
    """

    m, n = [int((ss - 1.) / 2.) for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    kernel = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    kernel[kernel < np.finfo(kernel.dtype).eps * kernel.max()] = 0
    sumh = kernel.sum()
    if sumh != 0:
        kernel /= sumh
    return kernel


def merfish_prefilter(zarr_path, group, sigma_high, psf, iterations,
                      sigma_low, mask_size, footer="_mfp"):
    """
    Applies MERFISH prefiltering steps, including high-pass filtering, 
    Richardson-Lucy deconvolution, and low-pass filtering.

    Args:
        zarr_path (str): Path to the Zarr file containing the image data.
        group (str): Group name in the Zarr file where the image data is stored.
        sigma_high (float): Standard deviation for the high-pass Gaussian filter.
        psf (numpy.ndarray or cupy.ndarray): Point spread function used for deconvolution.
        iterations (int): Number of iterations for the Richardson-Lucy deconvolution.
        sigma_low (float): Standard deviation for the low-pass Gaussian filter.
        mask_size (int): Size of the dilation mask for edge removal.
        footer (str, optional): Footer string to append to the output Zarr group name; defaults to "_mfp".

    Returns:
        None: This function saves the processed image data in the Zarr file.
    """

    def _merfish_prefilter(img_raw, sigma_high, psf, iterations, sigma_low,
                           mask_size):
        """
        Internal function to apply high-pass filtering, Richardson-Lucy deconvolution,
        and low-pass filtering on a single image chunk.

        Args:
            img_raw (numpy.ndarray or cupy.ndarray): Raw input image.
            sigma_high (float): Standard deviation for high-pass filter.
            psf (numpy.ndarray or cupy.ndarray): Point spread function for deconvolution.
            iterations (int): Number of deconvolution iterations.
            sigma_low (float): Standard deviation for low-pass filter.
            mask_size (int): Size of the dilation mask for edge removal.

        Returns:
            numpy.ndarray or cupy.ndarray: Processed image after filtering steps.
        """
        if USE_GPU:
            # Convert image to GPU array if available
            img = cp.asarray(img_raw.astype(np.float32))
            frm = cp.asarray(img[0, :, :])

            # 1) High-pass filter
            lowpass = gaussian(frm, sigma=sigma_high, out=None,
                               cval=0, preserve_range=True,
                               truncate=4.0)
            lowpass = cp.clip(lowpass, 0, None)
            highpass = cp.clip(frm - lowpass, 0, None)

            # 2) Richardson-Lucy deconvolution
            im_deconv = 0.5 * cp.ones(frm.shape)
            psf_mirror = cp.asarray(psf[::-1, ::-1])

            eps = cp.finfo(frm.dtype).eps
            for _ in range(iterations):
                x = fftconvolve(im_deconv, cp.asarray(psf), 'same')
                cp.place(x, x == 0, eps)
                relative_blur = highpass / x + eps
                im_deconv *= fftconvolve(relative_blur, psf_mirror, 'same')

            # 3) Low-pass filter
            lowpass = gaussian(
                im_deconv, sigma=sigma_low, out=None,
                cval=0, preserve_range=True, truncate=4.0)
            lowpass = cp.clip(lowpass, 0, None)  # TODO

            # 4) Remove zero positions using dilation mask
            footprint = [(cp.ones((1, 3)), mask_size),
                         (cp.ones((3, 1)), mask_size)]
            mask = frm == 0
            mask = binary_dilation(mask, footprint=footprint)
            lowpass[mask] = 0

            img_raw[0, :, :] = lowpass.get()

        else:
            # Process image using CPU
            img = img_raw.astype(np.float32)
            frm = img[0, :, :]

            # 1) High-pass filter
            lowpass = gaussian(frm, sigma=sigma_high, output=None,
                               cval=0, preserve_range=True,
                               truncate=4.0)
            lowpass = np.clip(lowpass, 0, None)
            highpass = np.clip(frm - lowpass, 0, None)

            # 2) Richardson-Lucy deconvolution
            im_deconv = 0.5 * np.ones(frm.shape)
            psf_mirror = psf[::-1, ::-1]

            eps = np.finfo(frm.dtype).eps
            for _ in range(iterations):
                x = fftconvolve(im_deconv, psf, 'same')
                np.place(x, x == 0, eps)
                relative_blur = highpass / x + eps
                im_deconv *= fftconvolve(relative_blur, psf_mirror, 'same')

            # 3) Low-pass filter
            lowpass = gaussian(
                im_deconv, sigma=sigma_low, output=None,
                cval=0, preserve_range=True, truncate=4.0)
            lowpass = np.clip(lowpass, 0, None)  # TODO

            # 4) Remove zero positions using dilation mask
            footprint = [(np.ones((1, 3)), mask_size),
                         (np.ones((3, 1)), mask_size)]
            mask = frm == 0
            mask = binary_dilation(mask, footprint=footprint)
            lowpass[mask] = 0

            img_raw[0, :, :] = lowpass

        return img_raw

    # Open the Zarr dataset and extract the image data
    root = xr.open_zarr(zarr_path, group=group + "/0")
    xar = root["data"]

    template = xr.DataArray(
        da.empty_like(xar.data, dtype="float32"),
        dims=xar.dims, coords=xar.coords)

    print("MERFISH prefilter: " + group + show_resource())
    with ProgressBar():
        flt = xar.map_blocks(_merfish_prefilter, kwargs={
            "sigma_high": sigma_high, "psf": psf, "iterations": iterations,
            "sigma_low": sigma_low, "mask_size": mask_size}, template=template)

        # Chunk the result and save it to the Zarr file
        flt = flt.chunk(xar.chunks)
        ds = flt.to_dataset(name="data")
        ds.to_zarr(zarr_path, group=group + footer + "/0", mode="w")


def scaling(zarr_path, group, percentile, factor, footer="_scl"):
    """
    Scales the intensity of the image data stored in a Zarr file based on a given percentile and scaling factor.

    Args:
        zarr_path (str): Path to the Zarr file containing the image data.
        group (str): Group name in the Zarr file where the image data is stored.
        percentile (float): The percentile value used for scaling the image intensity.
        factor (float): The scaling factor applied to normalize the image intensity.
        footer (str, optional): Footer string to append to the output Zarr group name; defaults to "_scl".

    Returns:
        None: This function saves the scaled image data in the Zarr file.
    """
    root = xr.open_zarr(zarr_path, group=group + "/0")
    xar = root["data"]

    def _scaling(img, percentile, factor):
        """
        Internal function to scale image intensity based on the specified percentile and factor.

        Args:
            img (numpy.ndarray): The input image chunk.
            percentile (float): The percentile value for intensity scaling.
            factor (float): The scaling factor for normalization.

        Returns:
            numpy.ndarray: Scaled image data.
        """
        # Extract the first frame and scale based on the percentile and factor
        frm = img[0, :, :]
        frm = frm / np.percentile(frm, percentile) * factor
        img[0, :, :] = frm
        return img

    # Create a template for the output data array with the same shape as the input
    template = xr.DataArray(
        da.empty_like(xar.data, dtype="float32"),
        dims=xar.dims, coords=xar.coords)

    print("Scaling: " + group + show_resource())
    with ProgressBar():
        # Apply the scaling operation across all image blocks
        res = xar.map_blocks(_scaling, kwargs={
            "percentile": percentile, "factor": factor}, template=template)

        # Re-chunk the result and save it to the Zarr file
        res = res.chunk(xar.chunks)
        ds = res.to_dataset(name="data")
        ds.to_zarr(zarr_path, group=group + footer + "/0", mode="w")


def norm_value(zarr_path, group, footer="_nmv"):
    """
    Calculates the L2 norm (Euclidean norm) of the image data across cycles stored in a Zarr file.

    Args:
        zarr_path (str): Path to the Zarr file containing the image data.
        group (str): Group name in the Zarr file where the image data is stored.
        footer (str, optional): Footer string to append to the output Zarr group name; defaults to "_nmv".

    Returns:
        None: This function saves the calculated norm values as a new dataset in the Zarr file.
    """
    def _norm_value(img):
        """
        Internal function to calculate the L2 norm of the image data across the cycle dimension.

        Args:
            img (numpy.ndarray or xarray.DataArray): The input image chunk.

        Returns:
            xarray.DataArray: Norm values calculated for the image chunk.
        """
        norm_order = 2
        # Calculate the L2 norm along the cycle dimension
        norm = np.linalg.norm(img.values, ord=norm_order, axis=0)
        dims = ("y", "x")
        coords = {"y": img.coords["y"], "x": img.coords["x"], }
        return xr.DataArray(norm, dims=dims, coords=coords)

    # Open the Zarr dataset and extract the image data
    root = xr.open_zarr(zarr_path, group=group + "/0")
    xar = root["data"]
    n_cycle, n_y, n_x = xar.shape

    # Set up the chunk sizes for processing
    original_chunks = xar.chunks
    chunk_dict = {dim_name: chunk[0]
                  for dim_name, chunk in zip(xar.dims, original_chunks)}
    chunk_dict["cycle"] = n_cycle  # Set the entire cycle dimension as a chunk
    xar = xar.chunk(chunk_dict)

    # Set up the dimensions and coordinates for the output template
    new_dims = ("y", "x")
    new_coords = {
        "y": np.arange(n_y), "x": np.arange(n_x), }

    # Create a template DataArray for the output
    template = xr.DataArray(
        da.zeros((n_y, n_x),
                 chunks=(chunk_dict["y"], chunk_dict["x"]),
                 dtype=np.float32),
        dims=new_dims, coords=new_coords)

    print("Calculating norm: " + group + show_resource())
    with ProgressBar():
        # Apply the norm calculation across all image blocks
        res = xar.map_blocks(_norm_value, template=template)
        ds = res.to_dataset(name="data")
        # Save the result to the Zarr file
        ds.to_zarr(zarr_path, group=group + footer + "/0", mode="w")


def divide_by_norm(zarr_path, group_mfp, group_nmv, footer="_nrm"):
    """
    Divides the filtered image data by the calculated norm values for normalization.

    Args:
        zarr_path (str): Path to the Zarr file containing the image data.
        group_mfp (str): Group name in the Zarr file where the filtered image data is stored.
        group_nmv (str): Group name in the Zarr file where the norm values are stored.
        footer (str, optional): Footer string to append to the output Zarr group name; defaults to "_nrm".

    Returns:
        None: This function saves the normalized image data as a new dataset in the Zarr file.
    """

    print("Dividing by norm: " + group_mfp + show_resource())

    # Open the Zarr datasets and extract the filtered image and norm values
    ds = xr.open_zarr(zarr_path, group=group_mfp + "/0")
    xar_flt = ds["data"]
    ds = xr.open_zarr(zarr_path, group=group_nmv + "/0")
    xar_nmv = ds["data"]

    # Set up the chunk sizes for efficient processing
    original_chunks = xar_flt.chunks
    chunk_dict = {dim_name: chunk[0]
                  for dim_name, chunk in zip(xar_flt.dims, original_chunks)}

    # Divide the filtered image data by the norm values
    res = xar_flt / xar_nmv

    # Re-chunk the result to match the original chunks
    res = res.chunk(chunk_dict)

    # Save the result to the Zarr file as a new dataset
    ds = res.to_dataset(name="data")
    ds.to_zarr(zarr_path, group=group_mfp + footer + "/0", mode="w")


def nearest_neighbor(zarr_path, group, code_intensity_path, footer="_nnd"):
    """
    Calculates the nearest neighbor for each pixel's intensity trace in an image
    dataset using a precomputed codebook.

    Args:
        zarr_path (str): Path to the Zarr file containing the image data.
        group (str): Group name in the Zarr file where the image data is stored.
        code_intensity_path (str): Path to the NumPy file containing intensity codes for matching.
        footer (str, optional): Footer string to append to the output Zarr group name; defaults to "_nnd".

    Returns:
        None: This function saves the nearest neighbor indices and distances as a new dataset in the Zarr file.
    """
    def _nearest_neighbor(img, nn):
        """
        Internal function to calculate the nearest neighbor indices and distances
        for a given image chunk.

        Args:
            img (xarray.DataArray): Input image data chunk.
            nn (NearestNeighbors): Pre-trained nearest neighbors model.

        Returns:
            xarray.DataArray: Nearest neighbor indices and distances for the image chunk.
        """
        # Reshape and prepare the image data for nearest neighbor search
        pixel_traces = img.stack(features=("y", "x"))
        pixel_traces = pixel_traces.transpose("features", "cycle")
        pixel_traces = pixel_traces.values
        pixel_traces = pixel_traces.astype(np.float32)
        pixel_traces = np.nan_to_num(pixel_traces)

        # Perform nearest neighbor search
        metric_output, indices = nn.kneighbors(pixel_traces)

        # Reshape the results to match the image dimensions
        indices = indices.reshape(img.sizes["y"], img.sizes["x"])
        metric_output = metric_output.reshape(
            img.sizes["y"], img.sizes["x"])

        # Create a 3D array to store the indices and distances
        res = np.zeros((2, img.sizes["y"], img.sizes["x"]))
        res[0] = indices
        res[1] = metric_output

        dims = ("iddist", "y", "x")
        coords = {"iddist": np.arange(2),
                  "y": img.coords["y"], "x": img.coords["x"], }
        res = xr.DataArray(res, dims=dims, coords=coords)
        return res

    print("Calculating nearest neighbor: " + group + show_resource())

    # Open the Zarr dataset and extract the data
    root = xr.open_zarr(zarr_path, group=group + "/0")
    xar = root["data"]

    # Set up chunk sizes for efficient processing
    n_cycle, n_y, n_x = xar.shape
    original_chunks = xar.chunks
    chunk_dict = {dim_name: chunk[0]
                  for dim_name, chunk in zip(xar.dims, original_chunks)}
    chunk_dict["cycle"] = n_cycle

    xar = xar.chunk(chunk_dict)

    # Load the intensity codebook and train the nearest neighbors model
    linear_codes = np.load(code_intensity_path)
    nn = NearestNeighbors(n_neighbors=1, algorithm='auto',
                          metric="euclidean").fit(linear_codes)

    # Set up the dimensions and coordinates for the output
    new_dims = ("iddist", "y", "x")
    new_coords = {
        "iddist": np.arange(2),
        "y": np.arange(n_y),
        "x": np.arange(n_x), }

    template = xr.DataArray(
        da.zeros((2, n_y, n_x),
                 chunks=(2, chunk_dict["y"], chunk_dict["x"]),
                 dtype=np.float32),
        dims=new_dims, coords=new_coords)

    with ProgressBar():
        # Apply the nearest neighbor function to each chunk
        res = xar.map_blocks(
            _nearest_neighbor, kwargs={"nn": nn},
            template=template)

        # Save the result to the Zarr file as a new dataset
        ds = res.to_dataset(name="data")
        ds.to_zarr(zarr_path, group=group + footer + "/0", mode="w")


def split_nnd(zarr_path, group, footers=["_cde", "_dst"]):
    """
    Splits the nearest neighbor dataset into two separate datasets: one for 
    the code indices and one for the distances. The function saves each 
    as a new dataset in the Zarr file.

    Args:
        zarr_path (str): Path to the Zarr file containing the image data.
        group (str): Group name in the Zarr file where the nearest neighbor 
                     dataset is stored.
        footers (list, optional): List of two strings specifying the footers 
                                  for the output datasets; defaults to ["_cde", "_dst"].

    Returns:
        None: This function saves the split datasets in the Zarr file.
    """
    print("Splitting nnd: " + group + show_resource())

    # Open the Zarr dataset and extract the data
    root = xr.open_zarr(zarr_path, group=group + "/0")
    xar = root["data"]

    # Extract the code indices (iddist=0) and cast them to uint32
    xar_code = xar.sel(iddist=0).drop_vars("iddist").astype(np.uint32)

    # Extract the distances (iddist=1)
    xar_dist = xar.sel(iddist=1).drop_vars("iddist")

    # Save the code indices dataset
    ds = xar_code.to_dataset(name="data")
    ds.to_zarr(zarr_path, group=group + footers[0] + "/0", mode="w")

    # Save the distance dataset
    ds = xar_dist.to_dataset(name="data")
    ds.to_zarr(zarr_path, group=group + footers[1] + "/0", mode="w")


def select_decoded(zarr_path, group_nmv, group_nnd, min_intensity,
                   max_distance, area_limits, footer="_dec"):
    """
    Filters decoded spots based on intensity, distance, and area criteria.

    Args:
        zarr_path (str): Path to the Zarr file containing the image data.
        group_nmv (str): Group name in the Zarr file where the norm value data is stored.
        group_nnd (str): Group name in the Zarr file where the nearest neighbor data is stored.
        min_intensity (float): Minimum intensity threshold for valid spots.
        max_distance (float): Maximum distance threshold for valid spots.
        area_limits (tuple): A tuple containing the minimum and maximum area values (min_area, max_area).
        footer (str, optional): Footer string to append to the output Zarr group name; defaults to "_dec".

    Returns:
        None: This function saves the filtered dataset in the Zarr file.
    """
    def _select_decode(img, min_intensity, max_distance, area_limits):
        """
        Internal function to apply intensity, distance, and area-based filtering on a single image chunk.

        Args:
            img (numpy.ndarray or cupy.ndarray): Input image with decoded values, distances, and norm values.
            min_intensity (float): Minimum intensity threshold.
            max_distance (float): Maximum distance threshold.
            area_limits (tuple): Minimum and maximum allowed area values.

        Returns:
            numpy.ndarray or cupy.ndarray: Processed image after applying the filtering.
        """

        # Select the appropriate device (GPU or CPU)
        if USE_GPU:
            decoded_img = cp.asarray(img[0, :, :].values)
        else:
            decoded_img = img[0, :, :]

        dist_img = img[1, :, :]
        norm_img = img[2, :, :]

        # Label connected regions and filter based on intensity and distance
        label_img = label(decoded_img, connectivity=2)
        label_img[norm_img < min_intensity] = 0
        label_img[dist_img > max_distance] = 0

        # Calculate area properties of labeled regions
        props = regionprops_table(label_img, properties=("label", "area"))
        min_area, max_area = area_limits

        if USE_GPU:
            df = pd.DataFrame(
                {"label": props["label"].get(), "area": props["area"].get()})
            df = df[(df["area"] >= min_area) & (df["area"] <= max_area)]

            valid_labels = cp.asarray(df["label"].values)
            label_img[~cp.isin(label_img, valid_labels)] = 0
        else:
            df = pd.DataFrame(
                {"label": props["label"], "area": props["area"]})
            df = df[(df["area"] >= min_area) & (df["area"] <= max_area)]

            valid_labels = np.asarray(df["label"].values)
            label_img[~np.isin(label_img, valid_labels)] = 0

        # Mask regions not meeting the criteria
        label_img = label_img > 0  # TODO: gene_id 0 is removed here
        decoded_img = decoded_img * label_img

        res = np.zeros(img.shape[1:], dtype=np.uint32)
        res = decoded_img.get() if USE_GPU else decoded_img
        res = xr.DataArray(res, dims=("y", "x"),
                           coords={"y": img.coords["y"], "x": img.coords["x"]})
        return res

    print("Selecting decoded: " + group_nnd + show_resource())

    # Open the Zarr dataset and extract norm and nearest neighbor data
    root = xr.open_zarr(zarr_path, group=group_nmv + "/0")
    xar_nmv = root["data"]
    root = xr.open_zarr(zarr_path, group=group_nnd + "/0")
    xar_nnd = root["data"]

    original_chunks = xar_nnd.chunks
    chunk_dict = {dim_name: chunk[0]
                  for dim_name, chunk in zip(xar_nnd.dims, original_chunks)}
    n_y, n_x = xar_nmv.shape

    # Expand dimensions and concatenate norm and nearest neighbor data
    nmv_expanded = xar_nmv.expand_dims(dim={"iddist": [2]})
    xar_in = xr.concat([xar_nnd, nmv_expanded], dim="iddist")
    xar_in = xar_in.chunk({"iddist": 3,
                           "y": chunk_dict["y"], "x": chunk_dict["x"]})

    # Prepare the template for storing the result
    new_dims = ("y", "x")
    new_coords = {"y": np.arange(n_y), "x": np.arange(n_x), }
    template = xr.DataArray(
        da.empty((n_y, n_x),
                 chunks=(chunk_dict["y"], chunk_dict["x"]),
                 dtype=np.uint32),
        dims=new_dims, coords=new_coords)

    with ProgressBar():
        res = xar_in.map_blocks(
            _select_decode, kwargs={
                "min_intensity": min_intensity,
                "max_distance": max_distance,
                "area_limits": area_limits},
            template=template)

        ds = res.to_dataset(name="data")
        ds.to_zarr(zarr_path, group=group_nnd + footer + "/0", mode="w")


def coordinates_decoded(zarr_path, group_dec, group_nuc, footer="_crd"):
    """
    Extracts and records the coordinates of decoded spots within nuclei, saving the information
    in a CSV file for each chunk.

    Args:
        zarr_path (str): Path to the Zarr file containing the image data.
        group_dec (str): Group name in the Zarr file where decoded data is stored.
        group_nuc (str): Group name in the Zarr file where nuclear data is stored.
        footer (str, optional): Footer string to append to the output group name; defaults to "_crd".

    Returns:
        None: This function saves the coordinates of decoded spots in CSV files.
    """
    def _coordinates_decoded(img, csv_dir, block_info=None):
        """
        Internal function to extract spot coordinates and their associated properties
        for each chunk.

        Args:
            img (numpy.ndarray or cupy.ndarray): Input image containing nuclear and decoded data.
            csv_dir (str): Directory path to save the CSV files.
            block_info (dict, optional): Information about the current chunk.

        Returns:
            numpy.ndarray: An empty array as a placeholder.
        """
        chunk_y = block_info[0]["chunk-location"][1]
        chunk_x = block_info[0]["chunk-location"][2]

        nuc = img[0, :, :]
        dec = img[1, :, :]

        # Use GPU if available
        if USE_GPU:
            nuc = cp.asarray(nuc)
            dec = cp.asarray(dec)

        nuc_ids = np.unique(nuc)
        dfs = []
        for nuc_id in nuc_ids:
            if nuc_id == 0:
                continue

            # Isolate decoded spots within each nucleus
            dec_nuc = dec * (nuc == nuc_id)
            # Label the connected regions
            dec_label = label(dec_nuc, connectivity=2)
            props = regionprops_table(
                intensity_image=dec_nuc,
                label_image=dec_label,
                properties=["label", "centroid", "mean_intensity"])

            # Convert properties to DataFrame
            if USE_GPU:
                props_out = {key: value.get() for key, value in props.items()}
            else:
                props_out = props
            df = pd.DataFrame(props_out)
            df["nuc_id"] = nuc_id
            dfs.append(df)

        # Return empty array if no spots are found
        if len(dfs) == 0:
            return np.zeros((1, 1, 1), dtype=np.uint8)

        # Compile and save the DataFrame
        df = pd.concat(dfs)
        df["chunk_y"] = chunk_y
        df["chunk_x"] = chunk_x
        df = df[["chunk_y", "chunk_x", "nuc_id", "label",
                "centroid-0", "centroid-1", "mean_intensity"]]
        df.columns = ["chunk_y", "chunk_x", "nuc_id", "spot_id",
                      "centroid_y", "centroid_x", "barcode"]

        csv_path = os.path.join(
            csv_dir, str(chunk_y) + "_" + str(chunk_x) + ".csv")
        df.to_csv(csv_path, index=False)

        return np.zeros((1, 1, 1), dtype=np.uint8)

    def _merge_decoded_csv(zarr_path, group, column_names=[
            "nuc_id", "spot_id", "centroid_y", "centroid_x", "barcode"]):
        """
        Merges individual CSV files from each chunk into a single CSV file.

        Args:
            zarr_path (str): Path to the Zarr file.
            group (str): Group name in the Zarr file.
            column_names (list): Column names for the final merged DataFrame.

        Returns:
            None
        """
        csv_root = zarr_path.replace(".zarr", "_csv")
        csv_dir = os.path.join(csv_root, group, "0")
        csv_files = os.listdir(csv_dir)
        csv_files = natural_sort(csv_files)
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

        # Combine all data and save as a single CSV
        df = pd.concat(dfs, axis=0)
        df = df.reset_index(drop=True)
        df = df[['chunk_y', 'chunk_x'] + column_names]

        sample_name = os.path.splitext(os.path.basename(zarr_path))[0]
        save_name = sample_name + "_" + group + ".csv"
        save_path = os.path.join(csv_root, save_name)
        df.to_csv(save_path, index=False)

    # Load decoded and nuclear data from the Zarr file
    root = xr.open_zarr(zarr_path, group=group_dec + "/0")
    xar_dec = root["data"]

    root = xr.open_zarr(zarr_path, group=group_nuc + "/0")
    xar_nuc = root["data"]

    original_chunks = xar_dec.chunks
    chunk_dict = {dim_name: chunk[0]
                  for dim_name, chunk in zip(xar_dec.dims, original_chunks)}

    print("Getting coordinates decoded: " + group_dec + show_resource())

    # Expand dimensions and concatenate nuclear and decoded data
    xar_nuc_exp = xar_nuc.expand_dims(dim={"labeldec": [0]})
    xar_dec_exp = xar_dec.expand_dims(dim={"labeldec": [1]})
    xar = xr.concat([xar_nuc_exp, xar_dec_exp], dim="labeldec")
    xar = xar.chunk({
        "labeldec": 2, "y": chunk_dict["y"], "x": chunk_dict["x"]})

    # Save the combined data temporarily
    ds = xar.to_dataset(name="data")
    ds.to_zarr(zarr_path, group="_temp", mode="w")

    dar = da.from_zarr(zarr_path, component="_temp/data")

    # Create the directory for CSV output
    csv_root = zarr_path.replace(".zarr", "_csv")
    csv_dir = os.path.join(csv_root, group_dec + footer, "0")
    if os.path.exists(csv_dir):
        shutil.rmtree(csv_dir)
    os.makedirs(csv_dir)

    # Apply the function to each chunk and save the CSV files
    with ProgressBar():
        dar.map_blocks(_coordinates_decoded, csv_dir, dtype=np.uint8).compute()

    # Merge individual CSVs into a final output
    _merge_decoded_csv(zarr_path, group_dec + footer)

    # Remove the temporary Zarr store and CSV directory
    zarr_store = zarr.open(zarr_path, mode='a')
    zarr_store["_temp"].store.rmdir("_temp")
    shutil.rmtree(csv_dir)
