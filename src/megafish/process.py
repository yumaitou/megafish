import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar
import dask.array as da

from .config import USE_GPU, show_resource
if USE_GPU:
    import cupy as cp
    from cupyx.scipy.ndimage import gaussian_filter
else:
    from scipy.ndimage import gaussian_filter


def projection(zarr_path, group, dim="z", method="max",
               footer="_mip"):
    """
    Creates a projection of the image data along a specified dimension using the specified method.

    Args:
        zarr_path (str): Path to the Zarr file containing the image data.
        group (str): Group name in the Zarr file where the data is stored.
        dim (str, optional): The dimension along which to project (e.g., "z"); defaults to "z".
        method (str, optional): The projection method to use, either "max" for maximum projection
                                or "min" for minimum projection; defaults to "max".
        footer (str, optional): Footer string to append to the output Zarr group name; defaults to "_mip".

    Returns:
        None: The function saves the projection result to a new Zarr group.
    """
    # Open the Zarr dataset and extract the data array
    ds = xr.open_zarr(zarr_path, group=group + "/0")
    xar = ds["data"]

    # Retrieve the original chunk sizes and adjust for the projection dimension
    original_chunks = xar.chunks
    chunk_dict = {dim_name: chunk[0]
                  for dim_name, chunk in zip(xar.dims, original_chunks)}
    chunk_dict.pop(dim)

    print("Making " + method + " " + dim + " projection: "
          + group + show_resource())

    with ProgressBar():
        # Apply the projection method along the specified dimension
        if method == "max":
            res = xar.max(dim=dim)
        elif method == "min":
            res = xar.min(dim=dim)
        else:
            raise ValueError("Unsupported method")

        # Re-chunk the result and save it to the new Zarr group
        res = res.chunk(chunks=chunk_dict)
        res.to_zarr(zarr_path, group=group + footer + "/0", mode="w")


def masking(zarr_path, group_int, group_mask, invert_mask=False,
            footer="_msk"):
    """
    Applies a mask to the image data, setting masked regions to zero.

    Args:
        zarr_path (str): Path to the Zarr file containing the image data.
        group_int (str): Group name in the Zarr file for the intensity data to be masked.
        group_mask (str): Group name in the Zarr file for the mask data.
        invert_mask (bool, optional): If True, inverts the mask to apply the mask where the value is zero; defaults to False.
        footer (str, optional): Footer string to append to the output Zarr group name; defaults to "_msk".

    Returns:
        None: The function saves the masked result to a new Zarr group.
    """
    # Open the Zarr datasets for the intensity data and the mask
    ds_int = xr.open_zarr(zarr_path, group=group_int + "/0")
    xar_int = ds_int["data"]

    ds_msk = xr.open_zarr(zarr_path, group=group_mask + "/0")
    xar_msk = ds_msk["data"]

    # Retrieve the original chunk sizes for the intensity data
    original_chunks = xar_int.chunks
    chunk_dict = {dim_name: chunk[0]
                  for dim_name, chunk in zip(xar_int.dims, original_chunks)}

    # Expand the mask dimensions if the intensity data has 3 dimensions
    if len(xar_int.dims) == 3:
        xar_msk_broadcast = xar_msk.expand_dims(
            dim='cycle', axis=0).broadcast_like(xar_int)
    else:
        xar_msk_broadcast = xar_msk

    print("Masking: " + group_int + show_resource())
    with ProgressBar():
        # Apply the mask; invert if specified
        if invert_mask:
            res = xar_int.where(xar_msk_broadcast == 0, 0)
        else:
            res = xar_int.where(xar_msk_broadcast > 0, 0)

        # Re-chunk the result to match the original chunk sizes
        res = res.chunk(chunks=chunk_dict)
        # Save the masked data as a new Zarr group
        res.to_zarr(zarr_path, group=group_int + footer + "/0", mode="w")


def gaussian_blur(zarr_path, group, sigma, dtype="float32", footer="_gbr"):
    """
    Applies a Gaussian blur to the image data.

    Args:
        zarr_path (str): Path to the Zarr file containing the image data.
        group (str): Group name in the Zarr file where the data is stored.
        sigma (float or tuple of float): The standard deviation(s) for the Gaussian kernel.
        dtype (str, optional): The data type of the output image; defaults to "float32".
        footer (str, optional): Footer string to append to the output Zarr group name; defaults to "_gbr".

    Returns:
        None: The function saves the blurred result to a new Zarr group.
    """
    def _gaussian_filter(img, sigma):
        """
        Applies a Gaussian filter to the input image.

        Args:
            img (numpy.ndarray): The input image.
            sigma (float or tuple of float): The standard deviation(s) for the Gaussian kernel.

        Returns:
            numpy.ndarray: The blurred image.
        """
        img = img.astype(np.float32)

        # Handle 2D or 3D image shapes
        if len(img.shape) == 2:
            frm = img
        elif len(img.shape) == 3:
            frm = img.squeeze()
        else:
            raise ValueError("Unsupported shape")

        # Apply the Gaussian filter using GPU if available
        if USE_GPU:
            result = gaussian_filter(cp.asarray(frm), sigma).get()
        else:
            result = gaussian_filter(frm, sigma)

        # Reshape back if the input was 3D
        if len(img.shape) == 3:
            result = result[np.newaxis, :, :]

        return result

    # Open the Zarr dataset and extract the data array
    ds = xr.open_zarr(zarr_path, group=group + "/0")
    xar = ds["data"]

    # Retrieve the original chunk sizes and prepare chunking configuration
    original_chunks = xar.chunks
    chunk_dict = {dim_name: chunk[0]
                  for dim_name, chunk in zip(xar.dims, original_chunks)}
    dims = xar.dims
    coords = xar.coords
    dar = da.from_zarr(zarr_path, component=group + "/0/data")

    # Define the depth for the map_overlap operation based on image dimensions
    if len(dims) == 2:
        depth = (sigma, sigma)
    elif len(dims) == 3:
        depth = (1, sigma, sigma)
    else:
        raise ValueError("Unsupported shape")

    # Apply the Gaussian filter across chunks with overlap
    res = dar.map_overlap(_gaussian_filter, depth=depth,
                          sigma=sigma, dtype=dtype)

    print("Gaussian blur: " + group + show_resource())
    with ProgressBar():
        # Convert the result back to an xarray DataArray and set the chunks
        out = xr.DataArray(res, dims=dims, coords=coords)
        out = out.to_dataset(name="data")
        out = out.chunk(chunk_dict)
        # Save the blurred data as a new Zarr group
        out.to_zarr(zarr_path, group=group + footer + "/0", mode="w")


def binarization(zarr_path, group, threshold, reverse=False, dtype="uint8",
                 footer="_bin"):
    """
    Applies binarization to the image data based on a threshold.

    Args:
        zarr_path (str): Path to the Zarr file containing the image data.
        group (str): Group name in the Zarr file where the data is stored.
        threshold (float): The threshold value for binarization; values above the threshold are set to 1, others to 0.
        reverse (bool, optional): If True, reverses the binary values (0 becomes 1 and vice versa); defaults to False.
        dtype (str, optional): The data type of the output image; defaults to "uint8".
        footer (str, optional): Footer string to append to the output Zarr group name; defaults to "_bin".

    Returns:
        None: The function saves the binarized result to a new Zarr group.
    """
    # Open the Zarr dataset and extract the data array
    ds = xr.open_zarr(zarr_path, group=group + "/0")
    xar = ds["data"]

    # Retrieve the original chunk sizes for the data
    original_chunks = xar.chunks
    chunk_dict = {dim_name: chunk[0]
                  for dim_name, chunk in zip(xar.dims, original_chunks)}

    print("Binarization: " + group + show_resource())

    with ProgressBar():
        # Apply binarization based on the threshold
        res = (xar > threshold).astype(dtype)
        # Reverse binary values if specified
        if reverse:
            res = 1 - res
        # Re-chunk the result to match the original chunk sizes
        res = res.chunk(chunks=chunk_dict)
        # Save the binarized data as a new Zarr group
        res.to_zarr(zarr_path, group=group + footer + "/0", mode="w")
