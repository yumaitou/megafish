import numpy as np
import xarray as xr
import dask.array as da
from skimage.morphology import disk
from dask.diagnostics import ProgressBar
from tqdm import tqdm
from .config import USE_GPU, show_resource

if USE_GPU:
    import cupy as cp
    from cucim.skimage.morphology import binary_dilation, binary_erosion
    from cupyx.scipy.ndimage import maximum_filter
else:
    from skimage.morphology import binary_dilation, binary_erosion
    from scipy.ndimage import maximum_filter


def make_pyramid(zarr_path, group, level=None):
    """
    Generates a pyramid of downsampled image data from a zarr dataset and writes it back to zarr storage.

    Args:
        zarr_path (str): The path to the zarr storage.
        group (str): The zarr group path containing the image data to be downsampled.
        level (int, optional): The specific downsampling level to create. If not specified,
                               all downsampling levels will be generated.

    Returns:
        None. The function writes the downsampled data directly into the zarr storage at each level.

    Notes:
        - The downsampling is performed by reducing the size of the yx-dimensions by half in each level.
        - Chunking and shape of the input data are considered to ensure proper processing and output.

    """

    print("Making pyramid: " + group + show_resource())
    dar_in = da.from_zarr(
        zarr_path, component=group + "/0/data")
    init_size = dar_in.shape
    index_dims = len(init_size) - 2

    # Determine the number of index dimensions (e.g., cycle, round) and chunk size
    if index_dims == 0:
        n_indices = []
    else:
        n_indices = list(init_size[:-2])
    chunk_size = dar_in.chunksize

    max_zoom = 0
    zoom_size = init_size
    # Calculate the maximum zoom level based on the initial image size and chunk size
    while zoom_size[-2] // 2 >= chunk_size[-2] \
            and zoom_size[-1] // 2 >= chunk_size[-1]:
        zoom_size_yx = [(zoom_size[-2] + 1) // 2, (zoom_size[-1] + 1) // 2]
        zoom_size = tuple(n_indices + zoom_size_yx)
        max_zoom += 1
    max_zoom += 1

    def func(block):
        new_block = np.zeros(chunk_size, dtype=block.dtype)
        clip_slices = [slice(None)] * len(n_indices) + \
            [slice(None, block.shape[-2]), slice(None, block.shape[-1])]
        new_block[tuple(clip_slices)] = block

        skip_slices = [slice(None)] * len(n_indices) + \
            [slice(None, None, 2), slice(None, None, 2)]
        new_block = new_block[tuple(skip_slices)]
        return new_block

    if level is not None:
        zooms = [level]
    else:
        zooms = range(1, max_zoom + 1)

    for i in tqdm(zooms, desc="Levels"):
        dar_in = da.from_zarr(
            zarr_path, component=group + "/" + str(i - 1) + "/data")

        chunk_slices = [1] * len(n_indices) + \
            [chunk_size[-2] // 2, chunk_size[-1] // 2]

        dar_dist = da.map_blocks(
            func, dar_in, dtype=dar_in.dtype,
            chunks=tuple(chunk_slices))

        # Define dimensions based on index count and y, x
        if index_dims == 0:
            dims = ["y", "x"]
        if index_dims == 1:
            dims = ['cycle', "y", "x"]

         # Set up coordinates for the DataArray
        coords = {}
        for k, n in enumerate(n_indices):
            coords[dims[k]] = np.arange(n)
        coords['y'] = np.arange(dar_dist.shape[-2])
        coords['x'] = np.arange(dar_dist.shape[-1])

        chunk_size_dict = {}
        for k, n in enumerate(n_indices):
            chunk_size_dict[dims[k]] = 1
        chunk_size_dict['y'] = chunk_size[-2]
        chunk_size_dict['x'] = chunk_size[-1]

        out = xr.DataArray(dar_dist, dims=dims, coords=coords)
        out = out.to_dataset(name='data')
        out = out.chunk(chunk_size_dict)

        out.to_zarr(zarr_path, mode='w', consolidated=True,
                    group=group + "/" + str(i))


def mask_edge(zarr_path, group, radius, dtype="uint8", footer="_edg"):
    """
    Creates an edge mask around binary regions in an image using dilation and erosion.

    Args:
        zarr_path (str): The path to the Zarr file containing the image data.
        group (str): The Zarr group name where the image data is located.
        radius (int): The radius of the dilation disk for creating the edge mask.
        dtype (str, optional): The data type for the output mask. Defaults to "uint8".
        footer (str, optional): The suffix for the output Zarr group name. Defaults to "_edg".

    Returns:
        None. This function saves the edge mask data directly to the Zarr file.
    """
    def _mask_edge(img, radius):
        # Convert the image to binary
        img = img > 0
        # Create a disk-shaped footprint for dilation based on the given radius
        footprint_dil = disk(radius, dtype=bool)[np.newaxis, :, :]
        if USE_GPU:
            img = cp.array(img)
            footprint_dil = cp.array(footprint_dil)
        # Apply binary dilation to create the spots
        spots = binary_dilation(img, footprint_dil)

        # Create a disk-shaped footprint for erosion with a radius of 1
        footprint_ero = disk(1, dtype=bool)[np.newaxis, :, :]
        if USE_GPU:
            footprint_ero = cp.array(footprint_ero)

        # Apply binary erosion to the dilated spots
        ero = binary_erosion(spots, footprint_ero).astype(np.uint8)
        # Subtract eroded spots from dilated spots to get the edge
        out = spots.astype(np.uint8) - ero
        if USE_GPU:
            out = cp.asnumpy(out)
        return out.astype(dtype)

    # Load the dataset and set up chunks and coordinates
    ds = xr.open_zarr(zarr_path, group=group + "/0")
    xar = ds["data"]

    original_chunks = xar.chunks
    chunk_dict = {dim_name: chunk[0]
                  for dim_name, chunk in zip(xar.dims,
                                             original_chunks)}
    dims = xar.dims
    coords = xar.coords
    dar = da.from_zarr(zarr_path, component=group + "/0/data")
    depth = (1, radius, radius)

    # Apply the edge masking function across the dataset with overlap
    res = dar.map_overlap(_mask_edge, depth=depth, radius=radius,
                          dtype=dtype)

    print("Making mask edge: " + group + show_resource())
    with ProgressBar():
        out = xr.DataArray(res, dims=dims, coords=coords)
        out = out.to_dataset(name="data")
        out = out.chunk(chunk_dict)
        out.to_zarr(zarr_path, group=group + footer + "/0", mode="w")


def max_filter(zarr_path, group_name, radius, axes, footer="_max"):

    if USE_GPU:
        def _max_filter(img, footprint):
            # Convert the image to a CuPy array for GPU processing
            img_cp = cp.asarray(img.copy())
            img_max = maximum_filter(img_cp, footprint=cp.asarray(footprint))
            # Convert the result back to an xarray.DataArray
            res_array = xr.DataArray(
                img_max.get(), dims=img.dims, coords=img.coords)
            return res_array

    else:
        def _max_filter(img, footprint):
            # Apply maximum filter to find local maxima
            img_max = maximum_filter(img, footprint=footprint)
            res_array = xr.DataArray(img_max, dims=img.dims, coords=img.coords)
            return res_array

    # Open the Zarr dataset and extract the data array
    root = xr.open_zarr(zarr_path, group=group_name + "/0")
    xar = root["data"]

    # Create a template for the output based on the input data
    template = xr.DataArray(
        da.empty_like(xar.data, dtype="float32"),
        dims=xar.dims, coords=xar.coords)

    # Expand the footprint based on the specified axes
    footprint = disk(radius)
    for _ in range(axes[0]):
        footprint = np.expand_dims(footprint, axis=0)

    print("Applying Maximum filter: " + group_name + show_resource())
    with ProgressBar():
        # Apply the local maxima detection to the image data
        lmx = xar.map_blocks(_max_filter, kwargs=dict(
            footprint=footprint), template=template)

        # Re-chunk the detected maxima using the original chunk sizes
        original_chunks = xar.chunks
        lmx = lmx.chunk(original_chunks)

        # Save the detected maxima as a new group in the Zarr file
        ds = lmx.to_dataset(name="data")
        ds.to_zarr(zarr_path, group=group_name + footer + "/0", mode="w")
