import os
import shutil
import numpy as np
import xarray as xr
import dask.array as da
from dask.diagnostics import ProgressBar
import tifffile
from skimage.transform import resize

from .utils import natural_sort


def save(zarr_path, group, zoom=0):
    """
    Saves image data from a Zarr file as individual TIFF files for each chunk.

    Args:
        zarr_path (str): Path to the Zarr file containing the image data.
        group (str): Group name in the Zarr file where the image data is stored.
        zoom (int, optional): Zoom level to select the data from; defaults to 0.

    Returns:
        None: This function saves the image chunks as TIFF files in the tif directory.
    """
    def _save_tif(img, tif_dir, block_info=None):
        """
        Saves a chunk of image data as a TIFF file.

        Args:
            img (numpy.ndarray): The image chunk to be saved.
            tif_dir (str): Directory where the TIFF file will be saved.
            block_info (dict, optional): Dask block information for chunk location.

        Returns:
            numpy.ndarray: A dummy array to match the expected return type.
        """
        # Get chunk position information from block_info and format the file name
        chunk_pos = block_info[0]["chunk-location"]
        cunnk_name = [str(pos) for pos in chunk_pos]
        cunnk_name = "_".join(cunnk_name)
        tif_path = os.path.join(tif_dir, cunnk_name + ".tif")

        # Save the image chunk as a TIFF file
        tifffile.imwrite(tif_path, img)

        # Return a dummy array as a placeholder
        dummy_shape = tuple([1] * len(img.shape))
        return np.zeros(dummy_shape, dtype=np.uint8)

    print("Saving tif images: " + group + " zoom" + str(zoom))

    # Load the image data from Zarr as a Dask array
    dar = da.from_zarr(zarr_path, component=group + "/" + str(zoom) + "/data")

    # Set up the directory for saving TIFF files
    tif_dir = zarr_path.replace(".zarr", "_tif")
    tif_dir = os.path.join(tif_dir, group, str(zoom))
    if os.path.exists(tif_dir):
        shutil.rmtree(tif_dir)
    os.makedirs(tif_dir)

    # Apply the function to save each chunk as a TIFF file
    dar.map_blocks(_save_tif, tif_dir, dtype=np.uint8).compute()


def save_tile_montage(zarr_path, group, tile_size, footer="_mtg"):
    """
    Creates a tiled montage of images from a Zarr file and saves it as a single TIFF file.

    Args:
        zarr_path (str): Path to the Zarr file containing the image data.
        group (str): Group name in the Zarr file where the image data is stored.
        tile_size (tuple of int): Size of each tile in the montage (height, width).
        footer (str, optional): Footer string to append to the output TIFF file name; defaults to "_mtg".

    Returns:
        None: This function saves the montage as a single TIFF file in the tif directory.
    """
    print("Saving monage tif images: " + group)

    # Load the image data from Zarr as a Dask array
    dar = da.from_zarr(zarr_path, component=group + "/0/data")
    n_cycle, n_tile_y, n_tile_x, n_y, n_x = dar.shape

    # Set up a temporary directory for saving intermediate TIFF files
    temp_root = zarr_path.replace(".zarr", "_temp")
    temp_dir = os.path.join(temp_root, group)
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    def _save_temp_tif(img, tif_dir, tile_size, block_info=None):
        """
        Saves a resized tile as a temporary TIFF file.

        Args:
            img (numpy.ndarray): The image chunk to be saved.
            tif_dir (str): Directory where the temporary TIFF file will be saved.
            tile_size (tuple): Desired size of each tile (height, width).
            block_info (dict, optional): Dask block information for chunk location.

        Returns:
            numpy.ndarray: A dummy array to match the expected return type.
        """
        # Get chunk position information and format the file name
        chunk_pos = block_info[0]["chunk-location"]
        cunnk_name = [str(pos) for pos in chunk_pos]
        cunnk_name = "_".join(cunnk_name)
        tif_path = os.path.join(tif_dir, cunnk_name + ".tif")

        # Resize the image chunk to the specified tile size
        output_shape = tuple(list(img.shape[:-2]) + tile_size)
        img_reshaped = resize(img, output_shape, preserve_range=True)

        # Save the reshaped image as a TIFF file
        tifffile.imwrite(tif_path, img_reshaped.astype(np.uint16))

        # Return a dummy array as a placeholder
        dummy_shape = tuple([1] * len(img.shape))
        return np.zeros(dummy_shape, dtype=np.uint8)

    # Save each tile as a temporary TIFF file with progress tracking
    with ProgressBar():
        dar.map_blocks(_save_temp_tif, temp_dir, tile_size,
                       dtype=np.uint8).compute()

    # Create an empty array for the montage image
    img = np.zeros((n_cycle, n_tile_y * tile_size[0], n_tile_x * tile_size[1]),
                   dtype=dar.dtype)

    # Read and arrange each temporary TIFF file into the montage array
    file_names = os.listdir(temp_dir)
    file_names = natural_sort(file_names)

    for file_name in file_names:
        c, y, x = [int(pos) for pos in file_name.split("_")[:3]]
        img[c, y * tile_size[0]:(y + 1) * tile_size[0],
            x * tile_size[1]:(x + 1) * tile_size[1]] = tifffile.imread(
                os.path.join(temp_dir, file_name))

    # Set up the directory for saving the final montage TIFF file
    tif_dir = zarr_path.replace(".zarr", "_tif")
    if not os.path.exists(tif_dir):
        os.makedirs(tif_dir)

    # Define the name and path for the final montage TIFF file
    sample_name = os.path.splitext(os.path.basename(zarr_path))[0]
    tif_name = sample_name + "_" + group + footer + ".tif"
    tif_path = os.path.join(tif_dir, tif_name)

    # Save the montage image as a single TIFF file
    tifffile.imwrite(tif_path, img, imagej=True, metadata={'axes': 'TYX'})

    # Remove the temporary directory and its contents
    shutil.rmtree(temp_root)


def save_whole_image(zarr_path, group, zoom=0, clip=False):
    """
    Saves the entire image from a Zarr file as a TIFF file, with an option to clip the image.

    Args:
        zarr_path (str): Path to the Zarr file containing the image data.
        group (str): Group name in the Zarr file where the image data is stored.
        zoom (int, optional): Zoom level to select the data from; defaults to 0.
        clip (tuple or bool, optional): If a tuple (height, width) is provided, the image will be clipped 
                                        to this size. If False, the entire image is saved; defaults to False.

    Returns:
        None: This function saves the image as a single TIFF file.
    """
    print("Saving whole tif images: " + group + " zoom=" + str(zoom))

    # Load the image data from Zarr as a Dask array
    dar = da.from_zarr(zarr_path, component=group + "/" + str(zoom) + "/data")

    # Set up the directory for saving the TIFF file
    tif_dir = zarr_path.replace(".zarr", "_tif")
    if not os.path.exists(tif_dir):
        os.makedirs(tif_dir)

    # Define the name and path for the TIFF file
    sample_name = os.path.splitext(os.path.basename(zarr_path))[0]
    tif_name = sample_name + "_" + group + "_zoom" + str(zoom) + ".tif"

    # Compute the image from the Dask array
    img = dar.compute()
    if img.dtype == np.uint32:
        img = img.astype(np.float32)

    # Clip the image if clip dimensions are provided
    if clip:
        if len(img.shape) == 3:
            img = img[:, :clip[0], :clip[1]]
        elif len(img.shape) == 2:
            img = img[:clip[0], :clip[1]]

    # Save the image as a TIFF file with appropriate metadata
    tif_path = os.path.join(tif_dir, tif_name)
    if len(img.shape) == 3:
        tifffile.imwrite(tif_path, img, imagej=True, metadata={'axes': 'TYX'})
    elif len(img.shape) == 2:
        tifffile.imwrite(tif_path, img, imagej=True, metadata={'axes': 'YX'})


def save_chunk(zarr_path, group, chunk, footer):
    """
    Saves specific chunks of image data from a Zarr file as individual TIFF files.

    Args:
        zarr_path (str): Path to the Zarr file containing the image data.
        group (str): Group name in the Zarr file where the image data is stored.
        chunk (list of list of int): List of chunk coordinates to be saved as TIFF files.
        footer (str): Footer to append to the output directory name.

    Returns:
        None: This function saves the specified chunks as individual TIFF files.
    """
    def _save_tif_chunk(img, tif_dir, chunk_list, block_info=None):
        """
        Saves a specific chunk of image data as a TIFF file if it matches the target chunk coordinates.

        Args:
            img (numpy.ndarray): The image chunk to be saved.
            tif_dir (str): Directory where the TIFF file will be saved.
            chunk_list (list of list of int): List of target chunk coordinates.
            block_info (dict, optional): Dask block information for chunk location.

        Returns:
            numpy.ndarray: A dummy array to match the expected return type.
        """
        # Get the position of the current chunk from block_info
        chunk_pos = block_info[0]["chunk-location"]

        # Check if the current chunk matches any in the target chunk list
        for chunk_target in chunk_list:
            if np.all(np.array(chunk_pos) == np.array(chunk_target)):
                # Format the file name based on chunk position and save the TIFF file
                tif_name = [str(pos) for pos in chunk_pos]
                tif_name = "_".join(tif_name) + ".tif"
                tif_path = os.path.join(tif_dir, tif_name)
                tifffile.imwrite(tif_path, img)

        # Return a dummy array as a placeholder
        dummy_shape = tuple([1] * len(img.shape))
        return np.zeros(dummy_shape, dtype=img.dtype)

    # Load the image data from Zarr as a Dask array
    dar = da.from_zarr(zarr_path, component=group + "/0/data")

    # Set up the directory for saving the TIFF files
    tif_dir = zarr_path.replace(".zarr", "_tif")
    tif_dir = os.path.join(tif_dir, group, footer)
    if os.path.exists(tif_dir):
        shutil.rmtree(tif_dir)
    os.makedirs(tif_dir)

    print("Saving tif chunk images: " + group)
    with ProgressBar():
        # Apply the function to save each matching chunk as a TIFF file
        dar.map_blocks(_save_tif_chunk, tif_dir, chunk,
                       dtype=dar.dtype).compute()


def load(zarr_path, group_load, group_template, footer_ext, dtype=None):
    """
    Loads TIFF images into a Zarr dataset using the structure of a template group.

    Args:
        zarr_path (str): Path to the Zarr file where the images will be stored.
        group_load (str): Group name in the Zarr file where the loaded images will be saved.
        group_template (str): Template group name in the Zarr file to match the structure of the data.
        footer_ext (str): Extension string to append to the file names of the TIFF images.
        dtype (type, optional): Data type for the loaded images; if None, uses the data type of the template group.

    Returns:
        None: This function saves the loaded images as a Zarr dataset.
    """
    def _load_tif(img, tif_dir, footer_ext, dtype, block_info=None):
        """
        Loads a TIFF file corresponding to the current chunk location.

        Args:
            img (numpy.ndarray): Placeholder array with the same shape as the chunk.
            tif_dir (str): Directory where the TIFF files are located.
            footer_ext (str): Extension to match the file name pattern.
            dtype (type): Data type for the output array.
            block_info (dict, optional): Dask block information for chunk location.

        Returns:
            numpy.ndarray: The loaded image or a zero array if the file does not exist.
        """
        # Determine chunk position from block_info
        chunk_y = block_info[0]["chunk-location"][0]
        chunk_x = block_info[0]["chunk-location"][1]

        # Construct the TIFF file name based on chunk position
        tif_name = str(chunk_y) + "_" + str(chunk_x) + footer_ext
        tif_path = os.path.join(tif_dir, tif_name)

        # Initialize an output array with zeros
        output = np.zeros(img.shape, dtype=dtype)
        # If the TIFF file exists, load it
        if os.path.exists(tif_path):
            output = tifffile.imread(tif_path)

        return output

    # Set up the directory where the TIFF files are located
    tif_dir = zarr_path.replace(".zarr", "_tif")
    tif_dir = os.path.join(tif_dir, group_load, "0")

    # Load the template structure from the Zarr file as a Dask array
    dar = da.from_zarr(zarr_path, component=group_template + "/0/data")

    n_y, n_x = dar.shape
    original_chunks = dar.chunks
    chunk_y, chunk_x = original_chunks

    # If dtype is not provided, use the dtype of the template
    if dtype is None:
        dtype = dar.dtype

    # Apply the _load_tif function to each block
    res = da.map_blocks(_load_tif, dar, tif_dir, footer_ext, dtype,
                        dtype=dtype)
    print("Loading tif images: " + group_load)

    with ProgressBar():
        # Define dimensions, coordinates, and chunking for the xarray DataArray
        dims = ["y", "x"]
        coords = {"y": range(dar.shape[0]),
                  "x": range(dar.shape[1])}
        chunks = {"y": chunk_y[0], "x": chunk_x[0]}

        # Create the xarray DataArray and save it as a Zarr dataset
        out = xr.DataArray(res, dims=dims, coords=coords)
        out = out.to_dataset(name="data")
        out = out.chunk(chunks=chunks)
        out.to_zarr(zarr_path, mode="w", group=group_load + "/0")


def save_rgb(zarr_path, group_r, group_g, group_b, group_out):
    """
    Combines individual red, green, and blue image groups from a Zarr file into RGB images and saves them as TIFF files.

    Args:
        zarr_path (str): Path to the Zarr file containing the image data.
        group_r (str or None): Group name for the red channel; set to None if not available.
        group_g (str or None): Group name for the green channel; set to None if not available.
        group_b (str or None): Group name for the blue channel; set to None if not available.
        group_out (str): Group name for saving the combined RGB images.

    Returns:
        None: This function saves the combined RGB images as TIFF files.
    """
    def _save_rgb(dar_r, dar_g, dar_b, tif_dir, block_info=None):
        """
        Combines and normalizes the R, G, and B channels into an RGB image and saves it as a TIFF file.

        Args:
            dar_r (numpy.ndarray or None): The red channel image chunk.
            dar_g (numpy.ndarray or None): The green channel image chunk.
            dar_b (numpy.ndarray or None): The blue channel image chunk.
            tif_dir (str): Directory where the TIFF file will be saved.
            block_info (dict, optional): Dask block information for chunk location.

        Returns:
            numpy.ndarray: A dummy array to match the expected return type.
        """
        # Determine the chunk position based on block_info
        for i, dar in enumerate([dar_r, dar_g, dar_b]):
            if dar is not None:
                block_no = i
        chunk_pos = block_info[block_no]["chunk-location"]
        cunnk_name = [str(pos) for pos in chunk_pos]
        cunnk_name = "_".join(cunnk_name)

        # Normalize and scale each channel to 0-255 if it exists
        if dar_r is not None:
            img_r = dar_r.astype(np.float32)
            r_min = img_r[img_r > 0]
            if len(r_min) > 0:
                r_min = r_min.min()
                img_r_norm = (img_r - np.array(r_min)) / \
                    (img_r.max() - np.array(r_min)) * 255
                img_r_norm[img_r_norm < 0] = 0
                img_r_norm = img_r_norm.astype(np.uint8)
            img_shape = img_r.shape
        if dar_g is not None:
            img_g = dar_g.astype(np.float32)
            g_min = img_g[img_g > 0]
            if len(g_min) > 0:
                g_min = g_min.min()
                img_g_norm = (img_g - np.array(g_min)) / \
                    (img_g.max() - np.array(g_min)) * 255
                img_g_norm[img_g_norm < 0] = 0
                img_g_norm = img_g_norm.astype(np.uint8)
            img_shape = img_g.shape
        if dar_b is not None:
            img_b = dar_b.astype(np.float32)
            b_min = img_b[img_b > 0]
            if len(b_min) > 0:
                b_min = b_min.min()
                img_b_norm = (img_b - np.array(b_min)) / \
                    (img_b.max() - np.array(b_min)) * 255
                img_b_norm[img_b_norm < 0] = 0
                img_b_norm = img_b_norm.astype(np.uint8)
            img_shape = img_b.shape

        # Handle cases where any channel is missing
        if dar_r is None:
            img_r_norm = np.zeros(img_shape, dtype=np.uint8)
        if dar_g is None:
            img_g_norm = np.zeros(img_shape, dtype=np.uint8)
        if dar_b is None:
            img_b_norm = np.zeros(img_shape, dtype=np.uint8)

        # Combine the normalized channels into an RGB image
        rgb_img = np.zeros((img_shape[0], img_shape[1], 3), dtype=np.uint8)
        rgb_img[:, :, 0] = img_r_norm
        rgb_img[:, :, 1] = img_g_norm
        rgb_img[:, :, 2] = img_b_norm

        # Save the RGB image as a TIFF file
        tif_path = os.path.join(tif_dir, cunnk_name + ".tif")
        tifffile.imwrite(tif_path, rgb_img)

        # Return a dummy array as a placeholder
        dummy_shape = tuple([1] * len(img_shape))
        return np.zeros(dummy_shape, dtype=np.uint8)

    # Set up the directory for saving the TIFF files
    tif_dir = zarr_path.replace(".zarr", "_tif")
    tif_dir = os.path.join(tif_dir, group_out, "0")
    if os.path.exists(tif_dir):
        shutil.rmtree(tif_dir)
    os.makedirs(tif_dir)

    # Load the red, green, and blue channels as Dask arrays if they exist
    if group_r is not None:
        dar_r = da.from_zarr(zarr_path, component=group_r + "/0/data")
    else:
        dar_r = None

    if group_b is not None:
        dar_b = da.from_zarr(zarr_path, component=group_b + "/0/data")
    else:
        dar_b = None

    if group_g is not None:
        dar_g = da.from_zarr(zarr_path, component=group_g + "/0/data")
    else:
        dar_g = None

    # Apply the function to combine and save the RGB images
    da.map_blocks(_save_rgb, dar_r, dar_g, dar_b,
                  tif_dir, dtype=np.uint8).compute()
