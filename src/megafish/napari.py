import dask.config
import napari
from functools import partial
import pandas as pd
import zarr
import dask.array as da
dask.config.set(scheduler='threads')


def tile_2d(zarr_path, groups, pitch, colors, limits):
    """
    Visualizes 2D tiled image data from a Zarr dataset using napari.

    Args:
        zarr_path (str): Path to the Zarr file containing the image data.
        groups (list of str): List of Zarr group names to be visualized.
        pitch (tuple of float): Physical size of each pixel in micrometers (y, x).
        colors (list of str): List of colormap names for each image channel.
        limits (list of tuple): Contrast limits for each image channel in the format [(min, max), ...].

    Returns:
        None. The function opens a napari viewer displaying the images.
    """
    zr = zarr.open(zarr_path, mode='r')

    # Load each channel data from the specified groups
    channels = []
    for group in groups:
        channels.append(zr[group + "/0/data"])

    # Initialize napari viewer
    viewer = napari.Viewer()
    for channel, group, color, limits in zip(channels, groups, colors, limits):
        # Add each channel to the viewer with specified settings
        viewer.add_image(channel, name=group, colormap=color,
                         blending="additive", contrast_limits=limits,
                         scale=pitch)

    # Configure the viewer's scale bar and labels
    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = "um"
    viewer.scale_bar.font_size = 18
    viewer.dims.axis_labels = ["cycle", "tile_y", "tile_x", "", ""]
    napari.run(max_loop_level=2)


def registered(zarr_path, pitch, max_level, groups, colors, limits,
               genename_path=None):
    """
    Visualizes multiscale registered image data from a Zarr dataset using napari, with optional gene name display.

    Args:
        zarr_path (str): Path to the Zarr file containing the image data.
        pitch (tuple of float): Physical size of each pixel in micrometers (y, x).
        max_level (int): Maximum pyramid level to load for each group.
        groups (list of str): List of Zarr group names containing the image data to be visualized.
        colors (list of str): List of colormap names for each image channel.
        limits (list of tuple): Contrast limits for each image channel in the format [(min, max), ...].
        genename_path (str, optional): Path to a CSV file containing gene names for each cycle. If provided, gene names
                                       will be displayed based on the current cycle slider value.

    Returns:
        None. The function opens a napari viewer displaying the images with multiscale support and optional annotations.
    """
    # Load multiscale channels from the Zarr groups
    channels = []
    for group in groups:
        channels.append(
            [da.from_zarr(zarr_path, component=group + "/" +
                          str(i) + "/data") for i in range(max_level + 1)])

    # Load gene names from the CSV file if provided
    if genename_path is not None:
        df = pd.read_csv(genename_path)

        # Update text overlay based on the current cycle in the viewer
        def update_slider(event, df):
            cycle = viewer.dims.current_step[0]
            row = df.iloc[cycle].values.tolist()
            txt = ""
            for i, r in enumerate(row[1:]):
                txt += "ch" + str(i + 1) + ": " + str(r) + "\n"
            viewer.text_overlay.text = txt
            viewer.text_overlay.font_size = 18
            viewer.text_overlay.position = "top_left"
            viewer.text_overlay.color = 'white'

        update_slider_partial = partial(update_slider, df=df)

    # Initialize napari viewer
    viewer = napari.Viewer()
    for channel, group, color, limits in zip(channels, groups, colors, limits):
        # Add each channel with multiscale and specified settings
        viewer.add_image(channel, name=group, colormap=color,
                         blending="additive", contrast_limits=limits,
                         scale=pitch, multiscale=True)

    # Configure the scale bar and gene name overlay (if applicable)
    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = "um"
    viewer.scale_bar.font_size = 18
    if genename_path is not None:
        viewer.dims.events.current_step.connect(update_slider_partial)
        viewer.text_overlay.visible = True
    viewer.dims.axis_labels = ["cycle", "", ""]
    napari.run(max_loop_level=2)


def registered_is(zarr_path, pitch, max_level, groups, colors, limits,
                  genename_path=None):
    """
    Visualizes registered image data with image stitching from a Zarr dataset using napari, with optional gene name display.

    Args:
        zarr_path (str): Path to the Zarr file containing the image data.
        pitch (tuple of float): Physical size of each pixel in micrometers (y, x).
        max_level (int): Maximum pyramid level to load for each group.
        groups (list of str): List of Zarr group names containing the image data to be visualized.
        colors (list of str): List of colormap names for each image channel.
        limits (list of tuple): Contrast limits for each image channel in the format [(min, max), ...].
        genename_path (str, optional): Path to a CSV file containing gene names for each cycle. If provided, gene names
                                       will be displayed based on the current cycle slider value.

    Returns:
        None. The function opens a napari viewer displaying the images with multiscale support and optional annotations.

    Notes:
        - The function supports multiscale image data and displays different channels with specified settings.
        - If a gene name file is provided, gene names will update based on the current cycle index in the viewer.
    """
    # Load multiscale channels from the Zarr groups
    channels = []
    for group in groups:
        channels.append(
            [da.from_zarr(zarr_path, component=group + "/" +
                          str(i) + "/data") for i in range(max_level + 1)])

    # Load gene names from the CSV file if provided
    if genename_path is not None:
        df = pd.read_csv(genename_path)

        # Update text overlay based on the current cycle in the viewer
        def update_slider(event, df):
            cycle = viewer.dims.current_step[0]
            row = df.iloc[cycle * 2].values.tolist()
            txt = ""
            for i, r in enumerate(row[1:]):
                txt += "ch" + str(i + 1) + ": " + str(r) + "\n"
            viewer.text_overlay.text = txt
            viewer.text_overlay.font_size = 18
            viewer.text_overlay.position = "top_left"
            viewer.text_overlay.color = 'white'

        update_slider_partial = partial(update_slider, df=df)

    # Initialize napari viewer
    viewer = napari.Viewer()
    for channel, group, color, limits in zip(channels, groups, colors, limits):
        # Add each channel with multiscale and specified settings
        viewer.add_image(channel, name=group, colormap=color,
                         blending="additive", contrast_limits=limits,
                         scale=pitch, multiscale=True)

    # Configure the scale bar and gene name overlay (if applicable)
    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = "um"
    viewer.scale_bar.font_size = 18
    if genename_path is not None:
        viewer.dims.events.current_step.connect(update_slider_partial)
        viewer.text_overlay.visible = True
    viewer.dims.axis_labels = ["cycle", "", ""]
    napari.run(max_loop_level=2)


def segment(zarr_path, pitch, max_level, groups, colors, limits):
    """
    Visualizes segmented and labeled image data from a Zarr dataset using napari, with support for multiscale images.

    Args:
        zarr_path (str): Path to the Zarr file containing the image data.
        pitch (tuple of float): Physical size of each pixel in micrometers (y, x).
        max_level (int): Maximum pyramid level to load for each group.
        groups (list of str): List of Zarr group names containing the image data to be visualized.
        colors (list of str): List of colormap names for each image channel. Use "label" for segmented data.
        limits (list of tuple): Contrast limits for each image channel in the format [(min, max), ...].

    Returns:
        None. The function opens a napari viewer displaying the images with multiscale support.
    """
    # Load multiscale channels from the Zarr groups
    channels = []
    for group in groups:
        channels.append(
            [da.from_zarr(zarr_path, component=group + "/" +
                          str(i) + "/data") for i in range(max_level + 1)])

    # Initialize napari viewer
    viewer = napari.Viewer()
    for channel, group, color, limits in zip(channels, groups, colors, limits):
        if color == "label":
            # Add segmented data as labels
            viewer.add_labels(channel, name=group, scale=pitch,
                              multiscale=True, opacity=0.5)
        else:
            # Add regular image channels
            viewer.add_image(channel, name=group, colormap=color,
                             blending="additive", contrast_limits=limits,
                             scale=pitch, multiscale=True)
    # Configure the scale bar then run the viewer
    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = "um"
    viewer.scale_bar.font_size = 18
    napari.run(max_loop_level=2)
