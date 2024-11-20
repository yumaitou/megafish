================
User Guide
================

This section covers concepts, architecture, and detailed usage of **MEGA-FISH**.

In code blocks, we will assume that the common packages have been imported
as follows:

.. code-block:: python
    
    import megafish as mf


Data structure
================

The data structure used in MEGA-FISH is a Zarr file compatible with `Xarray <https://docs.xarray.dev/en/stable/>`_ . Each sample, representing different conditions or experimental replicates, consists of a ``.zarr`` folder and associated folders with suffixes like ``_tif`` and ``_csv``.
Within the ``.zarr`` folder, the structure is as follows:

1. **Processing group folders**: Contain data for each processing step or modality.
2. **Pyramid zoom subfolders**: Store progressively zoomed versions of the images for efficient visualization.

Below these levels, Xarray dimension folders organize the data, and chunked image data is stored within the ``data`` folder.

Users only need to specify the group names for processing; the pyramid zoom and dimension folders are automatically managed by the system.

**Folder Structure Example**:

.. code-block:: bash

    ├── sample1_csv
    ├── sample1_tif
    └── sample1.zarr
        ├── dapi (group)
        │   ├── 0 (pyramid zoom level)
        │   │   ├── data
        │   │   │   ├── 0.0.0
        │   │   │   ├── 0.0.1
        │   │   │   ├── 0.0.2
        │   │   │   └── ...
        │   │   ├── z (Xarray dimension)
        │   │   ├── x (Xarray dimension)
        │   │   └── y (Xarray dimension)

It is recommended to store images in groups based on modality or channel (e.g., DAPI, RNA channel 1). This organization facilitates streamlined processing and efficient data management.

Using workflow functions, users can generate new groups containing processed results. By default, a three-letter suffix indicating the processing type is appended to the group name. For example, processing results from the ``dapi`` group might be saved as follows:

.. code-block:: bash

    ├── sample1_csv
    ├── sample1_tif
    └── sample1.zarr
        ├── dapi (tiled image group)
        ├── dapi_mip (max intensity projection group)
        ├── dapi_mip_reg (registered mip image group)

Functions
====================

In MEGA-FISH workflows, image processing tasks such as registration, segmentation, spot detection are performed using workflow functions.
These functions follow a consistent structure:

.. code-block:: python

    mf.processing_module.workflow_function(zarr_path, target_group)

Here:

- **processing_module**: The module that contains the specific workflow function (e.g., ``mf.segment``, ``mf.register``).
- **workflow_function**: The function performing the desired processing task (e.g., ``watershed_label``, ``shift_cycle_cYXyx``).
- **zarr_path**: Path to the Zarr file for the sample to be processed.
- **target_group**: Name of the image group to be processed.

**Organization of Workflow Functions**

Executing this function generates a new group containing the processed results.
Users can re-run the function with modified parameters or resume the workflow from any step, eliminating the need to restart the entire process.
This flexibility is particularly useful for parameter tuning and iterative analyses.

Workflow functions are grouped into modules based on their functionality, such as:

- :py:mod:`megafish.register`: Registration-related functions.
- :py:mod:`megafish.segment`: Segmentation-related functions.
- :py:mod:`megafish.seqfish`: Spot detection and quantification for SeqFISH data.

For a comprehensive list of available functions and their detailed usage, refer to the `API reference <functions.html>`_.

Registration
====================

The registration functionality in MEGA-FISH is specialized for 2D image alignment, which is critical for accurate downstream analysis.

It is recommended to use nuclear staining images, such as DAPI, for registration. The registration process consists of the following steps:

1. **Cycle-wise alignment**:  
   
   Calculate the shifts between cycles, aligning each cycle relative to cycle 1. This step ensures that the same spatial regions are accurately aligned across different cycles.

2. **Tile-wise alignment**:  
   
   Calculate the shifts of each tile relative to the stitched reference image. This step corrects for any misalignments between tiles, producing a seamless full-cycle image.

The computed shifts are saved as transformation parameters in a CSV file.
This file can be manually edited if necessary to fine-tune the alignment.
Based on these parameters, MEGA-FISH applies the transformations to the tile images and generates chunked, fully registered images for each cycle.

For a detailed step-by-step workflow, refer to the :ref:`Registration <getting_started_registration>` section in the Getting Started guide.

Segmentation
====================

MEGA-FISH includes basic segmentation capabilities using binarization and watershed methods.
These methods are suitable for straightforward segmentation tasks, such as identifying well-separated nuclei.
For more complex datasets or densely packed cells, external segmentation tools may be required.

The segmentation snippet in the :ref:`Segmentation <getting_started_segmentation>` section of Getting Started demonstrates how to segment DAPI images after applying max intensity projection and registration.

**Integration with External Segmentation Tools**

MEGA-FISH provides input/output functions to facilitate the use of external segmentation tools, such as `Cellpose <https://github.com/mouseland/cellpose>`_ or `MEDIAR <https://github.com/Lee-Gihun/MEDIAR>`_:

- :py:func:`megafish.tif.save`: Exports images as chunked TIFF files for external processing.
- :py:func:`megafish.tif.load`: Imports segmented images chunk by chunk.
- :py:func:`megafish.segment.merge_split_label`: Merges labels split across chunk boundaries and ensures unique label identifiers across the entire image.

These functions enable seamless integration of external segmentation results into MEGA-FISH workflows, allowing users to benefit from advanced segmentation algorithms while maintaining compatibility with MEGA-FISH's data structure and processing pipeline.

Visualization
====================

MEGA-FISH integrates seamlessly with `Napari <https://napari.org/stable/>`_ for efficient visualization of large spatial omics datasets.

MEGA-FISH allows you to view full-cycle stitched images without worrying about PC memory limitations.
This is achieved through the use of pyramidal zoom images, which store multiple resolutions of the data.
By visualizing lower-resolution images when zoomed out, Napari minimizes memory usage and enhances performance.

To prepare your images for visualization, use the :py:func:`megafish.view.make_pyramid` function.

.. code-block:: python

    groups = ["hcst_mip_reg", "rna1_mip_reg", "rna2_mip_reg"]
    for group in groups:
        mf.view.make_pyramid(zarr_path, group)

This function generates pyramidal zoom images for a specified group.
Once prepared, you can load and visualize the images in Napari.

**Running in MEGA-FISH workflow**

The following script demonstrates how to visualize registered Hoechst and RNA channels 1 and 2 in :doc:`getting_started`:

.. code-block:: python

    mf.napari.registered(
        zarr_path, pitch=pitch[1:], max_level=2,
        groups=["hcst_mip_reg", "rna1_mip_reg", "rna2_mip_reg"],
        colors=["blue", "green", "magenta"],
        limits=[[100, 150], [100, 195], [100, 145]])

Here: 

- **zarr_path**: Path to the Zarr file containing your dataset.
- **pitch**: Spatial resolution in micrometers (e.g., ``[z, y, x]`` pitch values).
- **max_level**: Maximum pyramid level to visualize, controlling the zoom depth.
- **groups**: List of image groups to display.
- **colors**: Colors for each group in the visualization.
- **limits**: Intensity ranges for each channel.

See also the :py:func:`megafish.napari.registered` function documentation for more details.

**Launching Napari independently**

You can start Napari directly from the terminal in the Conda environment where MEGA-FISH is installed.
Use the following command:

.. code-block:: bash

    napari

Once Napari is open, you can paste the above script into the Napari console to load and visualize your images.