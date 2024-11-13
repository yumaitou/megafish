===============
Functions
===============

The list of functions. Full API documentation is available at :doc:`api/megafish`.

load
===============

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Workflow Functions
     - Description
   * - :py:mod:`megafish.load.ims_cYXzyx`
     - Creates Zarr arrays for image data in cycle, tile, and spatial (z, y, x) dimensions, then loads .ims images into these arrays using metadata from an image path CSV.
   * - :py:mod:`megafish.load.tif_cYXzyx`
     - Creates Zarr arrays for image data in cycle, tile, and spatial (z, y, x) dimensions, then loads TIFF images into these arrays using metadata from an image path CSV.
   * - :py:mod:`megafish.load.stitched_ims`
     - Reads a single stitched .ims image and divides it into smaller tile_y, tile_x segments for loading. This segmented image is primarily used for registration purposes.
   * - :py:mod:`megafish.load.stitched_tif`
     - 	Reads a single stitched TIFF image and divides it into smaller tile_y, tile_x segments for loading.

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Helper Functions
     - Description
   * - :py:mod:`megafish.load.make_dirlist`
     - Generates a CSV file listing all cycle directories within a specified image directory. Each cycle directory must contain tiled images organized by color, z, y, and x.
   * - :py:mod:`megafish.load.make_imagepath_cYX_from_dirlist`
     - Generates a CSV file mapping image paths to cycle, tile, and channel information based on a directory list CSV file.
   * - :py:mod:`megafish.load.make_imagepath_cYX`
     - Generates a CSV file mapping image paths to cycle, tile, and channel information using CSV files containing image paths and cycle directories.
     
register
===============

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Workflow Functions
     - Description
   * - :py:mod:`megafish.register.shift_cycle_cYXyx`
     - Calculates and stores cycle shifts for aligning image tiles based on phase correlation and feature matching.
   * - :py:mod:`megafish.register.shift_tile_cYXyx`
     - Calculates and stores tile shifts for aligning image tiles based on phase correlation and feature matching.
   * - :py:mod:`megafish.register.dummy_shift_tile`
     - Creates a dummy tile shifts CSV file with identity transformation values.
   * - :py:mod:`megafish.register.merge_shift_cYXyx`
     - Merges cycle and tile shift transformations and saves the combined shifts as a CSV file.
   * - :py:mod:`megafish.register.registration_cYXyx`
     - Registers and stitches image tiles based on transformation matrices, creating a registered dataset in Zarr format.
   * - :py:mod:`megafish.register.registration_cYXyx_noref`
     - Registers and stitches image tiles based on transformation matrices, creating a registered dataset in Zarr format. This version does not use a reference stitched image group, but instead takes the stitched shape directly as input.

process
===============

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Workflow Functions
     - Description
   * - :py:mod:`megafish.process.projection`
     - Creates a projection of the image data along a specified dimension using the specified method.
   * - :py:mod:`megafish.process.masking`
     - Applies a mask to the image data, setting masked regions to zero.
   * - :py:mod:`megafish.process.gaussian_blur`
     - Applies a Gaussian blur to the image data.
   * - :py:mod:`megafish.process.binarization`
     - Applies binarization to the image data based on a threshold.

segment
===============

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Workflow Functions
     - Description
   * - :py:mod:`megafish.segment.dilation`
     - Applies a binary dilation operation to the image data.
   * - :py:mod:`megafish.segment.merge_split_label`
     - Merges split labels in an image dataset by applying offsets and resolving conflicts.
   * - :py:mod:`megafish.segment.grow_voronoi`
     - Expands labeled regions in an image using a Voronoi-like approach, filling the image based on the nearest labeled pixel within a specified maximum distance.
   * - :py:mod:`megafish.segment.masking`
     - Applies a mask to the target image data, setting values outside the mask to zero.
   * - :py:mod:`megafish.segment.fill_holes`
     - Fills holes in labeled regions of an image dataset.
   * - :py:mod:`megafish.segment.remove_edge_mask`
     - Removes labeled regions touching the edges of the image.
   * - :py:mod:`megafish.segment.label_edge`
     - Identifies and labels the edges of labeled regions in an image dataset, with adjustable thickness.
   * - :py:mod:`megafish.segment.repeat_cycle`
     - Repeats an image dataset over multiple cycles.
   * - :py:mod:`megafish.segment.info_csv`
     - Generates segment information CSV files from image data stored in a Zarr file, summarizing properties such as area and centroid for each segment. Merges the CSV files into a single summary.
   * - :py:mod:`megafish.segment.merge_groups`
     - Merges multiple groups of image data from a Zarr file into a single output group.
   * - :py:mod:`megafish.segment.normalize_groups`
     - Normalizes intensity values across groups within an image dataset and computes the maximum intensity projection (MIP).
   * - :py:mod:`megafish.segment.select_slice`
     - Selects a slice from an image dataset along a specified dimension.
   * - :py:mod:`megafish.segment.merge_to_one_group`
     - Merges multiple groups of image data from a Zarr file into a single output group.
   * - :py:mod:`megafish.segment.scaled_mip`
     - Normalizes intensity values across a specified dimension and computes the maximum intensity projection (MIP).

tif
===============

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Workflow Functions
     - Description
   * - :py:mod:`megafish.tif.save`
     - Saves image data from a Zarr file as individual TIFF files for each chunk.
   * - :py:mod:`megafish.tif.load`
     - Loads TIFF images into a Zarr dataset using the structure of a template group.
   * - :py:mod:`megafish.tif.save_tile_montage`
     - Creates a tiled montage of images from a Zarr file and saves it as a single TIFF file.
   * - :py:mod:`megafish.tif.save_whole_image`
     - Saves the entire image from a Zarr file as a TIFF file, with an option to clip the image.
   * - :py:mod:`megafish.tif.save_chunk`
     - Saves specific chunks of image data from a Zarr file as individual TIFF files.
   * - :py:mod:`megafish.tif.save_rgb`
     - Combines individual red, green, and blue image groups from a Zarr file into RGB images and saves them as TIFF files.

seqfish
===============

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Workflow Functions
     - Description
   * - :py:mod:`megafish.seqfish.DoG_filter`
     - Applies a Difference of Gaussians (DoG) filter to the image data in a Zarr file.
   * - :py:mod:`megafish.seqfish.local_maxima`
     - Detects local maxima in the image data within a Zarr file using a specified footprint.
   * - :py:mod:`megafish.seqfish.select_by_intensity_sd`
     - Selects spots in the image data based on intensity, using a threshold defined by the mean intensity and standard deviation.
   * - :py:mod:`megafish.seqfish.select_by_intensity_threshold`
     - Selects spots in the image data based on a specified intensity threshold.
   * - :py:mod:`megafish.seqfish.count_spots`
     - Counts spots within labeled segments in the image data stored in a Zarr file and saves the results as CSV files.
   * - :py:mod:`megafish.seqfish.count_summary`
     - Summarizes spot counts across multiple groups and cycles, merging the results with segment data and optionally gene names.
   * - :py:mod:`megafish.seqfish.spot_coordinates`
     - Extracts coordinates of spots from the image data stored in a Zarr file and saves them as CSV files.
   * - :py:mod:`megafish.seqfish.spot_intensity`
     - Computes the intensity of spots within segmented regions for each chunk of the image data and saves the results as CSV files.

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Helper Functions
     - Description
   * - :py:mod:`megafish.seqfish.dog_sds`
     - Calculates the standard deviations for the Difference of Gaussians (DoG) based on the point spread function (PSF) and imaging parameters.
   * - :py:mod:`megafish.seqfish.local_maxima_footprint`
     - Calculates the footprint for detecting local maxima based on the point spread function (PSF) and imaging parameters.

.. _functions_decode:

decode
===============

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Workflow Functions
     - Description
   * - :py:mod:`megafish.decode.merfish_prefilter`
     - Applies MERFISH prefiltering steps, including high-pass filtering, Richardson-Lucy deconvolution, and low-pass filtering.
   * - :py:mod:`megafish.decode.scaling`
     - Scales the intensity of the image data stored in a Zarr file based on a given percentile and scaling factor.
   * - :py:mod:`megafish.decode.norm_value`
     - Calculates the L2 norm (Euclidean norm) of the image data across cycles stored in a Zarr file.
   * - :py:mod:`megafish.decode.divide_by_norm`
     - Divides the filtered image data by the calculated norm values for normalization.
   * - :py:mod:`megafish.decode.nearest_neighbor`
     - Calculates the nearest neighbor for each pixel's intensity trace in an image dataset using a precomputed codebook.
   * - :py:mod:`megafish.decode.split_nnd`
     - Splits the nearest neighbor dataset into two separate datasets: one for the code indices and one for the distances.
   * - :py:mod:`megafish.decode.select_decoded`
     - Filters decoded spots based on intensity, distance, and area criteria.
   * - :py:mod:`megafish.decode.coordinates_decoded`
     - Extracts and records the coordinates of decoded spots within nuclei, saving the information in a CSV file for each chunk.

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Helper Functions
     - Description
   * - :py:mod:`megafish.decode.gaussian_kernel`
     - Generates a Gaussian kernel for spatial filtering. This kernel is used for spatial filtering in the MERFISH prefiltering step.

seqif
===============

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Workflow Functions
     - Description
   * - :py:mod:`megafish.seqif.TCEP_subtraction`
     -  Subtracts consecutive cycles in the image data, assuming cycles are organized as TCEP and non-TCEP pairs.
   * - :py:mod:`megafish.seqif.skip_odd_cycle`
     - Selects and retains only the even cycles in the dataset, removing all odd cycles.
   * - :py:mod:`megafish.seqif.get_intensity`
     - Calculates the mean intensity of labeled segments in an image dataset and saves the results as a CSV file.
   * - :py:mod:`megafish.seqif.intnensity_summary`
     - Summarizes intensity data for multiple groups and saves the results as a CSV file. 

view
===============

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Workflow Functions
     - Description
   * - :py:mod:`megafish.view.make_pyramid`
     - Generates a pyramid of downsampled image data from a zarr dataset and writes it back to zarr storage.
   * - :py:mod:`megafish.view.mask_edge`
     - Creates an edge mask around binary regions in an image using dilation and erosion.

napari
===============

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Function Name
     - Description
   * - :py:mod:`megafish.napari.tile_2d`
     - Visualizes 2D tiled image data from a Zarr dataset using napari.
   * - :py:mod:`megafish.napari.registered`
     - Visualizes multiscale registered image data from a Zarr dataset using napari, with optional gene name display.
   * - :py:mod:`megafish.napari.registered_is`
     - Visualizes registered image data with image stitching from a Zarr dataset using napari, with optional gene name display.
   * - :py:mod:`megafish.napari.segment`
     - Visualizes segmented and labeled image data from a Zarr dataset using napari, with support for multiscale images.

misc
===============

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Module Name
     - Description
   * - :py:mod:`megafish.config`
     - Manages the resource configuration for the MEGA-FISH framework, allowing for GPU usage and scheduling settings with Dask.
   * - :py:mod:`megafish.utils`
     - Provides utility functions for natural sorting, generating tile coordinates based on scanning patterns.
