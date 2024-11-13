=====================
Getting Started
=====================

This section outlines the installation and basic usage of MEGA-FISH, guiding you through setting up your environment and processing spatial omics images.

Installation
=====================

We recommend installing MEGA-FISH in a virtual environment using either Anaconda or Miniconda to ensure dependency management and isolation.
For installation instructions, please refer to the official guides for `Anaconda <https://www.anaconda.com/products/distribution#download-section>`_ or `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_.

GPU version
--------------------

MEGA-FISH supports GPU acceleration on Linux systems and Windows PCs with WSL2. This feature enables significantly faster processing of large-scale spatial omics data.

Currently, MEGA-FISH requires **CUDA Toolkit 11.8** for GPU acceleration. Ensure that your system has a compatible NVIDIA driver installed.
Below are the installation steps for a system running Ubuntu 22.04.5 LTS with CUDA 11.8 installed. 
For details on installing CUDA Toolkit 11.8, refer to the `official CUDA installation guide <https://developer.nvidia.com/cuda-11-8-0-download-archive>`_.

**Note:** On Windows, WSL2 must be properly configured with a supported NVIDIA driver. Please refer to the `WSL2 NVIDIA guide <https://docs.nvidia.com/cuda/wsl-user-guide/index.html>`_ for more information.

1. Create a conda environment and install the required packages.
 
   MEGA-FISH depends on `Rapids <https://rapids.ai/>`_ and `CuPy <https://cupy.dev/>`_ for GPU-accelerated computations.
   Below are the commands to set up the environment.
   For detailed instructions on Rapids installation, refer to the `official Rapids website <https://rapids.ai/start.html>`_.

   .. code-block:: bash

      conda create -n megafish-env python=3.11
      conda activate megafish-env
      conda install conda-libmamba-solver
      conda config --set solver libmamba
      conda install -c conda-forge mamba
      mamba install -y -c rapidsai -c conda-forge -c nvidia rapids=24.06 cuda-version=11.8 cupy=13.3.0

2. Install MEGA-FISH using `PyPI <https://pypi.org/project/megafish/>`_:

   .. code-block:: bash

      pip install megafish


CPU version
-------------------

MEGA-FISH supports multi-core CPU parallel computation and is available on Linux, Windows, and macOS.
Below are the installation steps for setting up MEGA-FISH on an Ubuntu 22.04.5 LTS system.

1. Create a conda environment and install the necessary packages.

   The use of `mamba` ensures faster and more reliable dependency resolution:

   .. code-block:: bash

      conda create -n megafish-env python=3.11
      conda activate megafish-env
      conda install conda-libmamba-solver
      conda config --set solver libmamba
      conda install -c conda-forge mamba
      mamba install -y pyqt

2. Install MEGA-FISH using `PyPI <https://pypi.org/project/megafish/>`_:

   .. code-block:: bash

      pip install megafish

Sample dataset
=====================

A SeqFISH image dataset used in this tutorial is available for download from Zenodo.
This dataset captures mRNA of 6 genes in human IMR90 cells across 2 channels and is a downsampled version of images from `Tomimatsu, K. et al., Nat Commun 15, 3657 (2024) <https://doi.org/10.1038/s41467-024-47989-9>`_.

**Dataset Overview**:

The dataset is organized by cycle, subfolders inside the ``images`` folder. Each cycle folder holds 9 TIFF images named sequentially from `1.tif` to `9.tif`, arranged in a 3x3 tile configuration. Each tile image is a 4-dimensional array with dimensions `(color, z, y, x) = (3, 3, 1024, 1024)`. The channels are structured as follows:

- **Channel 1**: Hoechst (nuclei stain)
- **Channel 2**: RNA channel 1
- **Channel 3**: RNA channel 2

In addition, the dataset includes a ``stitched`` folder, which contains a pre-stitched 2D image, ``hcst_mip_stitched.tif``. This image, with dimensions ``(3051, 3051)``, is used for tile-to-tile registration.

**Folder Structure**:

.. code-block:: text

    getting_started/
    ├── images/
    │   ├── cycle1/
    │   │   ├── 1.tif
    │   │   ├── ...
    │   │   └── 9.tif
    │   ├── cycle2/
    │   │   ├── 1.tif
    │   │   ├── ...
    │   │   └── 9.tif
    │   └── cycle3/
    │       ├── 1.tif
    │       ├── ...
    │       └── 9.tif
    └── stitched/
        └── hcst_mip_stitched.tif

To use this dataset, download it from `Zenodo <https://zenodo.org/record/example>`_ and save it to a directory of your choice, for example: ``/home/UserName/megafish_sample/getting_started/``.


Processing
=====================

First, import MEGA-FISH and the necessary libraries into your Python script or Jupyter notebook.

.. code-block:: python

   import os
   import megafish as mf

When running MEGA-FISH as a Python script, you must include the processing code within the ``if __name__ == '__main__':`` block.
This ensures proper handling of parallel computation using the ``concurrent.futures`` module.

.. code-block:: python

   if __name__ == '__main__':
       # Add your processing code here

**Note:** If you are using a Jupyter notebook, this step is not required.

Next, specify the analysis directory, sample name, Zarr file path, and key parameters such as image size and pixel dimensions.
If analysis directory does not exist, create it first.

.. code-block:: python

   root_dir = "/home/UserName/megafish_sample/getting_started/analysis/"
   sample_name = "IMR90_SeqFISH"
   zarr_path = os.path.join(root_dir, sample_name + ".zarr")

   pitch = [0.1370, 0.0994, 0.0994]
   n_cycle, n_tile_y, n_tile_x, n_z, n_y, n_x = 3, 3, 3, 3, 1024, 1024

Here, ``pitch`` defines the pixel size in micrometers for each dimension (z, y, x).
``n_cycle`` is the number of cycles, ``n_tile_y`` and ``n_tile_x`` are the number of tiles in the y and x directions, and ``n_z``, ``n_y``, and ``n_x`` are the pixel dimensions of the image.

In this tutorial, since the images are relatively small, using a GPU might increase computational overhead and slow down the processing.
For optimal performance, CPU processing is recommended in this tutorial dataset. You can specify the resource settings as follows:

.. code-block:: python

   mf.config.set_resource(GPU=False, scheduler="processes")

Loading the dataset
---------------------

This section explains how to load a sample dataset into MEGA-FISH for analysis.

1. **Specify the dataset directory and create a directory list**
   
   The directory list is a CSV file that records the cycle directories in the dataset, which is used to generate an image information list.
   The following code creates a directory list by searching for cycle directories in the image directory.

   .. code-block:: python

        img_dir = "/home/UserName/megafish_sample/getting_started/images/"
        dirlist_path = os.path.join(root_dir, sample_name + "_directorylist.csv")
        mf.load.make_dirlist(dirlist_path, img_dir)

This will generate a ``IMR90_SeqFISH_directorylist.csv`` file in the analysis directory with the following structure:

   .. list-table::
      :header-rows: 1
      :widths: 100

      * - path
      * - /home/UserName/megafish_sample/getting_started/images/cycle1
      * - /home/UserName/megafish_sample/getting_started/images/cycle2
      * - /home/UserName/megafish_sample/getting_started/images/cycle3

1. **Generate an image information list**
     
   The image information list is a CSV file that records the image paths and metadata (e.g., cycle, tile, and channel) for each group.
   The following code generates an image information list based on the directory list and specified parameters.

   .. code-block:: python

        groups = ["hcst", "rna1", "rna2"]
        channels = [1, 2, 3]
        scan_type = "row_right_down"
        mf.load.make_imagepath_cYX_from_dirlist(
            zarr_path, groups, channels, n_cycle, n_tile_y, n_tile_x,
            scan_type, dirlist_path, ext=".tif")

   This will generate a ``IMR90_SeqFISH_imagepath.csv`` file in the analysis directory with the following structure:

   .. list-table::
      :header-rows: 1
      :widths: 15 10 10 10 45 10

      * - group
        - cycle
        - tile_y
        - tile_x
        - path
        - channel
      * - hcst
        - 1
        - 1
        - 1
        - /home/UserName/megafish_sample/getting_started/images/cycle1/1.tif
        - 1
      * - hcst
        - 1
        - 1
        - 2
        - /home/UserName/megafish_sample/getting_started/images/cycle1/2.tif
        - 1
      * - ...
        - ...
        - ...
        - ...
        - ...
        - ...
      * - rna1
        - 1
        - 1
        - 1
        - /home/UserName/megafish_sample/getting_started/images/cycle1/1.tif
        - 2
      * - ...
        - ...
        - ...
        - ...
        - ...
        - ...
      * - rna2
        - 3
        - 3
        - 3
        - /home/UserName/megafish_sample/getting_started/images/cycle3/9.tif
        - 3

   **Note**: If the image order in your dataset differs from the expected order, you can manually create the image path CSV file without using functions.

2. **Load the images into a Zarr file**
   
   Convert the raw TIFF images into a Zarr file using the image information list.

   .. code-block:: python

        mf.load.tif_cYXzyx(zarr_path, n_z, n_y, n_x, tif_dims="zyxc")

.. _getting_started_registration:

Registration
---------------------

This section describes how to align and register tiled images across different cycles.

1. **Convert the 3D image stack into 2D images**

   Currently, MEGA-FISH only supports 2D image processing.
   Use maximum intensity projection to reduce the 3D image stack along the z-axis.

   .. code-block:: python

        groups = ["hcst", "rna1", "rna2"]
        for group in groups:
            mf.process.projection(zarr_path, group)

2. **Calculate shifts between cycles for the same tile**  
   
   First, specify the parameters for SIFT (Scale-Invariant Feature Transform) and RANSAC (Random Sample Consensus) algorithms.
   These parameters are critical for robust feature matching and outlier rejection.

   .. code-block:: python

        sift_kwargs = {
            "upsampling": 1, "n_octaves": 8, "n_scales": 3, "sigma_min": 2,
            "sigma_in": 1, "c_dog": 0.01, "c_edge": 40, "n_bins": 12,
            "lambda_ori": 1.5, "c_max": 0.8, "lambda_descr": 6,
            "n_hist": 4, "n_ori": 8}
        match_kwargs = {"max_ratio": 0.5}
        ransac_kwargs = {
            "min_samples": 4, "residual_threshold": 10, "max_trials": 500}

   **Note**: For detailed information on the parameters, refer to the documentation of the following functions:
   `skimage.feature.SIFT <https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.SIFT>`_, 
   `skimage.feature.match_descriptors <https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.match_descriptors>`_,
   `skimage.measure.ransac <https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.ransac>`_.
    
   Next, calculate the shifts using the Hoechst channel as the reference.

   .. code-block:: python
    
        mf.register.shift_cycle_cYXyx(
            zarr_path, "hcst_mip", sift_kwargs, match_kwargs, ransac_kwargs)

1. **Load the stitched image and calculate tile shifts**
 
   Load a pre-stitched image for accurate tile registration.
   
   **Note:** MEGA-FISH does not currently support automatic stitched image creation. You can use external tools such as the ImageJ plugin or Imaris Stitcher.

   .. code-block:: python
   
        stitched_dir = "/home/UserName/megafish_sample/getting_started/stitched/"
        stitched_path = os.path.join(stitched_dir, "hcst_mip_stitched.tif")
        mf.load.stitched_tif(
            zarr_path, "stitched", stitched_path, n_tile_y, n_tile_x)

   Then, calculate the shifts for each tile and integrate these shifts with the cycle-wise shifts.

   .. code-block:: python

        mf.register.shift_tile_cYXyx(zarr_path, "hcst_mip", "stitched", 1000,
                                     sift_kwargs, match_kwargs, ransac_kwargs)
        mf.register.merge_shift_cYXyx(zarr_path, "hcst_mip")


2. **Generate a stitched image for each group across all cycles**  
   
   Using the computed shifts, create a large Zarr group for each channel (e.g., Hoechst, RNA channel 1, RNA channel 2) that combines all cycles into a single seamless image.

   .. code-block:: python

        groups = ["hcst_mip", "rna1_mip", "rna2_mip"]
        for group in groups:
            mf.register.registration_cYXyx(
                zarr_path, group, "stitched", (1526, 1526))

   **Note**: It is recommended to adjust the chunk size based on the available memory capacity of your computer. Larger chunk sizes may improve performance but require more memory.

.. _getting_started_segmentation:

Segmentation
---------------------

For segmentation, it is recommended to use external segmentation tools such as Cellpose or Ilastik.
However, for demonstration purposes, this tutorial uses a simple watershed segmentation method.
This method is effective for segmenting nuclei in well-separated cells and includes the following steps:

1. Extract a first cycle from the sequential Hoechst image.
2. Apply Gaussian blur to reduce noise and enhance nuclei boundaries.
3. Binarize the image to create a mask.
4. Perform watershed segmentation to identify individual nuclei.
5. Refine the segmentation results by merging split labels and filling small holes.
6. Save the segmentation results to a CSV file for downstream analysis.

The following code demonstrates the segmentation process:

.. code-block:: python

   # Select the slice from Hoechst in first cycle
   mf.segment.select_slice(zarr_path, "hcst_mip_reg",
                           "cycle", 0, None, "_slc")

   # Smooth the image of the nuclei using Gaussian blur
   mf.process.gaussian_blur(zarr_path, "hcst_mip_reg_slc", 2)

   # Binarize the image
   mf.process.binarization(zarr_path, "hcst_mip_reg_slc_gbr", 110)
   
   # Perform segmentation using the watershed method
   mf.segment.watershed_label(zarr_path, "hcst_mip_reg_slc_gbr_bin", 50)
   
   # Merge the segmentation results
   mf.segment.merge_split_label(zarr_path, "hcst_mip_reg_slc_gbr_bin_wts")
   
   # Fill holes in the segmentation results
   mf.segment.fill_holes(zarr_path, "hcst_mip_reg_slc_gbr_bin_wts_msl")
   
   # Save the segmentation label information to a CSV file
   mf.segment.info_csv(zarr_path, "hcst_mip_reg_slc_gbr_bin_wts_msl_fil", pitch[1:])


Spot detection
---------------------

RNA spot detection in MEGA-FISH involves two main steps: applying a Difference of Gaussians (DoG) filter to enhance spot-like structures and detecting local maxima to identify potential RNA spots. Below is an example workflow.

1. **Apply DoG filter and detect local maxima**
   
   This step enhances spot-like features using a DoG filter and identifies potential RNA spots based on local maxima detection.

   .. code-block:: python

        NA = 1.4 # Numerical Aperture of the objective
        wavelengths_um = [0.592, 0.671] # Emission wavelengths in micrometers
        mean_pitch_yx = (pitch[1] + pitch[2]) / 2 # Average pixel size in the XY plane

        group_names = ["rna1_mip_reg", "rna2_mip_reg"]
        for group_name, wavelength_um in zip(group_names, wavelengths_um):
            dog_sd1, dog_sd2 = mf.seqfish.dog_sds(NA, wavelength_um, mean_pitch_yx)
            mf.seqfish.DoG_filter(zarr_path, group_name, dog_sd1,
                                dog_sd2, axes=(1, 2), mask_radius=9)

        group_names = ["rna1_mip_reg_dog", "rna2_mip_reg_dog"]
        for group_name, wavelength_um in zip(group_names, wavelengths_um):
            footprint = mf.seqfish.local_maxima_footprint(
                NA, wavelength_um, mean_pitch_yx)
            mf.seqfish.local_maxima(
                zarr_path, group_name, footprint, axes=(1, 2))

2. **Set intensity thresholds for detected spots**  
   
   To filter out false positives, apply intensity thresholds to the detected local maxima.

   .. code-block:: python

        groups = ["rna1_mip_reg_dog_lmx", "rna2_mip_reg_dog_lmx"]
        thrs = [2.8, 1] # Intensity thresholds for each channel
        for group, thr in zip(groups, thrs):
            mf.seqfish.select_by_intensity_threshold(zarr_path, group, threshold=thr)


3. **Generate the cell-by-gene expression matrix**  
   
   Aggregate the RNA spot counts across all channels and segments to create a cell-by-gene expression matrix. The final output is saved as a CSV file.


   .. code-block:: python

        groups = ["rna1_mip_reg_dog_lmx_ith", "rna2_mip_reg_dog_lmx_ith"]
        for group in groups:
            mf.seqfish.count_spots(zarr_path, group,
                                "hcst_mip_reg_slc_gbr_bin_wts_msl")

        # Summarize counts across all channels and save the cell-by-gene expression matrix
        groups = ["rna1_mip_reg_dog_lmx_ith_cnt",
                "rna2_mip_reg_dog_lmx_ith_cnt"]
        group_seg = "hcst_mip_reg_slc_gbr_bin_wts_msl_seg"
        channels = [2, 3]
        genename_path = "/home/UserName/megafish_sample/getting_started/IMR90_SeqFISH_genename.csv"
        group_out = "rna_cnt"
        mf.seqfish.count_summary(
            zarr_path, groups, group_seg, group_out, channels, genename_path)

**Note**:

- The ``NA`` and ``wavelengths_um`` parameters should match the imaging system used to acquire the data.
- Intensity thresholds (``thrs``) may need to be adjusted depending on the dataset to optimize spot detection.
- The final CSV file contains the cell-by-gene expression matrix, integrating data across all channels. This matrix can be used for downstream analysis, such as differential gene expression or clustering.