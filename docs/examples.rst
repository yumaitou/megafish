===============
Examples
===============

This section provides examples of how to use MEGA-FISH for different modalities.

SeqIS
=========

Sequential Immunostaining (SeqIS) is a technique for quantifying protein expression in cells using multiple antibodies.  
It involves calculating the fluorescence intensity of antibody-bound proteins on a per-cell basis.

When acquiring images using platforms such as PECAb, MACSima, or PhenoCycler, background fluorescence can be effectively removed by subtracting the fluorescence signals of staining images from those captured after de-staining.
This subtraction enhances the accuracy of protein-derived fluorescence measurements by reducing background noise.

The following example demonstrates how to process SeqIS image data in MEGA-FISH, where pairs of stained and de-stained images are repeatedly acquired.  
It is assumed that Registration and Segmentation have already been completed.

.. code-block:: python

    # Subtract even-cycle images from odd-cycle images (0-indexed)
    mf.seqif.TCEP_subtraction(zarr_path, 'is1_mip_reg')
    
    # Skip odd-cycle images for Hoechst (0-indexed)
    mf.seqif.skip_odd_cycle(zarr_path, "hcst_mip_reg")

    # Retrieve fluorescence intensity for each cell mask
    mf.seqif.get_intensity(zarr_path, "is1_mip_reg_sub", "hcst_mip_reg_mip_lbl")

    # Create a cell-by-gene matrix for all channels
    groups = ["is1_mip_reg_sub_int"]
    group_seg = "hcst_mip_reg_mip_lbl_seg"
    channels = [2]
    genename_path = os.path.join(root_dir, "SeqIF_genename.csv")
    group_out = "is_int"
    mf.seqif.intensity_summary(
        zarr_path, groups, group_seg, group_out, channels, genename_path)


Barcode-based seqFISH
=======================

Barcode-based SeqFISH quantifies gene expression in cells using DNA barcodes labeled with multiple fluorophores.  
Currently, MEGA-FISH adopts **pixel vector decoding**, which identifies barcodes by comparing observed pixel intensities with a predefined codebook.

**Codebook Preparation**

For decoding, you need to provide a **codebook** in the form of a `numpy.array`.  
Each row in the codebook represents a gene, and each column indicates whether the gene is "on" or "off" in a given cycle.

.. code-block:: python

    import numpy as np

    codebook = np.array([
        [1, 0, 0, 1, 1],
        [0, 1, 1, 0, 0],
        [1, 1, 0, 0, 1],
        [0, 0, 1, 1, 0],
    ])

Each row must be normalized to ensure that its norm equals 1.
The following code snippet demonstrates how to normalize the codebook:

.. code-block:: python

    codebook = codebook / np.linalg.norm(codebook, axis=1)[:, None]

**Image Preprocessing and Decoding**

Images captured in each cycle require **prefiltering** and **normalization** before decoding.
The following steps outline the complete workflow:

1. **Prefiltering**: Apply Gaussian smoothing and prefiltering to enhance signal quality.
2. **Normalization**: Normalize pixel values across all cycles to reduce variability.
3. **Decoding**: Match observed pixel intensities to the nearest barcode using the provided codebook.
4. **Selection**: Filter decoded pixels based on intensity, distance, and area constraints.
5. **Coordinate Extraction**: Retrieve the coordinates of decoded pixels for downstream analysis.

Below is an example code snippet demonstrating this process:

.. code-block:: python

    # ----- Prefilter -----
    sigma_high = (3, 3)
    sigma = 2
    kernel_size = int(2 * np.ceil(2 * sigma) + 1)
    psf = mf.decode.gaussian_kernel(
        shape=(kernel_size, kernel_size), sigma=sigma)
    iterations = 15
    sigma_low = (1, 1)
    mask_size = 5

    mf.decode.merfish_prefilter(
            zarr_path, "rna1_mip_reg", sigma_high, psf, iterations, sigma_low,
            mask_size)
    mf.decode.scaling(zarr_path, "rna1_mip_reg_mfp", 95, 1)

    # ----- Normalize -----
    mf.decode.norm_value(zarr_path, "rna1_mip_reg_mfp_scl")
    mf.decode.divide_by_norm(
        zarr_path, "rna1_mip_reg_mfp_scl", "rna1_mip_reg_mfp_scl_nmv")

    # ----- Decoding -----
    code_path = os.path.join(root_dir, "codebook.npy")
    mf.decode.nearest_neighbor(
        zarr_path, "rna1_mip_reg_mfp_scl_nrm", code_path)

    # ----- Selection -----
    min_intensity = 2
    max_distance = 0.6
    area_limits = (2, 1000)

    mf.decode.select_decoded(
        zarr_path, "rna1_mip_reg_mfp_scl_nmv", "rna1_mip_reg_mfp_scl_nrm_nnd",
        min_intensity, max_distance, area_limits)

    # ----- Get coordinates of decoded pixels -----      
    mf.decode.coordinates_decoded(
        zarr_path, "rna1_mip_reg_mfp_scl_nrm_nnd_dec", 
        "hcst_mip_reg_slc_lbl_msl_fil")

See the :ref:`decode module <functions_decode>` for more details on each function.
