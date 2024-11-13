########################
MEGA-FISH documentation
########################

**MEGA-FISH** (Multi-omics Extensible GPU-Accelerated FISH analysis framework) is a Python package designed to process large-scale fluorescence images for spatial omics applications, including SeqFISH, SeqIS, and decoding-based FISH methods.

MEGA-FISH stitches sequentially captured tile images into a large, chunked single image and utilizes efficient computational resources (GPU or multi-threaded/multi-process CPU) for each processing step.

Users can generate a cell-by-gene expression matrix by combining simple functions such as image registration, segmentation, and spot detection.

.. toctree::
   :maxdepth: 2

   Getting Started <getting_started>
   User Guide <user_guide>
   Examples <examples>
   Development <develop>
   About <about>
   API Reference <functions>
