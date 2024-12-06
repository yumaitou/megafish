# MEGA-FISH

MEGA-FISH (Multi-omics Extensible GPU-Accelerated FISH analysis framework) is a Python package designed to process large-scale fluorescence images for spatial omics applications, including SeqFISH, SeqIS, and decoding-based FISH methods.

MEGA-FISH stitches sequentially captured tile images into a large, chunked single image and utilizes efficient computational resources (GPU or multi-threaded/multi-process CPU) for each processing step.

Users can generate a cell-by-gene expression matrix by combining simple functions such as image registration, segmentation, and spot detection.

Please see documentation for more information about MEGA-FISH.

## Installation

MEGA-FISH supports GPU acceleration on Linux systems and Windows PCs with WSL2.
See the [documentation](https://megafish.readthedocs.io/en/stable/getting_started.html#installation) for more information.

MEGA-FISH supports multi-core CPU parallel computation and is available on Linux, Windows, and macOS. Below are the installation steps for setting up CPU-only MEGA-FISH.

```bash
conda create -n megafish python=3.11
conda activate megafish
pip install megafish
```

## Getting Started

Once you have installed MEGA-FISH, you can start by following the [tutorial](https://megafish.readthedocs.io/en/stable/getting_started.html#sample-dataset) using the example data. 

## Data Structure

MEGA-FISH is designed not to create a MEGA-FISH-specific data structure, but to use simple naming rules for xarray. This allows users to easily customize the analysis pipeline and transfer the data to other packages. See the [documentation](https://megafish.readthedocs.io/en/stable/) for the data structure and functions.

## Contributing

We welcome contributions to MEGA-FISH. Please see the [contribution guide](https://megafish.readthedocs.io/en/stable/develop.html) for more information.

## Citing

If MEGA-FISH was useful for your research, please consider citing the following our paper:

Coming soon.

## License

MEGA-FISH is licensed under the [BSD 3-Clause License](https://github.com/yumaitou/megafish/blob/main/LICENSE). 


