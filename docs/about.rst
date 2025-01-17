==========
About
==========

* **MEGA-FISH** is developed in `https://github.com/yumaitou/megafish/ <https://github.com/yumaitou/megafish/>`_.
* The documentation can be found at `https://megafish.readthedocs.io <https://megafish.readthedocs.io>`_.
* **MEGA-FISH** can be installed from `https://pypi.org/project/megafish/ <https://pypi.org/project/megafish/>`_.

Citation
==================

If **MEGA-FISH** was useful for your research, please consider citing the following our paper:

* Coming soon

Environments
==================

**MEGA-FISH** is tested on the following operating systems.

* Windows10
* Windows11 
* macOS Ventura 13
* Ubuntu 22.04.5 LTS

Dependencies
==================

The basic **MEGA-FISH** has the following dependencies:

* numpy
* pandas
* matplotlib
* dask
* xarray
* zarr
* scikit-image
* scikit-learn
* tifffile
* scipy
* napari
* imaris-ims-file-reader

License
==================
**MEGA-FISH** is distributed under the BSD 3-Clause License. 

.. code-block:: text

   BSD 3-Clause License

   Copyright (c) 2024-2025, Yuma Ito and contributors

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are met:

   1. Redistributions of source code must retain the above copyright notice, this
      list of conditions and the following disclaimer.

   2. Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.

   3. Neither the name of the copyright holder nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
   AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
   IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
   DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
   FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
   DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
   SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
   CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
   OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Changelog
=============

**MEGA-FISH** follows major version zero (rapid development) of Semantic Versioning.

0.1.3 (2025-01-16)
--------------------------

Bug Fixes
  * Fix save_tile_montage function to accept tile_size as tuple or list.
  * Restrict zarr dependency version to <3.0 in requirements files

Maintenance
  * Add unit tests for utility functions and processing methods, and update development requirements.

0.1.2 (2024-12-06)
--------------------------

Bug Fixes
  * Fix sample name extraction in CSV and TIFF file saving functions.
  
Documentation
  * Update getting_started documentation with corrected paths and additional dataset download instructions.
  * Fix minor typos in code snippets.
  * Update README.
  * Add Code of Conduct, update development documentation, and include development requirements.

0.1.1 (2024-11-13)
----------------------

Bug Fixes
  * Fix Python dependencies.

0.1.0 (2024-11-13)
----------------------

Features
  * Add test code.