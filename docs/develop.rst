================
Development
================

Installation
======================

If you want to edit the source code, please run as follows:

.. code-block:: bash

    git clone https://github.com/yumaitou/megafish.git
    cd megaish
    conda create -n megafish-dev python=3.11
    pip install -e .
    pip install -r requirement-dev.txt


Contributing
================

**MEGA-FISH** welcomes any contributions such as bug reports, bug fixes, 
enhancements, and documentation improvements from interested individuals and 
groups. All contributors to this project are expected to abide by our 
`code of conduct <https://github.com/yumaitou/megafish/CODE_OF_CONDUCT.md>`_.

You can contribute by creating a GitHub issue, pull request, and direct email 
to the author (<yumaitou@outlook.com>). Before submitting a contribution, 
check that you are using the latest version of **MEGA-FISH**.

Bug report
================================

If you experience bugs using **MEGA-FISH**, please open an issue on 
Github `Issue Tracker <https://github.com/yumaitou/megafish/issue>`_. 
Please ensure that your bug report includes:

* The environment information, including the operating system and version numbers of Python and involved packages.
* A description of the behavior of the issue you encountered.
* A short reproducible code snippet or the steps you took to reproduce the bug.
* The entire error message, if applicable.

Pull request
========================

You can fix bugs, improve codes, and implement new features by creating a pull request to the GitHub **MEGA-FISH** repository. To submit your pull request, please check that it meets the following guidelines:

1. Fork the `yumaitou/megafish` repository.
2. Create a feature branch from `main`.
3. Create your codes in accordance with PEP 8 style guide. We recommend using an automated formatter such as `flake8`.
4. Create documentation of the new class or function by adding a structured docstring with Sphinx google style format.
5. Commit your changes to the feature branch and push to GitHub forked repository.
6. Issue the pull request.

Documentation
========================

**MEGA-FISH** uses Sphinx to generate the documentation.
API documentation is compiled automatically from source code docstrings using 
`autodoc`. **MEGA-FISH** uses google style docstrings.