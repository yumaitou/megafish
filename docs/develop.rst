================
Development
================


  .. _detailed-installation:

Installation
======================

Installing all dependencies
--------------------------------
Installation from pip includes only minimal dependencies. It is recommended to
add the package when an import error occurs in the execution of the class.
Alternatively, you can install everything from the requirement-full.txt file
in the GitHub repository.

.. code-block:: bash

    pip install -r requirement-full.txt

Installing from the latest source code
----------------------------------------
The latest source code that includes bug fixes and improvements can be
installed from the GitHub repository. Please make a local clone of the repository as follows:

.. code-block:: bash

    git clone https://github.com/yumaitou/megafish.git

Then install as the editable mode.

.. code-block:: bash

    cd megafish
    python -m pip install -e .

Installing the development version
----------------------------------------
The development version of MEGA-FISH that includes useful experimental classes can be
installed from the dev branch of the GitHub repository. These classes usually need
to be better documented and tested. If you want to avoid cloning the repository,
please install it using pip directly.

.. code-block:: bash

    pip install git+https://github.com/yumaitou/megafish.git@dev

If you want to edit the source code, please run as follows:

.. code-block:: bash

    git clone https://github.com/yumaitou/megafish.git
    cd megaish
    git checkout dev
    python -m pip install -e .


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
5. Add test function in the tests directory using `pytest`. All lines of the new code should be tested to keep full code coverage. Run the whole test suite and ensure that all tests pass.
6. Commit your changes to the feature branch and push to GitHub forked repository.
7. Issue the pull request.

.. note:: 

  Dependencies for developers, including test, linter, and documentation, can be
  installed as follows.

  .. code-block:: bash

      pip install -r requirement-dev.txt

Documentation
========================
**MEGA-FISH** uses Sphinx to generate the documentation.
API documentation is compiled automatically from source code docstrings using 
`autodoc`. **MEGA-FISH** uses google style docstrings.