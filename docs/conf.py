# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../src'))

# -- Project information ------------------------------------------------------

project = 'MEGA-FISH'
copyright = '2024-2025, Yuma Ito'
author = 'Yuma Ito'

version = '0.1.3'

# -- General configuration ----------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    'sphinx.ext.autosectionlabel'
]

autosectionlabel_prefix_document = True

# -- Options for autodoc ------------------------------------------------------
autodoc_member_order = 'bysource'
napoleon_google_docstring = True
autodock_mock_imports = ["numpy", "pandas", "skimage"]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
    'numpy': ('https://docs.scipy.org/doc/numpy/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'skimage': ('https://scikit-image.org/docs/stable/', None),
}

intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output --------------------------------------------------

html_theme = 'sphinx_rtd_theme'


# -- Options for LaTeX output -------------------------------------------------

latex_documents = [
    ('index', "megafish.tex",
     "MEGA-FISH", "Yuma Ito", "manual", False),
]
latex_elements = {
    'fontpkg': '',
    'fncychap': '',
    'papersize': '',
    'pointsize': '',
    'preamble': '',
    'releasename': '',
    'babel': '',
    'printindex': '',
    'fontenc': '',
    'inputenc': '',
    'classoptions': '',
    'utf8extra': '',

}
add_module_names = False

# -- Options for EPUB output --------------------------------------------------
epub_show_urls = 'footnote'
