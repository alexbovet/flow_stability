# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import pathlib
import sys

# -- Add the project root for autodiscovery with sphinx.ext.autodoc ----------
sys.path.insert(0, pathlib.Path(__file__).parents[1].resolve().as_posix())
autodoc_typehints = 'description'
autodoc_class_signature = 'separated'

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Flow Stability'
copyright = '2023, Alexandre Bovet'
author = 'Alexandre Bovet'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "autoapi.extension",
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'insegel'
html_logo = "_static/"
html_static_path = ['_static']


# -- Sphinx AutoAPI config ---------------------------------------------------

autoapi_dirs = ["../src/", ]
autoapi_file_patterns = ['*.py', ]
autoapi_member_order = "groupwise"
autoclass_content = "both"  # use docstring of both class and its __init__
# autoapi_ignore = ["*conf.py", "*setup.py" , "*_cython*.pyx", ]
