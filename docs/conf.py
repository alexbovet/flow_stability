# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import pathlib
import sys

# -- Add the project root for autodiscovery ----------
sys.path.insert(0, pathlib.Path(__file__).parents[1].resolve().as_posix())

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
autoapi_python_class_content = "both"
# autoapi_ignore = ["*conf.py", "*setup.py" , "*_cython*.pyx", ]
