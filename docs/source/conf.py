"""Configuration for the Sphinx documentation builder."""

import sys
from pathlib import Path

# -- Package import path -----------------------------------------------------

config_directory = Path(__file__).resolve().parent

for directory in (config_directory, *config_directory.parents):
    source_directory = directory / "src"
    if (source_directory / "sephora_test").is_dir():
        sys.path.insert(0, str(source_directory))
        break
else:
    raise RuntimeError("Cannot find src/sephora_test.")

# -- Project information -----------------------------------------------------

project = "sephora_test"
copyright = "2026, debache"
author = "debache"
release = "0.1"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
]

# -- API documentation -------------------------------------------------------

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

autodoc_member_order = "bysource"
autoclass_content = "both"

napoleon_google_docstring = True
napoleon_numpy_docstring = False

# -- HTML output -------------------------------------------------------------

html_theme = "furo"
html_static_path = ["_static"]