"""Configuration Sphinx de sephora_test."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

project = "sephora_test"
author = "Mounsef Debache"
copyright = "2026, Mounsef Debache"
release = "0.1.0"
language = "fr"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "api/modules.rst",
    "api/sephora_test*.rst",
]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

autodoc_member_order = "bysource"
autoclass_content = "both"

napoleon_google_docstring = True
napoleon_numpy_docstring = True

html_theme = "furo"
html_title = "sephora_test — Documentation"


def skip_metadata_methods(app, what, name, obj, skip, options):
    """Écarter les méthodes de metadata routing générées par sklearn."""
    if name in {
        "set_fit_request",
        "set_score_request",
        "set_transform_request",
        "set_partial_fit_request",
        "set_inverse_transform_request",
    }:
        return True
    return None


def setup(app):
    app.connect("autodoc-skip-member", skip_metadata_methods)
