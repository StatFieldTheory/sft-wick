# Configuration file for the Sphinx documentation builder.

from importlib.metadata import version as get_version
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# -- Project information -----------------------------------------------------

project = "sft-wick"
copyright = "2026-present, Zheng Zhang"
author = "Zheng Zhang"

try:
    release = get_version("sft-wick")
except Exception:
    release = "0.1.0"
version = release

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx_copybutton",
    "nbsphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "**.ipynb_checkpoints"]

# -- Options for autodoc -----------------------------------------------------

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

# -- Options for Napoleon (Google-style docstrings) --------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_ivar = True
napoleon_preprocess_types = True

# -- Options for nbsphinx ---------------------------------------------------

nbsphinx_execute = "never"

# -- Options for intersphinx -------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "networkx": ("https://networkx.org/documentation/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# -- Options for HTML output -------------------------------------------------

html_theme = "furo"
html_title = "sft-wick"
html_static_path = ["_static"]

html_theme_options = {
    "source_repository": "https://github.com/StatFieldTheory/sft-wick",
    "source_branch": "main",
    "source_directory": "docs/",
}

# -- MathJax configuration --------------------------------------------------

mathjax3_config = {
    "tex": {
        "macros": {
            "braket": [r"\langle #1 \rangle", 1],
            "sint": r"S_{\mathrm{int}}",
        },
    },
}
