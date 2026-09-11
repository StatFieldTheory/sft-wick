# Configuration file for the Sphinx documentation builder.

from importlib.metadata import PackageNotFoundError, version as get_version
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

# -- Project information -----------------------------------------------------

project = "sft-wick"
copyright = "2026-present, Zheng Zhang"
author = "Zheng Zhang"


def _release() -> str:
    """Version of the tree autodoc is about to read.

    ``sys.path`` above points autodoc at ``src/``, so the stamped version
    has to come from the same tree.  Reading it from installed metadata
    instead lets the two diverge: an editable install whose metadata has
    gone stale labels the build with whatever release it was installed at.
    Installed metadata is the fallback, for a build run from outside a
    checkout.
    """
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10
        pass
    else:
        try:
            with open(_ROOT / "pyproject.toml", "rb") as fh:
                return tomllib.load(fh)["project"]["version"]
        except (OSError, tomllib.TOMLDecodeError, KeyError):
            pass
    try:
        return get_version("sft-wick")
    except PackageNotFoundError:
        return "0.0.0+unknown"


release = _release()
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
