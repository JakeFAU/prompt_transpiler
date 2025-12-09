import os
import sys

# 1. Path Setup: Point to the 'src' root so imports work
sys.path.insert(0, os.path.abspath("../src"))

project = "prompt-compiler"  # Fixed typo (was complier)
copyright = "2025, Jacob Bourne"
author = "Jacob Bourne"
release = "0.1.0"

# 2. Modern Extensions
extensions = [
    "sphinx.ext.autodoc",  # Core library
    "sphinx.ext.napoleon",  # Understands Google/NumPy style docstrings
    "sphinx.ext.viewcode",  # Links to source code
    "sphinx.ext.intersphinx",  # Links to Python stdlib docs (e.g. lists, dicts)
    "myst_parser",  # <--- ALLOWS MARKDOWN (.md) FILES
    "sphinx_copybutton",  # Adds 'copy' button to code blocks
]

# 3. Napoleon Settings (Make docstrings readable)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True  # Document __init__ if it has a docstring

# 4. The Theme (Switching to Furo or Sphinx-Book-Theme)
# 'sphinx_rtd_theme' is okay, but 'furo' is cleaner and darker-mode friendly.
html_theme = "furo"
html_title = "Prompt Compiler Physics"

# 5. Mapping for Intersphinx (Auto-link to external docs)
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

html_static_path = ["_static"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
