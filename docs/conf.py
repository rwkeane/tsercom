import os
import sys

sys.path.insert(0, os.path.abspath(".."))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "tsercom"
copyright = "2025, tsercom"
author = "tsercom"
release = "0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_rtd_theme",
    "sphinx_multiversion",
]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**/*_unittest.py",
    "**/proto/generated/**",
]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# Sphinx-multiversion settings
smv_tag_whitelist = r"^v\d+\.\d+(\.\d+)?$"  # Tags like v1.0, v1.0.1
smv_branch_whitelist = r"^(main|develop)$"  # Branches main and develop
smv_latest_version = "main"  # 'main' will be /latest/
smv_remote_whitelist = r"^origin$"  # Only use the 'origin' remote
smv_prefer_remote_refs = True  # Use remote refs if available
smv_outputdir_format = "{ref.name}"  # Output dirs like /main/ or /v1.0/
# smv_show_banner = True # Optional: show a banner for non-release versions
