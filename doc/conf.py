# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sphinx configuration."""

import datetime
import os
import shutil
import sys

sys.path.insert(0, os.path.abspath("../src/"))

import renate  # noqa: E402


def run_apidoc(app):
    """Generate doc stubs using sphinx-apidoc."""
    module_dir = os.path.join(app.srcdir, "../src/")
    output_dir = os.path.join(app.srcdir, "_apidoc")
    excludes = []

    # Ensure that any stale apidoc files are cleaned up first.
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    cmd = [
        "--separate",
        "--module-first",
        "--doc-project=API Reference",
        "-o",
        output_dir,
        module_dir,
    ]
    cmd.extend(excludes)

    try:
        from sphinx.ext import apidoc  # Sphinx >= 1.7

        apidoc.main(cmd)
    except ImportError:
        from sphinx import apidoc  # Sphinx < 1.7

        cmd.insert(0, apidoc.__file__)
        apidoc.main(cmd)


def setup(app):
    """Register our sphinx-apidoc hook."""
    app.connect("builder-inited", run_apidoc)


# Sphinx configuration below.
project = "Renate"
version = renate.__version__
release = renate.__version__
copyright = f"2022-{datetime.datetime.now().year}, Amazon"


extensions = [
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx_autodoc_typehints",  # typing shown in api reference
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.coverage",
    "hoverxref.extension",  # show reference preview
    "sphinx_copybutton",
    "sphinxext.opengraph",
    "sphinx_paramlinks",  # lightning uses it, we get warnings otherwise
]
coverage_show_missing_items = True

# Make sure the target is unique
autosectionlabel_prefix_document = True

hoverxref_auto_ref = True  # Enable preview for all refs
hoverxref_role_types = {"ref": "tooltip"}

source_suffix = [".rst", ".md"]

master_doc = "index"

autoclass_content = "class"
autodoc_member_order = "bysource"
default_role = "py:obj"

html_theme = "pydata_sphinx_theme"
html_sidebars = {"**": ["sidebar-nav-bs"]}
html_theme_options = {
    "primary_sidebar_end": [],
    "footer_start": ["copyright"],
    "footer_end": [],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/awslabs/Renate",  # required
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        }
    ],
    "use_edit_page_button": True,
    "collapse_navigation": True,
}
html_context = {
    "github_user": "awslabs",
    "github_repo": "Renate",
    "github_version": "dev",
    "doc_path": "doc",
    "default_mode": "light",
}

htmlhelp_basename = "{}doc".format(project)

napoleon_use_rtype = False

rst_prolog = """
.. role:: python(code)
    :language: python
    :class: highlight
"""
