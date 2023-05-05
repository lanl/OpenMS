# Configuration file for the Sphinx documentation builder.

# -- Project information
import os
import sys

project = 'OpenMS'
copyright = 'Copyright 2023. Triad National Security, LLC. All rights reserved.'
author = 'Yu Zhang'
sys.path.insert(0, os.path.abspath('../../../OpenMS/'))
sys.path.insert(0, os.path.abspath('../../openms/'))

#import openms
#version = openms.__version__
release = '0.1'
version = '0.1.0'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']


# The suffix of source filenames.
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'
