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
release = '0.1'
version = "0.1.beta" #openms.__version__

# -- General configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    #'sphinx.ext.duration',
    #'sphinx.ext.doctest',
    'sphinx.ext.autosummary',
    #'sphinx.ext.intersphinx',
]

intersphinx_mapping = {'http://docs.python.org/': None}

#intersphinx_mapping = {
#    'python': ('https://docs.python.org/3/', None),
#    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
#}
#intersphinx_disabled_domains = ['std']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# The suffix of source filenames.
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# Latex files
latex_documents = [
  ('index', 'OpenMS.tex', u'OpenMS Documentation',
   u'Yu Zhang \\textless{}zhy@lanl.gov\\textgreater{}', 'manual'),
]

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
#epub_show_urls = 'footnote'
