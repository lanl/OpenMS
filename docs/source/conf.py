# Configuration file for the Sphinx documentation builder.

# -- Project information
import os
import sys

project = 'OpenMS'
copyright = 'Copyright 2024. Triad National Security, LLC. All rights reserved.'
author = 'Yu Zhang'
sys.path.insert(0, os.path.abspath('../../openms'))

# import module seems not working in readthedoc build
#import openms
release = '0.2.0'
version = "0.2.0" #openms.__version__

# -- General configuration

extensions = [
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.githubpages',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    #'sphinx.ext.duration',
    #'sphinx.ext.doctest',
    'sphinxcontrib.bibtex',
]

bibtex_bibfiles = ['refs.bib']

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'scipy': ('https://docs.scipy.org/doc/scipy', None),
    'pyscf': ('https://pyscf.org', None),
}
#intersphinx_mapping = {
#    'matplotlib': ('https://matplotlib.org/stable', None),
#    'numpy': ('https://numpy.org/doc/stable', None),
#    'numpy-tutorials': ('https://numpy.org/numpy-tutorials', None),
#    'numpydoc': ('https://numpydoc.readthedocs.io/en/latest', None),
#    'python': ('https://docs.python.org/3/', None),
#    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
#}
intersphinx_disabled_reftypes = ['std']

napoleon_preprocess_types = True

# MathJax customization
mathjax3_config = {

    'loader': {'load': ['tex-chtml',
                        '[tex]/upgreek',
                        '[tex]/physics',
                        '[tex]/boldsymbol',
                        '[tex]/html',
                        '[tex]/mathtools',
                        '[tex]/mhchem']
    },

    'tex': {'packages': {'[+]': ['upgreek',
                                 'physics',
                                 'boldsymbol',
                                 'html',
                                 'mathtools',
                                 'mhchem']
            },

            'mathtools': {'multlinegap': '1em',
                          'multlined-pos': 'c',
                          'firstline-afterskip': '',
                          'lastline-preskip': '',
                          'smallmatrix-align': 'c',
                          'shortvdotsadjustabove': '.2em',
                          'shortvdotsadjustbelow': '.2em',
                          'centercolon': False,
                          'centercolon-offset': '.04em',
                          'thincolon-dx': '-.04em',
                          'thincolon-dw': '-.08em',
                          'use-unicode': False,
                          'prescript-sub-format': '',
                          'prescript-sup-format': '',
                          'prescript-arg-format': '',
                          'allow-mathtoolsset': True,
                          'pairedDelimiters': {},
                          'tagforms': {}
            },
    }
}

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# The suffix of source filenames.
source_suffix = {'.rst': 'restructuredtext'}

# The master toctree document.
master_doc = 'index'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# Latex files
latex_documents = [
  ('index', 'OpenMS.tex', u'OpenMS Documentation',
   u'Yu Zhang \\textless{}zhy@lanl.gov\\textgreater{}', 'manual'),
  # manual is the document class, must be at the end. otherwise it will cause problems
  # when 'make latexpdf'
]

latex_elements = {
    'preamble': r'''
\usepackage{braket}
\usepackage{amsmath}
\usepackage{amssymb}
'''
}

# -- Options for HTML output
html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'
