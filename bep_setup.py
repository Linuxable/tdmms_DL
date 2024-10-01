"""
Setup the BEP AI code

Creates an .so file of the _mask.pyx file for pycocotools.

Run:
python bep_setup.py build_ext --inplace
"""

from distutils.core import setup
from Cython.Build import cythonize

import numpy
import os

files_to_setup = {
    'pycocotools': ['_mask.pyx'],
    'skimage': [
        'transform\\_hough_transform.pyx',
        'transform\\_radon_transform.pyx',
        'transform\\_warps_cy.pyx',
        'measure\\_find_contours_cy.pyx',
    ]
}

setup(
    ext_modules = cythonize("add_num.pyx"),
    include_dirs=[numpy.get_include()]
)