from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
  name = 'CYTHON FUNC',
  ext_modules = cythonize("convc.pyx"),
  include_dirs=[numpy.get_include()]
)