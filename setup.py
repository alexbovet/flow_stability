"""
#
# flow stability
#
# Copyright (C) 2021 Alexandre Bovet <alexandre.bovet@maths.ox.ac.uk>
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation; either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.


"""

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

import Cython.Compiler.Options
Cython.Compiler.Options.annotate = False

extensions = [
    Extension("_cython_fast_funcs", ["_cython_fast_funcs.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-O3","-fopenmp"],
        extra_link_args=['-fopenmp']),
    Extension("_cython_sparse_stoch", ["_cython_sparse_stoch.pyx",
                                     "SPA.cpp"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-O3","-fopenmp"],
        extra_link_args=['-fopenmp'])]

setup(
    ext_modules = cythonize(extensions,
                      language_level=3,
                      annotate=False),
)

