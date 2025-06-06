"""
/*
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
*/
"""


from libcpp.vector cimport vector

cdef extern from "sparse_accumulator.cpp":
    pass

# Declare the class with cdef
cdef extern from "sparse_accumulator.h":
    cdef cppclass SPA:
        SPA() except + 
        SPA(int size) except +
        int size
        int current_row
        vector[double] w
        vector[int] b
        vector[int] LS
        void scatter(double value, int pos)
        void reset(int current_row)
