#
# This file is part of postpic.
#
# postpic is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# postpic is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with postpic. If not, see <http://www.gnu.org/licenses/>.
#
# Copyright Stephan Kuschel 2015
'''
This file adds some has some runtime critical funtions implemented in cython.
'''
cimport cython

import numpy as np
cimport numpy as np


def histogram(np.ndarray[np.double_t, ndim=1] data, range=None, int bins=20,
           np.ndarray[np.double_t, ndim=1] weights=None, int order=0):
    '''
    Mimics numpy.histogram.
    Additional Arguments:
        - order = 0:
            sets the order of the particle shapes.
            order = 0 returns a normal histogram.
            order = 1 uses top hat particle shape.
    '''    
    cdef np.ndarray[np.double_t, ndim=1] ret = np.zeros(bins, dtype=np.double)
    cdef double min, max
    if range is None:
        min = np.min(data)
        max = np.max(data)
    else:
        min = range[0]
        max = range[1]
    bin_edges = np.linspace(min, max, bins+1)
    cdef int n = len(data)
    cdef double tmp = 1.0 / (max - min) * bins
    cdef double x
    cdef int xr
    if order == 0:
        # normal Histogram
        for i in xrange(n):
            x = (data[i] - min) * tmp;
            if x > 0.0 and x < bins:
                if weights is None:
                    ret[<int>x] += 1.0
                else:
                    ret[<int>x] += weights[i]
    elif order == 1:
        # Particle shape is spline of order 1 = TopHat
        for i in xrange(n):
            x = (data[i] - min) * tmp;
            xr = <int>(x + 0.5);
            if (xr >= 0.0 and xr < bins):
                if weights is None:
                    ret[xr] += (0.5 + x - xr) * 1.0
                    if (xr > 1.0):
                        ret[xr - 1] += (0.5 - x + xr) * 1.0
                else:
                    ret[xr] += (0.5 + x - xr) * weights[i]
                    if (xr > 1.0):
                        ret[xr - 1] += (0.5 - x + xr) * weights[i]
    return ret, bin_edges
