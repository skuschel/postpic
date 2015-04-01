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


@cython.boundscheck(False)  # disable array boundscheck
@cython.wraparound(False)  # disable negative array indices
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
    cdef double min, max
    if range is None:
        min = np.min(data)
        max = np.max(data)
    else:
        min = range[0]
        max = range[1]
    bin_edges = np.linspace(min, max, bins+1)
    cdef int n = len(data)
    cdef double dx = 1.0 / (max - min) * bins
    cdef np.ndarray[np.double_t, ndim=1] ret
    cdef int shape_supp
    cdef double x
    cdef int xr
    cdef double xd
    if order == 0:
        # normal Histogram
        shape_supp = 0
        ret = np.zeros(bins, dtype=np.double)
        for i in xrange(n):
            x = (data[i] - min) * dx;
            if x > 0.0 and x < bins:
                if weights is None:
                    ret[<int>x] += 1.0
                else:
                    ret[<int>x] += weights[i]
    elif order == 1:
        # Particle shape is spline of order 1 = TopHat
        shape_supp = 1
        # use shape_supp ghost cells on both sides of the domain
        ret = np.zeros(bins + 2 * shape_supp, dtype=np.double)
        for i in xrange(n):
            x = (data[i] - min) * dx;
            xr = <int>(x + 0.5);
            if (xr >= 0.0 and xr <= bins):
                if weights is None:
                    ret[xr + shape_supp]     += (0.5 + x - xr) * 1.0
                    ret[xr + shape_supp - 1] += (0.5 - x + xr) * 1.0
                else:
                    ret[xr + shape_supp]     += (0.5 + x - xr) * weights[i]
                    ret[xr + shape_supp - 1] += (0.5 - x + xr) * weights[i]
    elif order == 2:
        # Particle shape is spline of order 2 = Triangle
        shape_supp = 2
        # use shape_supp ghost cells on both sides of the domain
        ret = np.zeros(bins + 2 * shape_supp, dtype=np.double)
        for i in xrange(n):
            x = (data[i] - min) * dx;
            xr = <int>x;
            xd = x - xr
            if (xr >= 0.0 and xr <= bins):
                if weights is None:
                    ret[xr + shape_supp - 1] += 0.5 * (1 - xd)**2
                    ret[xr + shape_supp]     += 0.5 + xd - xd**2
                    ret[xr + shape_supp + 1] += 0.5 * xd**2
                else:
                    ret[xr + shape_supp - 1] += (0.5 * (1 - xd)**2) * weights[i]
                    ret[xr + shape_supp]     += (0.5 + xd - xd**2) * weights[i]
                    ret[xr + shape_supp + 1] += (0.5 * xd**2) * weights[i]
    return ret[shape_supp:shape_supp + bins], bin_edges


@cython.boundscheck(False)  # disable array boundscheck
@cython.wraparound(False)  # disable negative array indices
def histogram2d(np.ndarray[np.double_t, ndim=1] datax, np.ndarray[np.double_t, ndim=1] datay,
                np.ndarray[np.double_t, ndim=1] weights=None,
                range=None, bins=(20, 20), int order=0):
    '''
    Mimics numpy.histogram2d.
    Additional Arguments:
        - order = 0:
            sets the order of the particle shapes.
            order = 0 returns a normal histogram.
            order = 1 uses top hat particle shape.
    '''
    cdef int n = len(datax)
    if n != len(datay):
            raise ValueError('datax and datay must be of equal length')
    cdef double xmin, xmax, ymin, ymax
    if range is None:
        xmin = np.min(datax)
        xmax = np.max(datax)
        ymin = np.min(datay)
        ymax = np.max(datay)
    else:
        xmin = range[0][0]
        xmax = range[0][1]
        ymin = range[1][0]
        ymax = range[1][1]
    cdef int xbins = bins[0]
    cdef int ybins = bins[1]
    xedges = np.linspace(xmin, xmax, xbins+1)
    yedges = np.linspace(ymin, ymax, ybins+1)
    cdef double dx = 1.0 / (xmax - xmin) * xbins
    cdef double dy = 1.0 / (ymax - ymin) * ybins
    cdef np.ndarray[np.double_t, ndim=2] ret
    cdef int shape_supp, xs, ys, xoffset, yoffset
    cdef double x, y, xd, yd
    cdef int xr, yr
    cdef double wx[3]
    cdef double wy[3]

    if order == 0:
        # normal Histogram
        shape_supp = 0
        ret = np.zeros(bins, dtype=np.double)
        for i in xrange(n):
            x = (datax[i] - xmin) * dx
            y = (datay[i] - ymin) * dy
            if x > 0.0 and y > 0.0 and x < xbins and y < ybins:
                if weights is None:
                    ret[<int>x, <int>y] += 1.0
                else:
                    ret[<int>x, <int>y] += weights[i]
    elif order == 1:
        # Particle shape is spline of order 1 = TopHat
        shape_supp = 1
        # use shape_supp ghost cells on both sides of the domain
        resshape = [b + 2 * shape_supp for b in bins]
        ret = np.zeros(resshape, dtype=np.double)
        for i in xrange(n):
            x = (datax[i] - xmin) * dx;
            y = (datay[i] - ymin) * dy;
            xr = <int>(x + 0.5);
            yr = <int>(y + 0.5);
            if (xr >= 0 and y >= 0 and xr <= xbins and yr <= ybins):
                wx[0] = (0.5 - x + xr)
                wx[1] = (0.5 + x - xr)
                wy[0] = (0.5 - y + yr)
                wy[1] = (0.5 + y - yr)
                if weights is None:
                    ret[xr + shape_supp - 1, yr + shape_supp - 1] += wx[0] * wy[0]
                    ret[xr + shape_supp - 1, yr + shape_supp - 0] += wx[0] * wy[1]
                    ret[xr + shape_supp - 0, yr + shape_supp - 1] += wx[1] * wy[0]
                    ret[xr + shape_supp - 0, yr + shape_supp - 0] += wx[1] * wy[1]
                else:
                    ret[xr + shape_supp - 1, yr + shape_supp - 1] += wx[0] * wy[0] * weights[i]
                    ret[xr + shape_supp - 1, yr + shape_supp - 0] += wx[0] * wy[1] * weights[i]
                    ret[xr + shape_supp - 0, yr + shape_supp - 1] += wx[1] * wy[0] * weights[i]
                    ret[xr + shape_supp - 0, yr + shape_supp - 0] += wx[1] * wy[1] * weights[i]
    elif order == 2:
        # Particle shape is spline of order 2 = Triangle
        shape_supp = 2
        # use shape_supp ghost cells on both sides of the domain
        resshape = [b + 2 * shape_supp for b in bins]
        ret = np.zeros(resshape, dtype=np.double)
        for i in xrange(n):
            x = (datax[i] - xmin) * dx;
            y = (datay[i] - ymin) * dy;
            xr = <int>x;
            yr = <int>y;
            xd = x - xr;
            yd = y - yr;
            if (xr >= 0 and y >= 0 and xr <= xbins and yr <= ybins):
                wx[0] = 0.5 * (1 - xd)**2
                wx[1] = 0.5 + xd - xd**2
                wx[2] = 0.5 * xd**2
                wy[0] = 0.5 * (1 - yd)**2
                wy[1] = 0.5 + yd - yd**2
                wy[2] = 0.5 * yd**2
                xoffset = xr + shape_supp - 1
                yoffset = yr + shape_supp - 1
                if weights is None:
                    for xs in xrange(3):
                        for ys in xrange(3):
                            ret[xoffset + xs, yoffset + ys] += wx[xs] * wy[ys]
                else:
                    for xs in xrange(3):
                        for ys in xrange(3):
                            ret[xoffset + xs, yoffset + ys] += wx[xs] * wy[ys] * weights[i]

    return ret[shape_supp:shape_supp + xbins, shape_supp:shape_supp + ybins], xedges, yedges




