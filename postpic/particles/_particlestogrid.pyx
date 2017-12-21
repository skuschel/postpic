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
# Copyright Stephan Kuschel 2015-2017
'''
This file adds the particle-to-grid routines, implemented in cython.

Do not call these functions directly. Use `postpic.particles.histogramdd` instead.
'''
from __future__ import absolute_import, division, print_function, unicode_literals
cimport cython

import numpy as np
cimport numpy as np

shapes = [
    [0, 'NGP'],
    [1, 'tophat'],
    [2, 'triangle'],
    [3, 'spline3']
]

@cython.boundscheck(False)  # disable array boundscheck
@cython.wraparound(False)  # disable negative array indices
def histogram(np.ndarray[np.double_t, ndim=1] data, range=None, int bins=20,
           np.ndarray[np.double_t, ndim=1] weights=None, shape=0):
    '''
    Never use directly. Use `postpic.particles.histogramdd` instead.

    Mimics numpy.histogram.
    Additional Arguments:
        - shape = 0:
            sets the order of the particle shapes.
            shape = 0 returns a normal histogram.
            shape = 1 uses top hat particle shape.
            shape = 2 uses triangle particle shape.
    '''
    cdef double xmin, xmax
    xmin, xmax = range
    # ensure max != min
    sx = np.spacing(xmax)
    if np.abs(xmax-xmin) < sx:
        xmax += sx
        xmin -= sx
    bin_edges = np.linspace(xmin, xmax, bins+1)
    cdef int n = len(data)
    cdef double dx = 1.0 / (xmax - xmin) * bins  # actually: 1/dx
    cdef np.ndarray[np.double_t, ndim=1] ret
    cdef int shape_supp
    cdef double x
    cdef int xr
    cdef double xd
    if shape in shapes[0]:
        # normal Histogram
        shape_supp = 0
        ret = np.zeros(bins, dtype=np.double)
        for i in xrange(n):
            x = (data[i] - xmin) * dx;
            if x > 0.0 and x < bins:
                if weights is None:
                    ret[<int>x] += 1.0
                else:
                    ret[<int>x] += weights[i]
    elif shape in shapes[1]:
        # Particle shape is spline of order 1 = TopHat
        shape_supp = 1
        # use shape_supp ghost cells on both sides of the domain
        ret = np.zeros(bins + 2 * shape_supp, dtype=np.double)
        for i in xrange(n):
            x = (data[i] - xmin) * dx;
            xr = <int>(x + 0.5);
            if (xr >= 0.0 and xr <= bins):
                if weights is None:
                    ret[xr + shape_supp]     += (0.5 + x - xr) * 1.0
                    ret[xr + shape_supp - 1] += (0.5 - x + xr) * 1.0
                else:
                    ret[xr + shape_supp]     += (0.5 + x - xr) * weights[i]
                    ret[xr + shape_supp - 1] += (0.5 - x + xr) * weights[i]
    elif shape in shapes[2]:
        # Particle shape is spline of order 2 = Triangle
        shape_supp = 2
        # use shape_supp ghost cells on both sides of the domain
        ret = np.zeros(bins + 2 * shape_supp, dtype=np.double)
        for i in xrange(n):
            x = (data[i] - xmin) * dx;
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
    elif shape in shapes[3]:
        # Particle shape is spline of order 3 = Spline3
        shape_supp = 3
        # use shape_supp ghost cells on both sides of the domain
        ret = np.zeros(bins + 2 * shape_supp, dtype=np.double)
        for i in xrange(n):
            x = (data[i] - xmin) * dx;
            xr = <int>(x + 0.5);
            xd = x - xr + 0.5
            if (xr >= 0.0 and xr <= bins):
                if weights is None:
                    ret[xr + shape_supp - 2] += 1./6. + xd*(-0.5 + (0.5 - xd/6.)*xd)
                    ret[xr + shape_supp - 1] += 2./3. + (-1 + xd/2.)*xd*xd
                    ret[xr + shape_supp]     += 1./6 + xd*(0.5 + (0.5 - xd/2.)*xd)
                    ret[xr + shape_supp + 1] += xd*xd*xd/6.0
                else:
                    ret[xr + shape_supp - 2] += weights[i] * (1./6. + xd*(-0.5 + (0.5 - xd/6.)*xd))
                    ret[xr + shape_supp - 1] += weights[i] * (2./3. + (-1 + xd/2.)*xd*xd)
                    ret[xr + shape_supp]     += weights[i] * (1./6 + xd*(0.5 + (0.5 - xd/2.)*xd))
                    ret[xr + shape_supp + 1] += weights[i] * (xd*xd*xd/6.0)
    return ret[shape_supp:shape_supp + bins], bin_edges


@cython.boundscheck(False)  # disable array boundscheck
@cython.wraparound(False)  # disable negative array indices
def histogram2d(np.ndarray[np.double_t, ndim=1] datax, np.ndarray[np.double_t, ndim=1] datay,
                np.ndarray[np.double_t, ndim=1] weights=None,
                range=None, bins=(20, 20), shape=0):
    '''
    Never use directly. Use `postpic.particles.histogramdd` instead.

    Mimics numpy.histogram2d.
    Additional Arguments:
        - shape = 0:
            sets the order of the particle shapes.
            shape = 0 returns a normal histogram.
            shape = 1 uses top hat particle shape.
            shape = 2 uses triangle particle shape.
    '''
    cdef int n = len(datax)
    if n != len(datay):
            raise ValueError('datax and datay must be of equal length')
    cdef double xmin, xmax, ymin, ymax
    xmin, xmax = range[0]
    ymin, ymax = range[1]
    # ensure max != min
    sx = np.spacing(xmax)
    if np.abs(xmax-xmin) < sx:
        xmax += sx
        xmin -= sx
    sy = np.spacing(ymax)
    if np.abs(ymax-ymin) < sy:
        ymax += sy
        ymin -= sy
    cdef int xbins = bins[0]
    cdef int ybins = bins[1]
    xedges = np.linspace(xmin, xmax, xbins+1)
    yedges = np.linspace(ymin, ymax, ybins+1)
    cdef double dx = 1.0 / (xmax - xmin) * xbins  # actually: 1/dx
    cdef double dy = 1.0 / (ymax - ymin) * ybins  # actually: 1/dy
    cdef np.ndarray[np.double_t, ndim=2] ret
    cdef int shape_supp, xs, ys, xoffset, yoffset
    cdef double x, y, xd, yd
    cdef int xr, yr
    cdef double wx[4]
    cdef double wy[4]

    if shape in shapes[0]:
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
    elif shape in shapes[1]:
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
    elif shape in shapes[2]:
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
    elif shape in shapes[3]:
        # Particle shape is spline of order 3
        shape_supp = 3
        # use shape_supp ghost cells on both sides of the domain
        resshape = [b + 2 * shape_supp for b in bins]
        ret = np.zeros(resshape, dtype=np.double)
        for i in xrange(n):
            x = (datax[i] - xmin) * dx;
            y = (datay[i] - ymin) * dy;
            xr = <int>(x + 0.5);
            yr = <int>(y + 0.5);
            xd = x - xr + 0.5;
            yd = y - yr + 0.5;
            if (xr >= 0 and y >= 0 and xr <= xbins and yr <= ybins):
                wx[0] = 1./6. + xd*(-0.5 + (0.5 - xd/6.)*xd)
                wx[1] = 2./3. + (-1 + xd/2.)*xd*xd
                wx[2] = 1./6 + xd*(0.5 + (0.5 - xd/2.)*xd)
                wx[3] = xd*xd*xd/6.0
                wy[0] = 1./6. + yd*(-0.5 + (0.5 - yd/6.)*yd)
                wy[1] = 2./3. + (-1 + yd/2.)*yd*yd
                wy[2] = 1./6 + yd*(0.5 + (0.5 - yd/2.)*yd)
                wy[3] = yd*yd*yd/6.0
                xoffset = xr + shape_supp - 2
                yoffset = yr + shape_supp - 2
                if weights is None:
                    for xs in xrange(4):
                        for ys in xrange(4):
                            ret[xoffset + xs, yoffset + ys] += wx[xs] * wy[ys]
                else:
                    for xs in xrange(4):
                        for ys in xrange(4):
                            ret[xoffset + xs, yoffset + ys] += wx[xs] * wy[ys] * weights[i]

    return ret[shape_supp:shape_supp + xbins, shape_supp:shape_supp + ybins], xedges, yedges



@cython.boundscheck(False)  # disable array boundscheck
@cython.wraparound(False)  # disable negative array indices
def histogram3d(np.ndarray[np.double_t, ndim=1] datax, np.ndarray[np.double_t, ndim=1] datay,
                np.ndarray[np.double_t, ndim=1] dataz,
                np.ndarray[np.double_t, ndim=1] weights=None,
                range=None, bins=(20, 20, 20), shape=0):
    '''
    Never use directly. Use `postpic.particles.histogramdd` instead.

    Additional Arguments:
        - shape = 0:
            sets the order of the particle shapes.
            shape = 0 returns a normal histogram.
            shape = 1 uses top hat particle shape.
            shape = 2 uses triangle particle shape.
    '''
    cdef int n = len(datax)
    if n != len(datay):
            raise ValueError('datax and datay must be of equal length')
    if n != len(dataz):
            raise ValueError('datax and dataz must be of equal length')
    cdef double xmin, xmax, ymin, ymax, zmin, zmax
    xmin, xmax = range[0]
    ymin, ymax = range[1]
    zmin, zmax = range[2]
    # ensure max != min
    sx = np.spacing(xmax)
    if np.abs(xmax-xmin) < sx:
        xmax += sx
        xmin -= sx
    sy = np.spacing(ymax)
    if np.abs(ymax-ymin) < sy:
        ymax += sy
        ymin -= sy
    sz = np.spacing(zmax)
    if np.abs(zmax-zmin) < sz:
        zmax += sz
        zmin -= sz
    cdef int xbins = bins[0]
    cdef int ybins = bins[1]
    cdef int zbins = bins[2]
    xedges = np.linspace(xmin, xmax, xbins+1)
    yedges = np.linspace(ymin, ymax, ybins+1)
    zedges = np.linspace(zmin, zmax, zbins+1)
    cdef double dx = 1.0 / (xmax - xmin) * xbins  # actually: 1/dx
    cdef double dy = 1.0 / (ymax - ymin) * ybins  # actually: 1/dy
    cdef double dz = 1.0 / (zmax - zmin) * zbins  # actually: 1/dz
    cdef np.ndarray[np.double_t, ndim=3] ret
    cdef int shape_supp, xs, ys, zs, xoffset, yoffset, zoffset
    cdef double x, y, z, xd, yd, zd
    cdef int xr, yr, zr
    cdef double wx[4]
    cdef double wy[4]
    cdef double wz[4]

    if shape in shapes[0]:
        # normal Histogram
        shape_supp = 0
        ret = np.zeros(bins, dtype=np.double)
        for i in xrange(n):
            x = (datax[i] - xmin) * dx
            y = (datay[i] - ymin) * dy
            z = (dataz[i] - zmin) * dz
            if x > 0.0 and y > 0.0 and z > 0.0 and x < xbins and y < ybins and z < zbins:
                if weights is None:
                    ret[<int>x, <int>y, <int>z] += 1.0
                else:
                    ret[<int>x, <int>y, <int>z] += weights[i]
    elif shape in shapes[1]:
        # Particle shape is spline of order 1 = TopHat
        shape_supp = 1
        # use shape_supp ghost cells on both sides of the domain
        resshape = [b + 2 * shape_supp for b in bins]
        ret = np.zeros(resshape, dtype=np.double)
        for i in xrange(n):
            x = (datax[i] - xmin) * dx;
            y = (datay[i] - ymin) * dy;
            z = (dataz[i] - zmin) * dz;
            xr = <int>(x + 0.5);
            yr = <int>(y + 0.5);
            zr = <int>(z + 0.5);
            if (xr >= 0 and y >= 0 and xr <= xbins and yr <= ybins and z >= 0 and zr <= zbins):
                wx[0] = (0.5 - x + xr)
                wx[1] = (0.5 + x - xr)
                wy[0] = (0.5 - y + yr)
                wy[1] = (0.5 + y - yr)
                wz[0] = (0.5 - z + zr)
                wz[1] = (0.5 + z - zr)
                xoffset = xr + shape_supp - 1
                yoffset = yr + shape_supp - 1
                zoffset = zr + shape_supp - 1
                if weights is None:
                    for xs in xrange(2):
                        for ys in xrange(2):
                            for zs in xrange(2):
                                ret[xoffset+xs, yoffset+ys, zoffset+zs] += wx[xs] * wy[ys] * wz[zs]
                else:
                    for xs in xrange(2):
                        for ys in xrange(2):
                            for zs in xrange(2):
                                ret[xoffset+xs, yoffset+ys, zoffset+zs] += wx[xs] * wy[ys] * wz[zs] * weights[i]
    elif shape in shapes[2]:
        # Particle shape is spline of order 2 = Triangle
        shape_supp = 2
        # use shape_supp ghost cells on both sides of the domain
        resshape = [b + 2 * shape_supp for b in bins]
        ret = np.zeros(resshape, dtype=np.double)
        for i in xrange(n):
            x = (datax[i] - xmin) * dx;
            y = (datay[i] - ymin) * dy;
            z = (dataz[i] - zmin) * dz;
            xr = <int>x;
            yr = <int>y;
            zr = <int>z;
            xd = x - xr;
            yd = y - yr;
            zd = z - zr;
            if (xr >= 0 and y >= 0 and zr >= 0 and xr <= xbins and yr <= ybins and zr <= zbins):
                wx[0] = 0.5 * (1 - xd)**2
                wx[1] = 0.5 + xd - xd**2
                wx[2] = 0.5 * xd**2
                wy[0] = 0.5 * (1 - yd)**2
                wy[1] = 0.5 + yd - yd**2
                wy[2] = 0.5 * yd**2
                wz[0] = 0.5 * (1 - zd)**2
                wz[1] = 0.5 + zd - zd**2
                wz[2] = 0.5 * zd**2
                xoffset = xr + shape_supp - 1
                yoffset = yr + shape_supp - 1
                zoffset = zr + shape_supp - 1
                if weights is None:
                    for xs in xrange(3):
                        for ys in xrange(3):
                            for zs in xrange(3):
                                ret[xoffset+xs, yoffset+ys, zoffset+zs] += wx[xs] * wy[ys] * wz[zs]
                else:
                    for xs in xrange(3):
                        for ys in xrange(3):
                            for zs in xrange(3):
                                ret[xoffset+xs, yoffset+ys, zoffset+zs] += wx[xs] * wy[ys] * wz[zs] * weights[i]
    elif shape in shapes[3]:
        # Particle shape is spline of order 3 = spine3
        shape_supp = 3
        # use shape_supp ghost cells on both sides of the domain
        resshape = [b + 2 * shape_supp for b in bins]
        ret = np.zeros(resshape, dtype=np.double)
        for i in xrange(n):
            x = (datax[i] - xmin) * dx;
            y = (datay[i] - ymin) * dy;
            z = (dataz[i] - zmin) * dz;
            xr = <int>(x + 0.5);
            yr = <int>(y + 0.5);
            zr = <int>(z + 0.5);
            xd = x - xr + 0.5;
            yd = y - yr + 0.5;
            zd = z - zr + 0.5;
            if (xr >= 0 and y >= 0 and zr >= 0 and xr <= xbins and yr <= ybins and zr <= zbins):
                wx[0] = 1./6. + xd*(-0.5 + (0.5 - xd/6.)*xd)
                wx[1] = 2./3. + (-1 + xd/2.)*xd*xd
                wx[2] = 1./6 + xd*(0.5 + (0.5 - xd/2.)*xd)
                wx[3] = xd*xd*xd/6.0
                wy[0] = 1./6. + yd*(-0.5 + (0.5 - yd/6.)*yd)
                wy[1] = 2./3. + (-1 + yd/2.)*yd*yd
                wy[2] = 1./6 + yd*(0.5 + (0.5 - yd/2.)*yd)
                wy[3] = yd*yd*yd/6.0
                wz[0] = 1./6. + zd*(-0.5 + (0.5 - zd/6.)*zd)
                wz[1] = 2./3. + (-1 + zd/2.)*zd*zd
                wz[2] = 1./6 + zd*(0.5 + (0.5 - zd/2.)*zd)
                wz[3] = zd*zd*zd/6.0
                xoffset = xr + shape_supp - 2
                yoffset = yr + shape_supp - 2
                zoffset = zr + shape_supp - 2
                if weights is None:
                    for xs in xrange(4):
                        for ys in xrange(4):
                            for zs in xrange(4):
                                ret[xoffset+xs, yoffset+ys, zoffset+zs] += wx[xs] * wy[ys] * wz[zs]
                else:
                    for xs in xrange(4):
                        for ys in xrange(4):
                            for zs in xrange(4):
                                ret[xoffset+xs, yoffset+ys, zoffset+zs] += wx[xs] * wy[ys] * wz[zs] * weights[i]

    return ret[shape_supp:shape_supp+xbins, shape_supp:shape_supp+ybins, shape_supp:shape_supp+zbins], xedges, yedges, zedges
