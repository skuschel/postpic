#!/usr/bin/env python2
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
#

import timeit
import numpy as np
import postpic.cythonfunctions as cf

# --- 1D

def time1D(data, bins, weights, order, tn):
    t = timeit.Timer(lambda: cf.histogram(data, range=(0.001,0.999), bins=bins, weights=weights, order=order))
    tc = t.timeit(number=5)/5.0
    ws = '       ' if weights is None else 'weights'
    print '1D, {:d} order, {:s}: {:0.2e} sec -> factor {:5.2f} faster'.format(order, ws, tc, tn/tc)

bins = 1000
npart = 1e6
print '=== Histogram 1D bins: {:6d}, npart: {:.1e} ==='.format(bins, npart)
data = np.random.random(npart)
weights = np.random.random(npart)
# time numpy function
t = timeit.Timer(lambda: np.histogram(data, range=(0.001,0.999), bins=bins, weights=None))
tn = t.timeit(number=2)/2.0
t = timeit.Timer(lambda: np.histogram(data, range=(0.001,0.999), bins=bins, weights=weights))
tnw = t.timeit(number=2)/2.0
print 'numpy        : {:0.2e} sec'.format(tn)
print 'numpy weights: {:0.2e} sec'.format(tnw)
time1D(data, bins, None, 0, tn)
time1D(data, bins, weights, 0, tnw)
time1D(data, bins, None, 1, tn)
time1D(data, bins, weights, 1, tnw)

# --- 2D

def time2D(datax, datay, bins, weights, order, tn):
    t = timeit.Timer(lambda: cf.histogram2d(datax, datay, range=((0.01,0.99),(0.01,0.99)), bins=bins, weights=weights, order=order))
    tc = t.timeit(number=5)/5.0
    ws = '       ' if weights is None else 'weights'
    print '2D, {:d} order, {:s}: {:0.2e} sec -> factor {:5.2f} faster'.format(order, ws, tc, tn/tc)

bins = (1000,700)
npart = 2e6
print '=== Histogram 2D bins: {:6s}, npart: {:.1e} ==='.format(bins, npart)
datax = np.random.rand(npart)
datay = np.random.rand(npart)
weights = np.random.random(npart)
# time numpy function
t = timeit.Timer(lambda: np.histogram2d(datax, datay, range=((0.01,0.99),(0.01,0.99)), bins=bins, weights=None))
tn = t.timeit(number=1)/1.0
t = timeit.Timer(lambda: np.histogram2d(datax, datay, range=((0.01,0.99),(0.01,0.99)), bins=bins, weights=weights))
tnw = t.timeit(number=1)/1.0
print 'numpy        : {:0.2e} sec'.format(tn)
print 'numpy weights: {:0.2e} sec'.format(tnw)
time2D(datax,datay, bins, None, 0, tn)
time2D(datax, datay, bins, weights, 0, tnw)


