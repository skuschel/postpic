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
# Stephan Kuschel 2017
# Alexander Blinne, 2017
"""
This module provides compatibility replacements of functions from external
libraries which have changed w.r.t. older versions of these libraries or
were not present in older versions of these libraries
"""

import numpy as np
import scipy as sp
import scipy.signal as sps
import collections


def np_meshgrid(*args, **kwargs):
    if len(args) == 0:
        return tuple()

    if len(args) == 1:
        if kwargs.get('copy', False):
            return (args[0].copy(),)
        return (args[0].view(),)

    return np.meshgrid(*args, **kwargs)


def np_broadcast_to(*args, **kwargs):
    array, shape = args
    a, b = np.broadcast_arrays(array, np.empty(shape), **kwargs)
    return a


def np_moveaxis(*args, **kwargs):
    a, source, destination = args

    # twice a quick implementation of numpy.numeric.normalize_axis_tuple
    if not isinstance(source, collections.Iterable):
        source = (source,)
    if not isinstance(destination, collections.Iterable):
        destination = (destination,)
    source = [s % a.ndim for s in source]
    destination = [d % a.ndim for d in destination]

    # the real work copied from np.moveaxis
    order = [n for n in range(a.ndim) if n not in source]
    for dest, src in sorted(zip(destination, source)):
        order.insert(dest, src)

    return np.transpose(a, order)


def sps_tukey(M, alpha=0.5, sym=True):
    """
    Copied from scipy commit 870abd2f1fcc1fcf491324cdf5f78b4310c84446
    and replaced some functions by their implementation
    """
    if int(M) != M or M < 0:
        raise ValueError('Window length M must be a non-negative integer')
    if M <= 1:
        return np.ones(M)

    if alpha <= 0:
        return np.ones(M, 'd')
    elif alpha >= 1.0:
        return hann(M, sym=sym)

    if not sym:
        M, needs_trunc = M + 1, True
    else:
        M, needs_trunc = M, False

    n = np.arange(0, M)
    width = int(np.floor(alpha*(M-1)/2.0))
    n1 = n[0:width+1]
    n2 = n[width+1:M-width-1]
    n3 = n[M-width-1:]

    w1 = 0.5 * (1 + np.cos(np.pi * (-1 + 2.0*n1/alpha/(M-1))))
    w2 = np.ones(n2.shape)
    w3 = 0.5 * (1 + np.cos(np.pi * (-2.0/alpha + 1 + 2.0*n3/alpha/(M-1))))

    w = np.concatenate((w1, w2, w3))

    if needs_trunc:
        return w[:-1]
    else:
        return w


ReplacementFunction = collections.namedtuple('ReplacementFunction', ['name', 'originalmodule',
                                                                     'replacement', 'lib',
                                                                     'minver'])


replacements = [
    ReplacementFunction('meshgrid', np, np_meshgrid, np, '1.9'),
    ReplacementFunction('broadcast_to', np, np_broadcast_to, np, '1.10'),
    ReplacementFunction('moveaxis', np, np_moveaxis, np, '1.11'),
    ReplacementFunction('tukey', sps, sps_tukey, sp, '0.16')
]
