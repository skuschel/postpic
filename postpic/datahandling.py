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
# Stephan Kuschel, 2014-2018
# Alexander Blinne, 2017
"""
The Core module for final data handling.

This module provides classes for dealing with axes, grid as well as the Field
class -- the final output of the postpic postprocessor.

Terminology
-----------

A data field with N numeric points has N 'grid' points,
but N+1 'grid_nodes' as depicted here:

.. code-block:: none

  +---+---+---+---+---+
  |   |   |   |   |   |
  +---+---+---+---+---+
  |   |   |   |   |   |
  +---+---+---+---+---+
  |   |   |   |   |   |
  +---+---+---+---+---+
    o   o   o   o   o     grid      (coordinates where data is sampled at)
  o   o   o   o   o   o   grid_node (coordinates of grid cell boundaries)
  |                   |   extent

"""
from __future__ import absolute_import, division, print_function, unicode_literals

import sys

import collections
import copy
import warnings
import os
import numbers

import numpy as np
import scipy.ndimage as spnd
import scipy.interpolate as spinterp
import scipy.integrate
import scipy.signal as sps
import numexpr as ne

from ._compat import tukey, meshgrid, broadcast_to, NDArrayOperatorsMixin
from . import helper
from . import io

if sys.version[0] == '2':
    import functools32 as functools
else:
    import functools

if sys.version[0] == '2':
    from itertools import izip_longest as zip_longest
else:
    from itertools import zip_longest


try:
    import psutil
    nproc = psutil.cpu_count(logical=False)
except ImportError:
    try:
        nproc = os.cpu_count()
    except AttributeError:
        import multiprocessing
        nproc = multiprocessing.cpu_count()


try:
    # pyfftw is, in most situations, faster than numpys fft,
    # although pyfftw will benefit from multithreading only on very large arrays
    # on a 720x240x240 3D transform multithreading still doesn't give a large benefit
    # benchmarks of a 720x240x240 transform of real data on a Intel(R) Xeon(R) CPU
    # E5-1620 v4 @ 3.50GHz:
    # numpy.fft: 3.6 seconds
    # pyfftw, nproc=4: first transform 2.2s, further transforms 1.8s
    # pyfftw, nproc=1: first transform 3.4s, further transforms 2.8s
    # Try to import pyFFTW's numpy_fft interface
    import pyfftw.interfaces.cache as fftw_cache
    import pyfftw.interfaces.numpy_fft as fftw
    fftw_cache.enable()
    fft = fftw
    fft_kwargs = dict(planner_effort='FFTW_ESTIMATE', threads=nproc)
except ImportError:
    # pyFFTW is not available, just import numpys fft
    import numpy.fft as fft
    fft_kwargs = dict()


try:
    with warnings.catch_warnings():
        # skimage produces a DeprecationWarning by importing `imp`. We will silence this warning
        # as we have nothing to do with it
        warnings.simplefilter("ignore", DeprecationWarning)
        from skimage.restoration import unwrap_phase
except ImportError:
    unwrap_phase = None


__all__ = ['Field', 'Axis']


class Axis(object):
    '''
    Axis handling for a single Axis.

    Create an Axis object from scratch.

    The least required arguments are any of:
        * grid
        * grid_node
        * extent _and_ n

    The remaining fields will be deduced from the givens.

    More arguments may be supplied, as long as they are compatible.
    '''

    def __init__(self, name='', unit='', **kwargs):
        self.name = name
        self.unit = unit

        self._grid_node = kwargs.pop('grid_node', None)

        if self._grid_node is not None:
            self._grid_node = np.array(self._grid_node)
            if self._grid_node.ndim != 1:
                raise ValueError("Passed array grid_node has ndim != 1.")
            if helper.monotonicity(self._grid_node) == 0:
                raise ValueError("Passed array grid_node is not monotonous.")

        self._grid = kwargs.pop('grid', None)

        if self._grid is not None:
            self._grid = np.array(self._grid)
            if self._grid.ndim != 1:
                raise ValueError("Passed array grid has ndim != 1.")
            if helper.monotonicity(self._grid) == 0:
                raise ValueError("Passed array grid is not monotonous.")

        self._extent = kwargs.pop('extent', None)

        if self._extent is not None:
            if not isinstance(self._extent, collections.Iterable) or len(self._extent) != 2:
                raise ValueError("Passed extent is not an iterable of length 2")

        self._n = kwargs.pop('n', None)

        # kwargs must be exhausted now
        if len(kwargs) > 0:
            raise TypeError('got an unexpcted keyword argument "{}"'.format(kwargs))

        if self._grid_node is None:
            if self._grid is None:
                if self._extent is None or self._n is None:
                    # If we are here really nothing has been passed, like with the old version
                    # of this class
                    raise ValueError("Missing required arguments for Axis construction.")
                # only extent and n have been passed, use that to create a linear grid_node
                self._grid_node = np.linspace(self._extent[0], self._extent[-1], self._n+1,
                                              endpoint=True)
            else:
                # grid has been passed, create grid_node from grid.
                if len(self._grid) > 3:
                    grid_spline = scipy.interpolate.UnivariateSpline(np.arange(len(self._grid)),
                                                                     self._grid, s=0)
                    gn_inner = grid_spline(np.arange(0.5, len(self._grid)-1))
                    gn = np.pad(gn_inner, 1, 'constant')
                    del grid_spline
                else:
                    gn = np.convolve(self._grid, np.ones(2) / 2.0, mode='full')
                if self._extent is not None:
                    # extent has been passed, use this for the end points of grid_node
                    if self._extent[0] >= self._grid[0] or self._extent[-1] <= self._grid[-1]:
                        raise ValueError("Passed invalid extent.")
                    gn[0] = self._extent[0]
                    gn[-1] = self._extent[-1]
                else:
                    # estimate end points of grid_node as in the old grid.setter
                    if len(self._grid) > 1:
                        gn[0] = self._grid[0] + (self._grid[0] - gn[1])
                        gn[-1] = self._grid[-1] + (self._grid[-1] - gn[-2])
                    else:
                        gn[0] = self._grid[0] - 0.5
                        gn[-1] = self._grid[0] + 0.5
                self._grid_node = gn

        # now we are garantueed to have a grid_node
        if self._grid is None:
            # create grid from grid_node like in the old grid.getter
            if len(self._grid_node) > 3:
                node_spline = scipy.interpolate.UnivariateSpline(np.arange(-0.5,
                                                                           len(self._grid_node)-1),
                                                                 self._grid_node, s=0)
                self._grid = node_spline(np.arange(len(self._grid_node)-1))
                del node_spline
            else:
                self._grid = np.convolve(self._grid_node, np.ones(2) / 2.0, mode='valid')
        else:
            # check if grid and grid_node are compatible
            if not np.all(self._grid > self._grid_node[:-1]) and \
               np.all(self._grid < self._grid_node[1:]):
                    raise ValueError("Points of passed grid are not within corresponding "
                                     "grid_nodes.")

        # set extent if not given or check if compatible with grid_node
        if self._extent is None:
            self._extent = [self._grid_node[0], self._grid_node[-1]]
        elif self._extent[0] != self._grid_node[0] or self._extent[-1] != self._grid_node[-1]:
            raise ValueError("Passed invalid extent.")

        # make sure grid and grid_node is immutable
        self._grid.flags.writeable = False
        self._grid_node.flags.writeable = False

        # make sure the extent is also immutable
        self._extent = tuple(self._extent)

        # set n if not given or check if compatible with grid
        if self._n is None:
            self._n = len(self._grid)
        elif self._n != len(self._grid):
            raise ValueError("Passed invalid value of n.")

        self._linear = None
        self._inv_map = None

    def __getstate__(self):
        """
        Excludes self._inv_map from the pickled state
        """
        state = dict(self.__dict__)  # shallow copy
        state['_inv_map'] = None
        return state

    def __eq__(self, other):
        '''
        equality test for axis
        '''
        testattribs = ['extent', 'grid_node', 'grid']
        for ta in testattribs:
            if not np.all(np.isclose(getattr(self, ta), getattr(other, ta))):
                return False
        return True

    def islinear(self, force=False):
        """
        Checks if the axis has a linear grid.
        """
        if len(self) < 3:
            return True
        if self._linear is None or force:
            self._linear = helper.islinear(self._grid_node)
        return self._linear

    @property
    def isreversed(self):
        return self.extent[0] > self.extent[1]

    @property
    def grid_node(self):
        return self._grid_node

    @property
    def grid(self):
        return self._grid

    @property
    def spacing(self):
        if not self.islinear():
            raise TypeError('Grid must be linear to calculate gridspacing')
        return np.abs(self.grid_node[1] - self.grid_node[0])

    @property
    def extent(self):
        return self._extent

    @property
    def physical_length(self):
        return np.abs(self._extent[1] - self._extent[0])

    @property
    def label(self):
        if self.unit == '':
            ret = self.name
        else:
            ret = self.name + ' [' + self.unit + ']'
        return ret

    def _value_to_index_nonlinear(self, value):
        if self._inv_map is None:
            grid_and_nodes = np.zeros(2*len(self)+1)

            grid_and_nodes[::2] = self.grid_node
            grid_and_nodes[1::2] = self.grid

            indices_grid_and_nodes = np.linspace(-0.5, len(self)-0.5, len(grid_and_nodes))

            self._inv_map = spinterp.interp1d(grid_and_nodes, indices_grid_and_nodes)

        # clip the input values to the axes grid range
        value = np.clip(value, *sorted(self.extent))

        return self._inv_map(value)

    def _value_to_index_linear(self, value):
        a, b = self.extent
        lg = len(self)
        return (value-a)/(b-a) * lg - 0.5

    def _find_nearest_index(self, value):
        """
        Gives the index i of the value array[i] which is closest to value.
        Assumes that the array is sorted.
        """
        sortgrid = self.grid[::-1] if self.isreversed else self.grid
        # assert sortgrid is actually sorted
        assert np.all(np.sort(sortgrid) == sortgrid)
        side = {False: 'left', True: 'right'}[self.isreversed]
        idx = np.searchsorted(sortgrid, value, side=side)
        if self.isreversed:
            idx = len(self) - idx
        if idx > 0 and (idx == len(self) or
                        np.fabs(value - self.grid[idx-1]) < np.fabs(value - self.grid[idx])):
            return idx-1
        else:
            return idx

    def value_to_index(self, value):
        """
        This funtion is used to map values to indices in an interpolating manner, this is
        mainly used by the `map_coordinates` method of the `Field` class.

        In contrast to the `_find_nearest_index` method, this method does not return an integer
        but a fractional index that refers to a position between actual pixels.

        In general the equality

        `ax._find_nearest_index(x) == np.round(ax.value_to_index(x))`

        should hold.
        """
        if self.islinear():
            return self._value_to_index_linear(value)
        else:
            return self._value_to_index_nonlinear(value)

    def half_resolution(self):
        '''
        removes every second grid_node.
        '''
        grid_node = self.grid_node[::2]
        grid = 0.5 * (self.grid[:-1:2] + self.grid[1::2])
        ret = type(self)(self.name, self.unit, grid=grid, grid_node=grid_node)
        return ret

    def _inside_domain(self, val):
        '''
        returns true if val is inside the extent.
        '''
        se = np.sort(self.extent)
        return val >= se[0] and val <= se[1]

    def _extent_to_slice(self, extent):
        '''
        if the extent reverses the axis, the step argument of the returned slice is
        automatically set to -1, effectively auto-reversing the axis.
        '''
        a, b = extent
        if a is None:
            a = self._grid_node[0]
        if b is None:
            b = self._grid_node[-1]

        if not (self._inside_domain(a) or self._inside_domain(b)):
            s = 'The extent limits {} must at least overlap with the domain extent {}.'
            raise ValueError(s.format((a, b), self.extent))

        sortgrid = self.grid[::-1] if self.isreversed else self.grid
        # assert sortgrid is actually sorted
        assert np.all(np.sort(sortgrid) == sortgrid)
        side = {False: 'left', True: 'right'}[self.isreversed]
        slicelims = np.searchsorted(sortgrid, [a, b], side=side)
        if self.isreversed:
            start, stop = len(self) - slicelims
        else:
            start, stop = slicelims
        # auto-reverse is necessary
        slicedir = -1 if (a > b) != self.isreversed else None
        return slice(start, stop, slicedir)

    def _normalize_slice(self, index):
        """
        Applies some checks and transformations to the object passed
        to __getitem__
        """
        if isinstance(index, slice):
            if any(helper.is_non_integer_real_number(x) for x in (index.start, index.stop)):
                if index.step is not None:
                    raise IndexError('Non-Integer slices must have step == None')
                return self._extent_to_slice((index.start, index.stop))
            return index
        else:
            if helper.is_non_integer_real_number(index):
                # Indexing to a single position outside the extent
                # will yield IndexError. Identical behaviour as numpy.ndarray
                if not self._inside_domain(index):
                    msg = 'Physical index position {} is outside of the ' \
                          'extent {} of axis {}'.format(index, self.extent, str(self))
                    raise IndexError(msg)
                index = self._find_nearest_index(index)
            return slice(index, index+1)

    def reversed(self):
        '''
        returns an reversed Axis object
        '''
        ax = type(self)(self.name, self.unit, grid_node=self.grid_node[::-1], grid=self.grid[::-1])
        return ax

    def __getitem__(self, key):
        """
        Returns an Axis which consists of a sub-part of this object defined by
        a slice containing floats or integers or a float or an integer
        """
        sl = self._normalize_slice(key)
        if not (sl.step is None or np.abs(sl.step) == 1):
            raise ValueError("slice.step must be 1, -1 or None (but is {})".format(sl.step))
        grid = self.grid[sl]
        stop = sl.stop
        if stop is not None and stop > 0:
            stop += -1 if sl.step == -1 else 1
        sln = slice(sl.start if sl.start else None, stop)
        grid_node = self.grid_node[sln]
        ax = type(self)(self.name, self.unit, grid=grid, grid_node=grid_node)
        return ax

    def __len__(self):
        return self._n

    def __str__(self):
        s = '<Axis "{}" ({} grid points from {} to {})>'
        return s.format(self.name, len(self), *self.extent)

    __repr__ = __str__


def _reducing_numpy_method(method):
    """
    This function produces methods that are suitable for the `Field` class
    that reproduce the behaviour of the corresponding numpy `method`
    """
    @functools.wraps(getattr(np.ndarray, method))
    def new_method(self, axis=None, out=None, keepdims=None, **kwargs):
        # we need to interpret the axis object and create an iterable axisiter
        # in order to iterate over the affected axes
        axisiter = axis
        if axisiter is None:
            axisiter = tuple(range(self.dimensions))
        if not isinstance(axisiter, collections.Iterable):
            axisiter = (axis,)

        # no `out` argument supplied, we need to figure out the axes of the result
        # and deal with other state of the Field
        if out is None:
            axes = copy.copy(self.axes)
            tao = copy.copy(self.transformed_axes_origins)
            ats = copy.copy(self.axes_transform_state)
            if keepdims:
                for i in axisiter:
                    axes[i] = Axis(axes[i].name, axes[i].unit, extent=axes[i].extent, n=1)
            else:
                for i in reversed(sorted(axisiter)):
                    del axes[i]
                    del tao[i]
                    del ats[i]

        # If the supplied `out` argument is a Field, we need to extract the plain array
        # from it in order to pass to the real method
        real_out = out
        if isinstance(out, type(self)):
            real_out = out.matrix
        elif isinstance(out, tuple):
            real_out = tuple(o.matrix if isinstance(o, type(self)) else o)

        # call the underlying method and pass on `keepdims` if it is different from
        # None. Passing on `None` does not work because the default value is a special
        # object, see <https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/
        # numpy.all.html#numpy.all>
        if keepdims is not None:
            kwargs['keepdims'] = keepdims
        o = getattr(self.matrix, method)(axis=axis, out=real_out, **kwargs)

        # if an `out` argument was supplied, just return it
        if out:
            return out

        # create and return a Field object from the result.
        if isinstance(o, tuple):
            ret = tuple(type(self)(a, self.name, self.unit, axes=axes,
                                   axes_transform_state=ats, transformed_axes_origins=tao)
                        for a in o
                        )
        else:
            ret = type(self)(o, self.name, self.unit, axes=axes,
                             axes_transform_state=ats, transformed_axes_origins=tao)
        return ret
    return new_method


# The NDArrayOperatorsMixin implements all arithmetic special functions through numpy
# ufuncs
class Field(NDArrayOperatorsMixin):
    '''
    The Field Object carries data in form of an `numpy.ndarray` together with as many Axis
    objects as the data's dimensions. Additionaly the Field object
    provides any information that is necessary to plot _and_ annotate
    the plot.

    Create a Field object from scratch. The only required argument is `matrix` which
    contains the actual data.

    A `name` and a `unit` may be supplied.

    The axis may be specified in different ways:

    * by passing a list of Axis object as `axes`
    * by passing arrays with the grid_nodes as `xedges`, `yedges` and `zedges`.
      This is intended to work with `np.histogram`.
    * by not passing anything, which will create default axes from 0 to 1.
    '''

    @classmethod
    @helper.append_doc_of(io.load_field)
    def loadfrom(cls, filename):
        return io.load_field(filename)

    @classmethod
    @helper.append_doc_of(io.import_field)
    def importfrom(cls, filename, **kwargs):
        return io.import_field(filename, **kwargs)

    def __init__(self, matrix, name='', unit='', **kwargs):
        if 'xedges' in kwargs or 'axes' in kwargs:
            # Some axes have been passed, let length-1-dimensions alone
            self._matrix = np.asarray(matrix)  # dont sqeeze. trust numpys histogram functions.
        else:
            # No axes have been passed. Squeeze away length-1-dimensions.
            self._matrix = np.squeeze(matrix)

        self.name = name
        self.unit = unit
        self.axes = []
        self.infostring = ''
        self.infos = []
        self._label = None  # autogenerated if None

        if 'axes' in kwargs:
            if len(kwargs['axes']) < len(self._matrix.shape):
                raise ValueError("Number of supplied axis to small")
            self.axes = [None] * len(self.matrix.shape)
            for i, ax in enumerate(kwargs['axes']):
                self.setaxisobj(i, ax)
        else:
            if 'xedges' in kwargs:
                self._addaxisnodes(kwargs['xedges'], name='x')
            elif self.dimensions > 0:
                self._addaxis((0, 1), name='x')
            if 'yedges' in kwargs:
                self._addaxisnodes(kwargs['yedges'], name='y')
            elif self.dimensions > 1:
                self._addaxis((0, 1), name='y')
            if 'zedges' in kwargs:
                self._addaxisnodes(kwargs['zedges'], name='z')
            elif self.dimensions > 2:
                self._addaxis((0, 1), name='z')

        # Additions due to FFT capabilities

        # self.axes_transform_state is False for axes which live in spatial domain
        # and it is True for axes which live in frequency domain
        # This assumes that fields are initially created in spatial domain.
        if 'axes_transform_state' in kwargs:
            self.axes_transform_state = kwargs['axes_transform_state']
        else:
            self.axes_transform_state = [False] * len(self.shape)

        # self.transformed_axes_origins stores the starting values of the grid
        # from before the last transform was executed, this is used to
        # recreate the correct axis interval upon inverse transform
        if 'transformed_axes_origins' in kwargs:
            self.transformed_axes_origins = kwargs['transformed_axes_origins']
        else:
            self.transformed_axes_origins = [None] * len(self.shape)

    def __copy__(self):
        '''
        returns a shallow copy of the object.
        This method is called by `copy.copy(obj)`.
        Just copy enough to create copies for operator overloading.
        '''
        cls = type(self)
        ret = cls.__new__(cls)
        ret.__dict__.update(self.__dict__)  # shallow copy
        for k in ['infos', 'axes_transform_state', 'transformed_axes_origins', 'axes']:
            # copy iterables one level deeper
            # but matrix is not copied!
            ret.__dict__[k] = copy.copy(self.__dict__[k])
        return ret

    # Stuff related with compatibility to Numpy's ufuncs starts here.
    def _get_axes_ats_tao_binary_ufunc_broadcasting(self, other):
        """
        compute the axes, axes_transform_state and transformed_axes_origins for the result
        of a binary ufunc __call__ operation between self and other
        """
        if isinstance(other, numbers.Number):
            # if other is just a number, all properties should be inherited from self
            return self.axes, self.axes_transform_state, self.transformed_axes_origins

        # some short hands...
        axes1 = self.axes
        ats1 = self.axes_transform_state
        tao1 = self.transformed_axes_origins

        # create short hands for the properties of other
        if not isinstance(other, Field):
            # if other is a plain array, fill everything with None
            axes2 = [None]*other.ndim
            ats2 = [None]*other.ndim
            tao2 = [None]*other.ndim
        else:
            axes2 = other.axes
            ats2 = other.axes_transform_state
            tao2 = other.transformed_axes_origins

        # print("_get_axes_ats_tao_binary_ufunc_broadcasting self:", axes1, ats1, tao1)
        # print("_get_axes_ats_tao_binary_ufunc_broadcasting other:", axes2, ats2, tao2)

        # resulting array has total_dim dimensions
        total_dim = max(len(axes1), len(axes2))

        # enumerate axes objects and convert to a list, to support reverse iteration
        axes1 = list(enumerate(axes1))
        axes2 = list(enumerate(axes2))

        axes = []
        axes_transform_state = []
        transformed_axes_origins = []
        # collect result properties starting from the last axis, as broadcasting logic
        # of numpy also starts with last axis
        for (i1, ax1), (i2, ax2) in zip_longest(reversed(axes1), reversed(axes2),
                                                fillvalue=(None, None)):
            if ax1 is None:
                # ax1 is None, just use ax2
                axes.append(ax2)
                axes_transform_state.append(ats2[i2])
                transformed_axes_origins.append(tao2[i2])
            elif ax2 is None:
                # ax2 is None, just use ax1
                axes.append(ax1)
                axes_transform_state.append(ats1[i1])
                transformed_axes_origins.append(tao1[i1])
            elif len(ax1) == len(ax2):
                # both axes have same length, both should be valid.
                # TODO: Check if axes are really equal
                # print(len(ax1), len(ax2), ats1[i1], tao1[i1], ats2[i2], tao2[i2])
                axes.append(ax1)

                # we are not sure from which axis we should take ats and tao. guess...
                if tao2[i2] is None and i1 is not None:
                    axes_transform_state.append(ats1[i1])
                    transformed_axes_origins.append(tao1[i1])
                else:
                    axes_transform_state.append(ats2[i2])
                    transformed_axes_origins.append(tao2[i2])
            elif len(ax1) == 1:
                # ax1 has length 1, use ax2
                axes.append(ax2)
                axes_transform_state.append(ats2[i2])
                transformed_axes_origins.append(tao2[i2])
            elif len(ax2) == 1:
                # ax2 has length 1, use ax1
                axes.append(ax1)
                axes_transform_state.append(ats1[i1])
                transformed_axes_origins.append(tao1[i1])
            else:
                raise ValueError("Incompatible shapes for broadcasting")

        return list(reversed(axes)), list(reversed(axes_transform_state)), \
            list(reversed(transformed_axes_origins))

    # make sure that np.array() * Field() returns a Field and not a plain array
    __array_priority__ = 1

    def __array__(self, dtype=None):
        '''
        will be called by numpy function in case an numpy array is needed.
        '''
        return np.asanyarray(self.matrix, dtype=dtype)

    # What kind of other objects do we support? so far any kind of numpy array or scalar number
    _HANDLED_TYPES = (np.ndarray, numbers.Number)

    # handle ufuncs, new interface.
    # see https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/[...]
    # [...]/numpy.lib.mixins.NDArrayOperatorsMixin.html#numpy.lib.mixins.NDArrayOperatorsMixin
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # Have implemented neither 'reduceat' because it is crazy nor
        # 'inner' because it is not documented
        if method not in ['__call__', 'reduce', 'outer', 'at', 'accumulate']:
            return NotImplemented

        out = kwargs.get('out', ())
        for x in inputs + out:
            # Only support operations with instances of _HANDLED_TYPES.
            if not isinstance(x, self._HANDLED_TYPES + (type(self),)):
                # Any unsupported operation should return NotImplemented such that
                # numpy can continue to look for other methods that might support it
                return NotImplemented

        # TODO: add check of Axes extent and unit here

        # If out-argument set, an output Field was already created. Do not wworry
        # about axes in that case
        if not out:
            if method == '__call__':
                if len(inputs) == 1:
                    # unary operation, leave everything as it is
                    axes = self.axes
                    ats = self.axes_transform_state
                    tao = self.transformed_axes_origins
                elif len(inputs) == 2:
                    # binary operation, use Field._get_axes_ats_tao_binary_ufunc_broadcasting
                    a, b = inputs
                    if isinstance(a, type(self)):
                        axes, ats, tao = a._get_axes_ats_tao_binary_ufunc_broadcasting(b)
                    elif isinstance(b, type(self)):
                        axes, ats, tao = b._get_axes_ats_tao_binary_ufunc_broadcasting(a)
                else:
                    raise NotImplemented
            elif method == 'accumulate':
                axes = self.axes
                ats = self.axes_transform_state
                tao = self.transformed_axes_origins
            elif method == 'reduce':
                axes = copy.copy(self.axes)
                ats = self.axes_transform_state[:]
                tao = self.transformed_axes_origins[:]
                reduceaxis = kwargs.get('axis', 0)
                if not isinstance(reduceaxis, collections.Iterable):
                    reduceaxis = (reduceaxis,)
                for axis in reversed(sorted(set(reduceaxis))):
                    del axes[axis]
                    del ats[axis]
                    del tao[axis]
            elif method == 'outer':
                axes = []
                ats = []
                tao = []
                for i in inputs:
                    if isinstance(i, type(self)):
                        axes.extend(i.axes)
                        ats.extend(i.axes_transform_state)
                        tao.extend(i.transformed_axes_origins)
                    elif isinstance(i, np.ndarray):
                        for j in range(i.ndim):
                            a = Axis()
                            a.setextent(0, 1, j.shape[j])
                            axes.append(a)
                            ats.append(False)
                            tao.append(None)
            elif method == 'at':
                axes = None
        # Defer to the implementation of the ufunc on unwrapped values.
        inputs = tuple(x.matrix if isinstance(x, type(self)) else x for x in inputs)
        if out:
            if isinstance(out, type(self)):
                kwargs['out'] = out.matrix
            elif isinstance(out, tuple):
                kwargs['out'] = tuple(
                    x.matrix if isinstance(x, type(self)) else x for x in out)
        result = getattr(ufunc, method)(*inputs, **kwargs)

        # If out-argument set, return it. Unpack a one-tuple (important for binary inplace ops)
        if out:
            if isinstance(out, tuple) and len(out) == 1:
                return out[0]
            return out

        # Otherwise, the ufunc has returned one or more simple array(s). Wrap this/these
        # with the `axes`
        if type(result) is tuple:
            # multiple return values
            return tuple(type(self)(x, self.name, self.unit, axes=axes,
                                    axes_transform_state=ats, transformed_axes_origins=tao)
                         for x in result)
        elif method == 'at':
            # no return value
            return None
        else:
            # one return value
            obj = type(self)(result, self.name, self.unit, axes=axes,
                             axes_transform_state=ats, transformed_axes_origins=tao)
            return obj

    # wrap ufunc results from numpy < 1.13 as Fields.
    # This is also used for __add__ and so on as they are implemented through ufuncs via
    # NDArrayOperatorsMixin.
    # This is not intended to be perfect because it is barely possible to get it right
    # with the old interface
    def __array_wrap__(self, array, context=None):
        # this is a Field already, leave it as is
        if isinstance(array, type(self)):
            return array

        # fallback defaults
        axes = self.axes
        ats = self.axes_transform_state
        tao = self.transformed_axes_origins

        # if we have `context`, there might be a chance...
        if context:
            f, inputs, d = context
            if len(inputs) == 2:
                # binary operation, use Field._get_axes_ats_tao_binary_ufunc_broadcasting
                a, b = inputs
                if isinstance(a, type(self)):
                    axes, ats, tao = a._get_axes_ats_tao_binary_ufunc_broadcasting(b)
                elif isinstance(b, type(self)):
                    axes, ats, tao = b._get_axes_ats_tao_binary_ufunc_broadcasting(a)

        if array.ndim == len(axes):
            # we might have gotten `axes` right...
            return type(self)(array, self.name, self.unit, axes=axes,
                              axes_transform_state=ats, transformed_axes_origins=tao)

        # Have no Idea what the axes should be. Return plain `ndarray`.
        return array

    def _addaxisobj(self, axisobj):
        '''
        uses the given axisobj as the axis obj in the given dimension.
        '''
        # check if number of grid points match
        matrixpts = self.shape[len(self.axes)]
        if matrixpts != len(axisobj):
            raise ValueError(
                'Number of Grid points in next missing Data '
                'Dimension ({:d}) has to match number of grid points of '
                'new axis ({:d})'.format(matrixpts, len(axisobj)))
        self.axes.append(axisobj)

    def _addaxisnodes(self, grid_node, **kwargs):
        ax = Axis(grid_node=grid_node, **kwargs)
        self._addaxisobj(ax)
        return

    def _addaxis(self, extent, **kwargs):
        '''
        adds a new axis that is supported by the matrix.
        '''
        matrixpts = self.shape[len(self.axes)]
        ax = Axis(extent=extent, n=matrixpts, **kwargs)
        self._addaxisobj(ax)

    def setaxisobj(self, axis, axisobj):
        '''
        replaces the current axisobject for axis axis by the
        new axisobj axisobj.
        '''
        axid = helper.axesidentify[axis]
        if not len(axisobj) == self.shape[axid]:
            raise ValueError('Axis object has {:3n} grid points, whereas '
                             'the data matrix has {:3n} on axis {:1n}'
                             ''.format(len(axisobj),
                                       self.shape[axid], axid))
        self.axes[axid] = axisobj

    def islinear(self):
        return [a.islinear() for a in self.axes]

    @property
    def label(self):
        if self._label:
            ret = self._label
        elif self.unit == '':
            ret = self.name
        else:
            ret = self.name + ' [' + self.unit + ']'
        return ret

    @label.setter
    def label(self, x):
        self._label = x
        return

    @property
    def matrix(self):
        return self._matrix

    @matrix.setter
    def matrix(self, other):
        if np.shape(other) != self.shape:
            raise ValueError("Shape of old and new matrix must be identical")
        self._matrix = other

    @property
    def shape(self):
        return np.asarray(self).shape

    @property
    def grid_nodes(self):
        return np.squeeze([a.grid_node for a in self.axes])

    @property
    def grid(self):
        return np.squeeze([a.grid for a in self.axes])

    def meshgrid(self, sparse=True):
        return meshgrid(*[ax.grid for ax in self.axes], indexing='ij', sparse=sparse)

    @property
    def dimensions(self):
        '''
        returns only present dimensions.
        [] and [[]] are interpreted as -1
        np.array(2) is interpreted as 0
        np.array([1,2,3]) is interpreted as 1
        and so on...
        '''
        ret = len(self.shape)  # works for everything with data.
        if np.prod(self.shape) == 0:  # handels everything without data
            ret = -1
        return ret

    @property
    def ndim(self):
        return self.matrix.ndim

    @property
    def extent(self):
        '''
        returns the extents in a linearized form,
        as required by "matplotlib.pyplot.imshow".
        '''
        return np.ravel([a.extent for a in self.axes])

    @extent.setter
    def extent(self, newextent):
        '''
        sets the new extent to the specific values.
        '''
        if not self.dimensions * 2 == len(newextent):
            raise TypeError('size of newextent doesnt match self.dimensions * 2')
        for i in range(len(self.axes)):
            newax = Axis(self.axes[i].name, self.axes[i].unit,
                         extent=newextent[2 * i:2 * i + 2], n=self.shape[i])
            self.setaxisobj(i, newax)
        return

    @property
    def spacing(self):
        '''
        returns the grid spacings for all axis.
        '''
        return np.array([ax.spacing for ax in self.axes])

    @property
    def real(self):
        return self.replace_data(self.matrix.real)

    @property
    def imag(self):
        return self.replace_data(self.matrix.imag)

    @property
    def angle(self):
        return self.replace_data(np.angle(self))

    def phase(self, do_unwrap_phase=True):
        '''
        Returns the (unwrapped) phase of the complex Field.

        do_unwrap_phase: True, if skimage.restoration.unwrap_phase should be applied to data
        '''
        phase = np.angle(self)
        if do_unwrap_phase:
            phase = unwrap_phase(phase)
        ret = self.replace_data(phase)
        ret.name = r'$\varphi$({})'.format(self.name)
        ret.unit = r'rad'
        return ret

    def conj(self):
        return np.conj(self)

    def replace_data(self, other):
        ret = copy.copy(self)
        ret.matrix = other
        return ret

    def evaluate(self, ex, local_dict=None, global_dict=None, **kwargs):
        """
        Evaluates the expression `ex` using `NumExpr` and returns a field containing the result.
        This copies all metadata from `self` and just replaces the matrix.

        This function is basically syntactic sugar simplifying

        ```field.replace_data(ne.evaluate(expr))```

        to

        ```field.evaluate(expr)```

        This method replicates some logic from NumExpr.necompiler.getArguments(), seems
        there is no way around it.
        """
        call_frame = sys._getframe(1)

        clear_local_dict = False
        if local_dict is None:
            local_dict = call_frame.f_locals
            clear_local_dict = True
        try:
            frame_globals = call_frame.f_globals
            if global_dict is None:
                global_dict = frame_globals

            # If `call_frame` is the top frame of the interpreter we can't clear its
            # `local_dict`, because it is actually the `global_dict`.
            clear_local_dict = clear_local_dict and frame_globals is not local_dict

            ret = self.replace_data(ne.evaluate(ex, local_dict=local_dict,
                                                global_dict=global_dict, **kwargs))

        finally:
            if clear_local_dict:
                local_dict.clear()

        return ret

    def pad(self, pad_width, mode='constant', **kwargs):
        '''
        Pads the data using `np.pad` and takes care of the axes.
        See documentation of `numpy.pad`.

        In contrast to `np.pad`, `pad_width` may be given as integers, which will be interpreted
        as pixels, or as floats, which will be interpreted as distance along the appropriate axis.

        All other parameters are passed to `np.pad` unchanged.
        '''
        ret = copy.copy(self)
        if not self.islinear():
            raise ValueError('Padding the axes is only meaningful with linear axes.'
                             'Please apply np.pad to the matrix by yourself and update the axes'
                             'as you like.')

        if not isinstance(pad_width, collections.Iterable):
            pad_width = [pad_width]

        if len(pad_width) == 1:
            pad_width *= self.dimensions

        if len(pad_width) != self.dimensions:
            raise ValueError('Please check your pad_width argument. If it is an Iterable, its'
                             'length must equal the number of dimensions of this Field.')

        pad_width_numpy = []

        padded_axes = []

        for i, axis_pad in enumerate(pad_width):
            if not isinstance(axis_pad, collections.Iterable):
                axis_pad = [axis_pad, axis_pad]

            if len(axis_pad) > 2:
                raise ValueError

            if len(axis_pad) == 1:
                axis_pad = list(axis_pad)*2

            axis = ret.axes[i]

            dx = axis.spacing
            axis_pad = [int(np.ceil(p/dx))
                        if helper.is_non_integer_real_number(p)
                        else p
                        for p
                        in axis_pad]
            pad_width_numpy.append(axis_pad)

            totalpad_axis = sum(axis_pad)

            if totalpad_axis:
                padded_axes.append(i)

                extent = axis.extent
                newextent = [extent[0] - axis_pad[0]*dx, extent[1] + axis_pad[1]*dx]
                gridpoints = len(axis.grid_node) - 1 + axis_pad[0] + axis_pad[1]

                ret.axes[i] = Axis(axis.name, axis.unit, extent=newextent, n=gridpoints)

        ret._matrix = np.pad(self, pad_width_numpy, mode, **kwargs)

        # This info is invalidated for all axes which have actually been padded
        for i in padded_axes:
            ret.transformed_axes_origins[i] = None

        return ret

    def half_resolution(self, axis):
        '''
        Halfs the resolution along the given axis by removing
        every second `grid_node` and averaging every second data point into one.

        If there is an odd number of grid points, the last point will
        be ignored (that means, the extent will change by the size of
        the last grid cell).

        Returns
        -------
        Field:
            the modified `Field`.
        '''
        axis = helper.axesidentify[axis]
        ret = copy.copy(self)
        n = ret.matrix.ndim
        s1 = [slice(None), ] * n
        s2 = [slice(None), ] * n
        # ignore last grid point if self.matrix.shape[axis] is odd
        lastpt = ret.shape[axis] - ret.shape[axis] % 2
        # Averaging over neighboring points
        s1[axis] = slice(0, lastpt, 2)
        s2[axis] = slice(1, lastpt, 2)
        m = (ret.matrix[s1] + ret.matrix[s2]) / 2.0
        ret._matrix = m
        ret.setaxisobj(axis, ret.axes[axis].half_resolution())

        # This info is invalidated
        ret.transformed_axes_origins[axis] = None

        return ret

    def map_axis_grid(self, axis, transform, preserve_integral=True, jacobian_func=None):
        '''
        Transform the Field to new coordinates along one axis.

        This function transforms the coordinates of one axis according to the function
        `transform` and applies the jacobian to the data.

        Please note that no interpolation is applied to the data, instead a non-linear
        axis grid is produced. If you want to interpolate the data to a new (linear) grid,
        use the method :meth:`map_coordinates` instead.

        In contrast to :meth:`map_coordinates`,
        the function transform is not used to pull the new data
        points from the old grid, but is directly applied to the axis. This reverses the
        direction of the transform. Therfore, in order to preserve the integral,
        it is necessary to divide by the Jacobian.

        Parameters
        ----------
        axis: int
            the index or name of the axis you want to apply transform to.

        transform: callable
            the transformation function which takes the old coordinates as an input
            and returns the new grid

        preserve_integral: bool
            Divide by the jacobian of transform, in order to preserve the
            integral.

        jacobian_func: callable
            If given, this is expected to return the derivative of transform.
            If not given, the derivative is numerically approximated.
        '''
        axis = helper.axesidentify[axis]

        ret = copy.copy(self)

        if preserve_integral:
            if jacobian_func is None:
                jacobian_func = helper.approx_1d_jacobian_det(transform)

            jac_shape = [1]*self.dimensions
            jac_shape[axis] = len(ret.axes[axis])

            ret.matrix = ret.matrix / np.reshape(jacobian_func(ret.axes[axis].grid),
                                                 jac_shape)

        grid = transform(ret.axes[axis].grid)
        grid_node = transform(ret.axes[axis].grid_node)
        ret.axes[axis] = Axis(ret.axes[axis].name, ret.axes[axis].unit,
                              grid=grid, grid_node=grid_node)

        return ret

    def _map_coordinates(self, newaxes, transform=None, complex_mode='polar',
                         preserve_integral=True, jacobian_func=None,
                         jacobian_determinant_func=None, **kwargs):
        '''
        complex_mode:
            The complex_mode specifies how to proceed with complex data.

            * complex_mode = 'cartesian' - interpolate real/imag part (fastest)
            * complex_mode = 'polar' - interpolate abs/phase \
            If skimage.restoration is available, the phase will be unwrapped first (default)
            * complex_mode = 'polar-no-unwrap' - interpolate abs/phase \
            Skip unwrapping the phase, even if skimage.restoration is available

        preserve_integral: bool
            If True (the default), the data will be multiplied with the
            Jacobian determinant of the coordinate transformation such that the integral
            over the data will be preserved.

            In general, you will want to do this, because the physical unit of the new Field will
            correspond to the new axis of the Fields. Please note that Postpic, currently, does not
            automatically change the unit members of the Axis and Field objects, this you will have
            to do manually.

            There are, however, exceptions to this rule. Most prominently, if you are converting to
            polar coordinates,
            it depends on what you are going to do with the transformed Field.
            If you intend to do a Cartesian r-theta plot or are interested in a lineout
            for a single value of theta, you do want to apply the Jacobian determinant.
            If you had a density in
            e.g. J/m^2 than, in polar coordinates, you want to have a density in J/m/rad.
            If you intend, on the other hand, to do a polar plot, you do not want to apply the
            Jacobian. In a polar plot, the data points are plotted with variable density which
            visually takes care of the Jacobian automatically. A polar plot of the polar data
            should look like a Cartesian plot of the original data with just a peculiar coordinate
            grid drawn over it.

        jacobian_determinant_func: callable
            A callable that returns the jacobian determinant of
            the transform. If given, this takes precedence over the following option.

        jacobian_func: callable
            a callable that returns the jacobian of the transform. If this is
            not given, the jacobian is numerically approximated.

        **kwargs:
            Additional keyword arguments are passed to `scipy.ndimage.map_coordinates`,
            see the documentation of that function.
        '''
        # Instantiate an identity if no transformation function was given
        if transform is None:
            def transform(*x):
                return x

            def jacobian_determinant_func(*x):
                return 1.0

        if preserve_integral:
            if jacobian_determinant_func is None:
                if jacobian_func is None:
                    jacobian_func = helper.approx_jacobian(transform)
                jacobian_determinant_func = helper.jac_det(jacobian_func)

        do_unwrap_phase = True
        if complex_mode == 'polar-no-unwrap':
            complex_mode = 'polar'
            do_unwrap_phase = False

        # Start a new Field object by inserting the new axes
        ret = copy.copy(self)
        ret.axes = list(newaxes)
        shape = [len(ax) for ax in newaxes]

        # Calculate the output grid
        out_coords = ret.meshgrid()

        # Calculate the source points for every point of the new mesh
        coordinates_ax = transform(*out_coords)

        # Rescale the source coordinates to pixel coordinates
        coordinates_px = [ax.value_to_index(x) for ax, x in zip(self.axes, coordinates_ax)]

        # Broadcast all coordinate arrays to the new shape
        coordinates_px = [broadcast_to(c, shape) for c in coordinates_px]

        # Map the matrix using scipy.ndimage.map_coordinates
        if np.isrealobj(self.matrix):
            ret._matrix = spnd.map_coordinates(self.matrix, coordinates_px, **kwargs)
        else:
            if complex_mode == 'cartesian':
                real, imag = self.matrix.real.copy(), self.matrix.imag.copy()
                ret._matrix = np.empty(np.broadcast(*coordinates_px).shape,
                                       dtype=self.matrix.dtype)
                spnd.map_coordinates(real, coordinates_px, output=ret.matrix.real, **kwargs)
                spnd.map_coordinates(imag, coordinates_px, output=ret.matrix.imag, **kwargs)
            elif complex_mode == 'polar':
                angle = np.angle(self)
                if do_unwrap_phase:
                    if unwrap_phase:
                        angle = unwrap_phase(angle)
                    else:
                        warnings.warn("Function unwrap_phase from skimage.restoration not "
                                      "available! Install scikit-image or use complex_mode = "
                                      "'polar-no-unwrap' to get rid of this warning.")

                absval = spnd.map_coordinates(abs(self), coordinates_px, **kwargs)
                angle = spnd.map_coordinates(angle, coordinates_px, **kwargs)
                ret._matrix = absval * np.exp(1.j * angle)
            else:
                raise ValueError('Invalid value of complex_mode.')

        if preserve_integral:
            ret._matrix = ret._matrix * jacobian_determinant_func(*out_coords)

        # This info is invalidated
        ret.transformed_axes_origins = [None]*ret.dimensions
        transform_state = self._transform_state()
        if transform_state is None:
            transform_state = False
        ret.axes_transform_state = [transform_state]*ret.dimensions

        return ret

    @helper.append_doc_of(_map_coordinates)
    def map_coordinates(self, newaxes, transform=None, complex_mode='polar',
                        preserve_integral=True, jacobian_func=None,
                        jacobian_determinant_func=None, **kwargs):
        r'''
        Transform the Field to new coordinates.

        Parameters
        ----------
        newaxes: list
            The new axes of the new coordinates.

        transform: callable
            a callable that takes the new coordinates as input and returns
            the old coordinates from where to sample the Field.
            It is basically the inverse of the transformation that you want to perform.
            If transform is not given, the identity will be used. This is suitable for
            simple interpolation to a new extent/shape.
            Example for cartesian -> polar:

            >>> def T(r, theta):
            >>>    x = r * np.cos(theta)
            >>>    y = r * np.sin(theta)
            >>>    return x, y

            Note that this function actually computes the cartesian coordinates from the polar
            coordinates, but stands for transforming a field in cartesian coordinates into a
            field in polar coordinates.

            However, in order to preserve the definite integral of
            the field, it is necessary to multiply with the Jacobian determinant of T.

            .. math::
                \tilde{U}(r, \theta) = U(T(r, \theta)) \cdot \det
                \frac{\partial (x, y)}{\partial (r, \theta)}
            such that

            .. math::
                \int_V \mathop{\mathrm{d}x} \mathop{\mathrm{d}y} U(x,y) =
                \int_{T^{-1}(V)} \mathop{\mathrm{d}r}\mathop{\mathrm{d}\theta}
                \tilde{U}(r,\theta)\,.
        '''
        return self._map_coordinates(newaxes, transform=transform, complex_mode=complex_mode,
                                     preserve_integral=preserve_integral,
                                     jacobian_func=jacobian_func,
                                     jacobian_determinant_func=jacobian_determinant_func,
                                     **kwargs)

    def autoreduce(self, maxlen=4000):
        '''
        Reduces the Grid to a maximum length of maxlen per dimension
        by just executing half_resolution as often as necessary.
        '''
        ret = self  # half_resolution will take care for the copy
        for i in range(len(ret.axes)):
            while len(ret.axes[i]) > maxlen:
                ret = ret.half_resolution(i)

        return ret

    def cutout(self, newextent):
        '''
        only keeps that part of the data, that belongs to newextent.
        '''
        slices = self._extent_to_slices(newextent)
        return self[slices]

    def autocutout(self, axes=None, fractions=(0.001, 0.002)):
        '''
        Automatically cuts out the main feature of the field by removing border regions
        that only contain small numbers.

        This is done axis by axis. For each axis, the mean across all other axes is taken.
        The maximum `max` of the remaining 1d-`array` is taken and searched for the outermost
        boundaries a, d such that all values out of array[a:d] are smaller then fractions[0]*max.
        A second set of boundaries b, c is searched such that all values out of array[b:c] are
        smaller then fractions[1]*max.
        Because fractions[1] should be larger than fractions[0], array[b:c] should be contained
        completely in array[a:d].

        A padding length x is chosen such that array[b-x:c+x] is entirely within array[a:d].

        Then the corresponding axis of the field is sliced to [b-x:c+x] and multiplied with a
        tukey-window such that the region [b:c] is left untouched and the field in the padding
        region smoothly vanishes on the outer border.

        This process is repeated for all axes in `axes` or for all axes if `axes` is None.
        '''
        field = self.squeeze()

        if axes is None:
            axes = range(field.dimensions)

        if not isinstance(axes, collections.Iterable):
            axes = (axes, )

        if len(axes) != len(set(axes)):
            raise ValueError("This should be applied only once to each axis")

        # collect the slices which we will apply to field
        slices = [slice(None)]*field.dimensions

        # collect the sparse window functions which we will all apply in the end using numexpr
        windows = []

        for axis in axes:
            field_mean = abs(field)
            for otheraxis in range(field.dimensions):
                if otheraxis != axis:
                    field_mean = field_mean.mean(otheraxis)

            k = field_mean.shape[0]

            # outer bounds for lower threshold
            a, d = helper.max_frac_bounds(field_mean, fractions[0])

            # inner bounds for higher threshold
            b, c = helper.max_frac_bounds(field_mean, fractions[1])

            # Above should result in a<=b<=c<=d
            assert a <= b <= c <= d

            # Length of inner region which will be passed through unchanged
            ll = c-b

            # length of remaining region befor/after inner region
            x = max(d-c, b-a)
            x = min(x, k-c, b)

            # final indices of slice
            e, f = b-x, c+x
            slices[axis] = slice(e, f)

            # new length of the axis
            m = f-e

            shape = [1]*field.dimensions
            shape[axis] = m
            windows.append(np.reshape(tukey(m, 1-ll/m), shape))

        field = field[slices]
        varnames = "abcdefg"
        expr = "*".join(varnames[:len(windows)+1])
        local_dict = {v: w for v, w in zip(varnames[1:], windows)}
        local_dict['a'] = field

        return field.replace_data(ne.evaluate(expr, local_dict=local_dict, global_dict=None))

    def squeeze(self):
        '''
        removes axes that have length 1, reducing self.dimensions.

        Note, that axis with length 0 will not be removed! `numpy.squeeze` also does not
        remove length=0 directions.

        Same as `numpy.squeeze`.
        '''
        ret = copy.copy(self)
        retained_axes = [i for i in range(len(self.shape)) if len(self.axes[i]) != 1]

        ret.axes = [self.axes[i] for i in retained_axes]
        ret.axes_transform_state = [self.axes_transform_state[i] for i in retained_axes]
        ret.transformed_axes_origins = [self.transformed_axes_origins[i] for i in retained_axes]

        ret._matrix = np.squeeze(ret.matrix)
        assert tuple(len(ax) for ax in ret.axes) == ret.shape
        return ret

    def atleast_nd(self, n):
        '''
        Make sure the field has at least 'n' dimensions
        '''
        if self.dimensions >= n:
            return self

        additional_dims = n - self.dimensions
        transform_state = self._transform_state()

        ret = copy.copy(self)

        for _ in range(additional_dims):
            ret._matrix = ret._matrix[..., np.newaxis]
            ret.axes.append(Axis(grid_node=np.array([-0.5, 0.5])))
            ret.transformed_axes_origins.append(None)
            ret.axes_transform_state.append(transform_state)

        return ret

    def transpose(self, *axes):
        '''
        transpose method equivalent to `numpy.ndarray.transpose`. If `axes` is empty,
        the order of the
        axes will be reversed. Otherwise axes[i] == j means that the i'th axis of the returned
        Field will be the j'th axis of the input Field.
        '''
        if not axes:
            axes = list(reversed(range(self.dimensions)))
        elif len(axes) == self.dimensions:
            pass
        else:
            axes = axes[0]

        if len(axes) != self.dimensions:
            raise ValueError('Invalid axes argument')

        ret = copy.copy(self)
        ret.axes = [ret.axes[i] for i in axes]
        ret._matrix = ret._matrix.transpose(*axes)
        return ret

    @property
    def T(self):
        """
        Return the Field with the order of axes reversed. In 2D this is the usual matrix
        transpose operation.
        """
        return self.transpose()

    def swapaxes(self, axis1, axis2):
        '''
        Swaps the axes `axis1` and `axis2`, equivalent to `numpy.swapaxes`.
        '''
        axes = list(range(self.dimensions))
        axes[axis1] = axis2
        axes[axis2] = axis1
        return self.transpose(*axes)

    all = _reducing_numpy_method("all")
    any = _reducing_numpy_method("any")
    max = _reducing_numpy_method("max")
    min = _reducing_numpy_method("min")
    prod = _reducing_numpy_method("prod")
    sum = _reducing_numpy_method("sum")
    ptp = _reducing_numpy_method("ptp")
    std = _reducing_numpy_method("std")
    mean = _reducing_numpy_method("mean")
    var = _reducing_numpy_method("var")

    def clip(self, a_min, a_max, out=None):
        o = np.clip(self.matrix, a_min, a_max, out=out)
        if out:
            return out
        return self.replace_data(o)

    def flip(self, axis):
        '''
        functionality of `numpy.flip`.

        `field.flip(0)` returns a `postpic.Field` object with the
        specified axis flipped. `np.flip(field)` returns only
        the `numpy.ndarray` and the axis information is lost.
        '''
        ret = self.replace_data(np.flip(self, axis))
        ax = ret.axes[axis].reversed()
        ret.axes[axis] = ax
        return ret

    def rot90(self, k=1, axes=(0, 1)):
        """
        Rotates the field by 90 degrees, `k` times. Rotates the field in the plane given by `axes`.
        """
        # copied most of the code from
        # https://github.com/numpy/numpy/blob/v1.15.1/numpy/lib/function_base.py#L62-L145
        axes = tuple(axes)
        if len(axes) != 2:
            raise ValueError("len(axes) must be 2.")
        if axes[0] == axes[1]:
            raise ValueError("Axes must be different.")
        if (axes[0] >= self.ndim or axes[0] < -self.ndim or
                axes[1] >= self.ndim or axes[1] < -self.ndim):
            raise ValueError("Axes={} out of range for array of ndim={}."
                             .format(axes, self.ndim))

        k %= 4

        if k == 0:
            return copy.copy(self)
        if k == 2:
            return self.flip(axes[0]).flip(axes[1])

        axes_list = np.arange(0, self.ndim)
        (axes_list[axes[0]], axes_list[axes[1]]) = (axes_list[axes[1]],
                                                    axes_list[axes[0]])

        if k == 1:
            return self.flip(axes[1]).transpose(axes_list)
        else:
            # k == 3
            return self.transpose(axes_list).flip(axes[1])

    def ensure_positive_axes(self):
        '''
        ensures, that all axis are going from smaller to greater numbers,
        i.e. none of the axis is reversed.
        '''
        ret = self
        for i, ax in enumerate(self.axes):
            if ax.isreversed():
                ret = ret.flip(i)
        return ret

    def _integrate_constant(self, axes):
        '''
        Integrate by assuming constant value across each grid cell, even for uneven grids.
        This effectively assumes cell-oriented data where each data point already represents
        an average over its cell
        '''
        axes = tuple(sorted(set(axes)))

        ret = self

        for axis in reversed(axes):
            box_sizes = self.axes[axis].grid_node[1:] - self.axes[axis].grid_node[:-1]
            shape = [1] * self.dimensions
            shape[axis] = len(box_sizes)
            box_sizes = np.reshape(box_sizes, shape)
            ret = ret * box_sizes

        ret = ret.sum(axes)

        return ret

    def _integrate_scipy(self, axes, method):
        ret = copy.copy(self)
        for axis in reversed(sorted(axes)):
            ret._matrix = method(ret, ret.axes[axis].grid, axis=axis)
            del ret.axes[axis]
            del ret.axes_transform_state[axis]
            del ret.transformed_axes_origins[axis]

        return ret

    def _integrate_fast(self, axes):
        '''
        Integrate by assuming constant value across each grid cell, even for uneven grids.
        This effectively assumes cell-oriented data where each data point already represents
        an average over its cell.

        Note: This is effectively equivalent to _integrate_constant, but should be a lot faster
        on larger arrays. However for the time being, _integrate_constant is left as is to have a
        refernce for testing and comparison of speed.
        '''
        ret = copy.copy(self)

        # sort the unique set of axes by the number of grid points, in ascending order
        axes = sorted(set(axes), key=lambda i: self.shape[i])
        while axes:
            # pops the last value from axes which is the index of the axis with the most points
            axis = axes.pop()

            # get the box sizes along that axis and put them into an array with a shape compatible
            # with the current matrix
            shape = [1] * ret.dimensions
            shape[axis] = ret.shape[axis]
            upper = ret.axes[axis].grid_node[1:].reshape(shape)
            lower = ret.axes[axis].grid_node[:-1].reshape(shape)

            # perform summation, replace matrix and adapt metadata according to the removal of the
            # current axis
            ret._matrix = ne.evaluate('sum((upper-lower) * ret, axis={})'.format(axis))
            del ret.axes[axis]
            del ret.axes_transform_state[axis]
            del ret.transformed_axes_origins[axis]

            # reduce the remaining axis indices by 1, if they refer to an axis with a higher index
            # than the current axis to account for removal of the current axis
            axes = [i-1 if i > axis else i for i in axes]

        return ret

    def integrate(self, axes=None, method=scipy.integrate.simps):
        '''
        Calculates the definite integral along the given axes.

        Parameters
        ----------
        method: callable
            Choose the method to use. Available options:

            * 'constant'
            * any function with the same signature as scipy.integrate.simps (default).
        '''
        if not callable(method) and method not in ['constant', 'fast']:
            raise ValueError("Requested method {} is not supported".format(method))

        if axes is None:
            axes = range(self.dimensions)

        if not isinstance(axes, collections.Iterable):
            axes = (axes,)

        if method == 'constant':
            return self._integrate_constant(axes)
        elif method == 'fast':
            return self._integrate_fast(axes)
        else:
            return self._integrate_scipy(axes, method)

    def _derivative(self, axis):
        from pkg_resources import parse_version
        if parse_version(np.__version__) < parse_version('1.9'):
            if not self.axes[axis].islinear():
                raise ValueError('This method can only be applied to linear axes.')
            g = np.gradient(self, self.axes[axis].spacing)
            if self.dimensions > 1:
                g = g[axis]
            der_field = self.replace_data(g)
        elif parse_version(np.__version__) < parse_version('1.13'):
            if not self.axes[axis].islinear():
                raise ValueError('This method can only be applied to linear axes.')
            der_field = self.replace_data(np.gradient(self, self.axes[axis].spacing, axis=axis))
        else:
            # this works fine even on non-linear axes
            der_field = self.replace_data(np.gradient(self, self.axes[axis].grid, axis=axis))

        der_field.name = "{}'".format(self.name)
        return der_field

    def _derivative_stagger(self, axis):
        oldax = self.axes[axis]
        if not oldax.islinear():
            raise ValueError('This method can only be applied to linear axes.')

        axes = self.axes[:]
        axes[axis] = Axis(grid=oldax.grid_node[1:-1], name=oldax.name, unit=oldax.unit)

        index1 = [slice(None) for _ in range(self.dimensions)]
        index1[axis] = slice(1, None)
        index2 = [slice(None) for _ in range(self.dimensions)]
        index2[axis] = slice(0, -1)

        deriv = (self.matrix[index1] - self.matrix[index2])/oldax.spacing

        return Field(deriv, name=self.name + "'", unit=self.unit, axes=axes)

    def derivative(self, axis, staggered=False):
        """
        Calculate the derivative of the field with respect to `axis`.

        Uses `np.gradient` by default which outputs the second order accurate derivative on the
        same grid as the input field.

        If `staggered=True` is passed, the method will instead calculate the second order
        accurate derivative at the points centered between the input grid points.
        """
        if staggered:
            return self._derivative_stagger(axis)
        else:
            return self._derivative(axis)

    def _transform_state(self, axes=None):
        """
        Returns the collective transform state of the given axes.

        If all mentioned axis i have `self.axes_transform_state[i]==True` return True
        (All axes live in frequency domain).

        If all mentioned axis i have self.axes_transform_state[i]==False return False
        (All axes live in spatial domain)

        Else return None
        (Axes have mixed transform_state)
        """
        if axes is None:
            axes = range(self.dimensions)

        for b in [True, False]:
            if all(self.axes_transform_state[i] == b for i in axes):
                return b
        return None

    def fft_autopad(self, axes=None, fft_padsize=helper.fftw_padsize):
        """
        Automatically pad the array to a size such that computing its FFT using FFTW will be fast.

        Parameters
        ----------
        fft_padsize: callable
            The default for keyword argument `fft_padsize` is a callable,
            that is used to calculate the padded size for a given size.

            By default, this uses `fft_padsize=helper.fftw_padsize`
            which finds the next larger "good"
            grid size according to what the FFTW documentation says.

            However, the FFTW documentation also says:
            "(...) Transforms whose sizes are powers of 2 are especially fast."

            If you don't worry about the extra padding, you can pass
            `fft_padsize=helper.fft_padsize_power2`
            and this method will pad to the next power of 2.
        """
        if axes is None:
            axes = range(self.dimensions)

        if not isinstance(axes, collections.Iterable):
            axes = (axes,)

        pad = [0] * self.dimensions

        for axis in axes:
            ll = self.shape[axis]
            pad0 = fft_padsize(ll) - ll
            pad1 = pad0 // 2
            pad2 = pad0 - pad1
            pad[axis] = [pad1, pad2]

        return self.pad(pad)

    def _conjugate_grid(self, axes=None):
        """
        Calculate the new grid that will emerge when a FFT would be transformed.
        """
        # If axes is None, transform all axes
        if axes is None:
            axes = range(self.dimensions)

        # If axes is not a tuple, make it a one-tuple
        if not isinstance(axes, collections.Iterable):
            axes = (axes,)

        dx = {i: self.axes[i].spacing for i in axes}
        new_axes = {
            i: fft.fftshift(2*np.pi*fft.fftfreq(self.shape[i], dx[i]))
            for i in axes
        }

        for i in axes:
            # restore original axes origins
            if self.transformed_axes_origins[i] is not None:
                new_axes[i] += self.transformed_axes_origins[i] - new_axes[i][0]
        return new_axes

    def fft(self, axes=None, exponential_signs='spatial', **kwargs):
        '''
        Performs Fourier transform on any number of axes.

        The argument axis is either an integer indicating the axis to be transformed
        or a tuple giving the axes that should be transformed. Automatically determines
        forward/inverse transform. Transform is only applied if all mentioned axes are
        in the same transform state.
        If an axis is transformed twice, the origin of the axis is restored.

        Parameters
        ----------

        exponential_signs:
            configures the sign convention of the exponential.

            * exponential_signs == 'spatial':  fft using exp(-ikx), ifft using exp(ikx)
            * exponential_signs == 'temporal':  fft using exp(iwt), ifft using exp(-iwt)

        **kwargs:
            keyword-arguments are passed to the underlying fft implementation.
        '''
        # If axes is None, transform all axes
        if axes is None:
            axes = range(self.dimensions)

        # If axes is not a tuple, make it a one-tuple
        if not isinstance(axes, collections.Iterable):
            axes = (axes,)

        if exponential_signs not in ['spatial', 'temporal']:
            raise ValueError('Argument exponential_signs has an invalid value.')

        # List axes uniquely and in ascending order
        axes = sorted(set(axes))

        if not all(self.axes[i].islinear() for i in axes):
            raise ValueError("FFT only allowed for linear grids")

        # Get the collective transform state of the axes
        transform_state = self._transform_state(axes)

        if transform_state is None:
            raise ValueError("FFT only allowed if all mentioned axes are in same transform state")

        # Record current axes origins of transformed axes
        new_origins = {i: self.axes[i].grid[0] for i in axes}

        # Grid spacing
        dx = {i: self.axes[i].spacing for i in axes}

        # Unit volume of transform
        dV = np.product(list(dx.values()))

        # Number of grid cells of transform
        N = np.product([self.shape[i] for i in axes])

        # Total volume of transform
        V = dV*N

        # Total volume of conjugate space
        # print('dx', dx, 'dV', dV, 'N', N, 'V', V)
        Vk = (2*np.pi)**len(dx)/dV

        # normalization factor ensuring Parseval's Theorem
        fftnorm = np.sqrt(V/Vk)

        # compile fft arguments, starting from default arguments `fft_kwargs` ...
        my_fft_args = fft_kwargs.copy()
        # ... and adding the user supplied `kwargs`
        my_fft_args.update(kwargs)
        # ... and also norm = 'ortho'
        my_fft_args['norm'] = 'ortho'

        # Workaround for missing `fft` argument `norm='ortho'`
        from pkg_resources import parse_version
        if parse_version(np.__version__) < parse_version('1.10'):
            del my_fft_args['norm']
            if transform_state is False:
                fftnorm /= np.sqrt(N)
            elif transform_state is True:
                fftnorm *= np.sqrt(N)

        mat = self.matrix

        if exponential_signs == 'temporal':
            mat = np.conjugate(mat)

        new_axes = self._conjugate_grid(axes)

        # Transforming from spatial domain to frequency domain ...
        if transform_state is False:
            new_axesobjs = {
                i: Axis('w' if self.axes[i].name == 't' else 'k'+self.axes[i].name,
                        '1/'+self.axes[i].unit,
                        grid=new_axes[i])
                for i in axes
            }
            mat = fftnorm \
                * fft.fftshift(fft.fftn(mat, axes=axes, **my_fft_args), axes=axes)

        # ... or transforming from frequency domain to spatial domain
        elif transform_state is True:
            new_axesobjs = {
                i: Axis('t' if self.axes[i].name == 'w' else self.axes[i].name.lstrip('k'),
                        self.axes[i].unit.lstrip('1/'),
                        grid=new_axes[i])
                for i in axes
            }
            mat = fftnorm \
                * fft.ifftn(fft.ifftshift(mat, axes=axes), axes=axes, **my_fft_args)

        if exponential_signs == 'temporal':
            mat = np.conjugate(mat)

        ret = copy.copy(self)
        ret.matrix = mat

        # Update axes objects

        for i in axes:
            # update axes objects
            ret.setaxisobj(i, new_axesobjs[i])

            # update transform state and record axes origins
            ret.axes_transform_state[i] = not transform_state
            ret.transformed_axes_origins[i] = new_origins[i]

        return ret

    def ensure_transform_state(self, transform_states):
        """
        Makes sure that the field has the given transform_states. `transform_states` may be
        a single boolean, indicating the same desired transform_state for all axes.
        It may be a list of the desired transform states for all the axes or a dictionary
        indicating the desired transform states of specific axes.
        """
        if not isinstance(transform_states, collections.Mapping):
            if not isinstance(transform_states, collections.Iterable):
                transform_states = [transform_states] * self.dimensions
            transform_states = dict(enumerate(transform_states))

        transform_axes = []
        for axid in sorted(transform_states.keys()):
            if self.axes_transform_state[axid] != transform_states[axid]:
                transform_axes.append(axid)

        if len(transform_axes) == 0:
            return self

        return self.fft(tuple(transform_axes))

    def ensure_spatial_domain(self):
        return self.ensure_transform_state(False)

    def ensure_frequency_domain(self):
        return self.ensure_transform_state(True)

    def _apply_linear_phase(self, dx):
        '''
        Apply a linear phase as part of translating the grid points.

        dx should be a mapping from axis number to translation distance
        All axes must have same transform_state and transformed_axes_origins not None
        '''
        transform_state = self._transform_state(dx.keys())
        if transform_state is None:
            raise ValueError("Translation only allowed if all mentioned axes"
                             "are in same transform state")

        if any(self.transformed_axes_origins[i] is None for i in dx.keys()):
            raise ValueError("Translation only allowed if all mentioned axes"
                             "have transformed_axes_origins not None")

        exp_ikdx = helper.linear_phase(self, dx)

        ret = self * exp_ikdx

        for i in dx.keys():
            ret.transformed_axes_origins[i] += dx[i]

        return ret

    def _shift_grid_by_fourier(self, dx):
        axes = sorted(dx.keys())
        ret = self.fft(axes)
        ret = ret._apply_linear_phase(dx)
        return ret.fft(axes)

    def _shift_grid_by_linear(self, dx):
        axes = sorted(dx.keys())
        shift = np.zeros(len(self.axes))
        for i, d in dx.items():
            shift[i] = d
        shift_px = shift/self.spacing
        ret = copy.copy(self)
        if np.isrealobj(self.matrix):
            ret.matrix = spnd.shift(self.matrix, -shift_px, order=1, mode='nearest')
        else:
            real, imag = self.matrix.real.copy(), self.matrix.imag.copy()
            ret.matrix = np.empty_like(matrix)
            spnd.shift(real, -shift_px, output=ret.matrix.real, order=1, mode='nearest')
            spnd.shift(imag, -shift_px, output=ret.matrix.imag, order=1, mode='nearest')

        for i in axes:
            ret.axes[i] = Axis(grid_node=self.axes[i].grid_node + dx[i],
                               grid=self.axes[i].grid + dx[i])

        return ret

    def shift_grid_by(self, dx, interpolation='fourier'):
        '''
        Translate the Grid by `dx`.
        This is useful to remove the grid stagger of field components.

        If all axis will be shifted, `dx` may be a list.
        Otherwise dx should be a mapping from axis to translation distance.

        The keyword-argument interpolation indicates the method to be used and
        may be one of `['linear', 'fourier']`.
        In case of interpolation = 'fourier' all axes must have same transform_state.
        '''
        methods = dict(fourier=self._shift_grid_by_fourier,
                       linear=self._shift_grid_by_linear)

        if interpolation not in methods.keys():
            raise ValueError("Requested method {} is not supported".format(interpolation))

        if not isinstance(dx, collections.Mapping):
            dx = dict(enumerate(dx))

        dx = {helper.axesidentify[i]: v for i, v in dx.items()}

        return methods[interpolation](dx)

    def adjust_stagger_to(self, other):
        origin = [ax.grid[0] for ax in other.axes]
        return helper.unstagger_fields(self, origin=origin)[0]

    @helper.append_doc_of(_map_coordinates)
    def topolar(self, extent=None, shape=None, angleoffset=0, **kwargs):
        '''
        Transform the Field to polar coordinates.

        This is a convenience wrapper for :meth:`map_coordinates` which will let you easily
        define the desired grid in polar coordinates.

        Parameters
        ----------
        extent:
            should be of the form `extent=(phimin, phimax, rmin, rmax)` or
            `extent=(phimin, phimax)`
        shape:
            should be of the form `shape=(N_phi, N_r)`,
        angleoffset:
            can be any real number and will rotate the zero-point of the angular axis.
        '''
        # Fill extent and shape with sensible defaults if nothing was passed
        if extent is None or len(extent) < 4:
            r_min = np.sqrt(np.min(np.abs(self.grid[0]))**2 +
                            np.min(np.abs(self.grid[1]))**2)
            r_max = np.sqrt(np.max(np.abs(self.grid[0]))**2 +
                            np.max(np.abs(self.grid[1]))**2)
            if extent is None:
                extent = [-np.pi, np.pi, r_min, r_max]
            else:
                extent = [extent[0], extent[1], r_min, r_max]
        extent = np.asarray(extent)
        if shape is None:
            ptr_r = int((extent[3]-extent[2])/np.min(self.spacing))
            ptr_r = min(1000, ptr_r)
            ptr_t = int((extent[1]-extent[0])*(extent[2]+extent[3])/2.0/min(self.spacing))
            ptr_t = min(1000, ptr_t)
            shape = (ptr_t, ptr_r)

        # Create the new axes objects
        theta = Axis(name='theta', unit='rad',
                     grid=np.linspace(extent[0], extent[1], shape[0]))

        theta_offset = Axis(name='theta', unit='rad',
                            grid=np.linspace(extent[0], extent[1], shape[0]) - angleoffset)

        if self.axes[0].name.startswith('k'):
            rname = 'k'
        else:
            rname = 'r'

        r = Axis(name=rname, unit=self.axes[0].unit,
                 grid=np.linspace(extent[2], extent[3], shape[1]))

        # Perform the transformation
        ret = self.map_coordinates([theta_offset, r],
                                   transform=helper.polar2linear,
                                   jacobian_determinant_func=helper.polar2linear_jacdet,
                                   **kwargs)

        # Remove the angleoffset from the theta grid
        ret.setaxisobj(0, theta)

        return ret

    @helper.append_doc_of(io.export_field)
    def export(self, filename, **kwargs):
        '''
        Uses `postpic.export_field` to export this field to a file. All ``**kwargs`
        will be forwarded to this function.
        Format is recognized by the extension
        of the filename.
        '''
        io.export_field(filename, self, **kwargs)

    def saveto(self, filename):
        '''
        Save a Field object as a file. Use `loadfrom()` to load Field objects.
        '''
        if not filename.endswith('.npz'):
            filename += '.npz'
        self.export(filename)

    def __str__(self):
        s = '<postpic.Field "{:}" {:}>'
        return s.format(self.name, self.shape)

    __repr__ = __str__

    def _extent_to_slices(self, extent):
        if not self.dimensions * 2 == len(extent):
            raise TypeError('size of extent doesnt match self.dimensions * 2')

        extent = np.reshape(np.asarray(extent), (self.dimensions, 2))
        return [ax._extent_to_slice(ex) for ax, ex in zip(self.axes, extent)]

    def _normalize_slices(self, key):
        if not isinstance(key, collections.Iterable):
            key = (key,)
        if len(key) != self.dimensions:
            raise IndexError("{}D Field requires a {}-tuple of slices as index"
                             "".format(self.dimensions, self.dimensions))

        return [ax._normalize_slice(sl) for ax, sl in zip(self.axes, key)]

    # Operator overloading
    def __getitem__(self, key):
        old_shape = self.shape

        key = self._normalize_slices(key)
        field = copy.copy(self)
        field._matrix = field.matrix[key]
        for i, sl in enumerate(key):
            field.setaxisobj(i, field.axes[i][sl])

        new_shape = field.shape

        # This info is invalidated
        for i, (o, n) in enumerate(zip(old_shape, new_shape)):
            if o != n:
                field.transformed_axes_origins[i] = None

        return field

    def __setitem__(self, key, other):
        key = self._normalize_slices(key)
        self._matrix[key] = other
