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
# Stephan Kuschel, 2014-2017
# Alexander Blinne, 2017
"""
The Core module for final data handling.

This module provides classes for dealing with axes, grid as well as the Field
class -- the final output of the postpic postprocessor.

Terminology
-----------

A data field with N numeric points has N 'grid' points,
but N+1 'grid_nodes' as depicted here:

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

import functools
import collections
import copy
import warnings
import os

import numpy as np
import scipy.ndimage as spnd
import scipy.interpolate as spinterp
import scipy.integrate
import scipy.signal as sps
import numexpr as ne

from ._compat import tukey, meshgrid, broadcast_to

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
    fftw_cache.set_keepalive_time(3600)
    fft = fftw
    fft_kwargs = dict(planner_effort='FFTW_ESTIMATE', threads=nproc)
except ImportError:
    # pyFFTW is not available, just import numpys fft
    import numpy.fft as fft
    using_pyfftw = False
    fft_kwargs = dict()


try:
    with warnings.catch_warnings():
        # skimage produces a DeprecationWarning by importing `imp`. We will silence this warning
        # as we have nothing to do with it
        warnings.simplefilter("ignore", DeprecationWarning)
        from skimage.restoration import unwrap_phase
except ImportError:
    unwrap_phase = None

from . import helper

__all__ = ['Field', 'Axis']


class Axis(object):
    '''
    Axis handling for a single Axis.
    '''

    def __init__(self, name='', unit='', **kwargs):
        self.name = name
        self.unit = unit

        self._grid_node = kwargs.get('grid_node', None)
        self._grid = kwargs.get('grid', None)
        self._extent = kwargs.get('extent', None)
        self._n = kwargs.get('n', None)

        if self._grid_node is None:
            if self._grid is None:
                if self._extent is None or self._n is None:
                    raise ValueError("Missing required arguments for Axis construction.")
                self._grid_node = np.linspace(*self._extent, self._n+1, endpoint=True)
            else:
                gn = np.convolve(self._grid, np.ones(2) / 2.0, mode='full')
                gn[0] = self._grid[0] + (self._grid[0] - gn[1])
                gn[-1] = self._grid[-1] + (self._grid[-1] - gn[-2])
                self._grid_node = gn

        if self._grid is None:
            self._grid = np.convolve(self._grid_node, np.ones(2) / 2.0, mode='valid')

        if self._extent is None:
            self._extent = [self._grid_node[0], self._grid_node[-1]]
        elif self._extent[0] > self._grid[0] or self._extent[-1] < self._grid[-1]:
            raise ValueError("Passed invalid extent.")

        if self._n is None:
            self._n = len(self._grid)
        elif self._n != len(self._grid):
            raise ValueError("Passed invalid value of n.")

        self._linear = None

    def __copy__(self):
        '''
        returns a shallow copy of the object.
        This method is called by `copy.copy(obj)`.
        '''
        cls = type(self)
        ret = cls.__new__(cls)
        ret.__dict__.update(self.__dict__)
        return ret

    def islinear(self, force=False):
        """
        Checks if the axis has a linear grid.
        """
        if self._linear is None or force:
            self._linear = helper.islinear(self._grid_node)
        return self._linear

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
        return self.grid_node[1] - self.grid_node[0]

    @property
    def extent(self):
        return self._extent

    @property
    def physical_length(self):
        return self._extent[1] - self._extent[0]

    @property
    def label(self):
        if self.unit == '':
            ret = self.name
        else:
            ret = self.name + ' [' + self.unit + ']'
        return ret

    def value_to_index(self, value):
        if not self.islinear():
            raise ValueError("This function is intended for linear grids only.")

        a, b = self.extent
        lg = len(self)
        return (value-a)/(b-a) * lg - 0.5

    def half_resolution(self):
        '''
        removes every second grid_node.
        '''
        grid_node = self.grid_node[::2]
        grid = 0.5 * (self.grid[:-1:2] + self.grid[1::2])
        ret = type(self)(self.name, self.unit, grid=grid, grid_node=grid_node)
        return ret

    def _extent_to_slice(self, extent):
        a, b = extent
        if a is None:
            a = self._grid_node[0]
        if b is None:
            b = self._grid_node[-1]
        return slice(*np.searchsorted(self.grid, np.sort([a, b])))

    def _normalize_slice(self, index):
        """
        Applies some checks and transformations to the object passed
        to __getitem__
        """
        if isinstance(index, slice):
            if any(helper.is_non_integer_real_number(x) for x in (index.start, index.stop)):
                if index.step is not None:
                    raise IndexError('Non-Integer slices should have step == None')
                return self._extent_to_slice((index.start, index.stop))
            return index
        else:
            if helper.is_non_integer_real_number(index):
                index = helper.find_nearest_index(self.grid, index)
            return slice(index, index+1)

    def __getitem__(self, key):
        """
        Returns an Axis which consists of a sub-part of this object defined by
        a slice containing floats or integers or a float or an integer
        """
        sl = self._normalize_slice(key)
        if sl.step is not None:
            raise ValueError("Slices with step!=None not supported")
        grid = self.grid[sl]
        sln = slice(sl.start if sl.start else None, sl.stop+1 if sl.stop else None)
        grid_node = self.grid_node[sln]
        ax = type(self)(self.name, self.unit, grid=grid, grid_node=grid_node)
        return ax

    def __len__(self):
        return self._n

    def __str__(self):
        return '<Axis "' + str(self.name) + '" (' + str(len(self)) + ' grid points)'


def _updatename(operator, reverse=False):
    def ret(func):
        @functools.wraps(func)
        def f(s, o):
            res = func(s, o)
            try:
                (a, b) = (o, s) if reverse else (s, o)
                res.name = a.name + ' ' + operator + ' ' + b.name
            except AttributeError:
                pass
            return res
        return f
    return ret


class Field(object):
    '''
    The Field Object carries a data matrix together with as many Axis
    Objects as the data matrix's dimensions. Additionaly the Field object
    provides any information that is necessary to plot _and_ annotate
    the plot. It will also suggest a content based filename for saving.

    {x,y,z}edges can be the edges or grid_nodes given for each dimension. This is
    made to work with np.histogram oder np.histogram2d.
    '''

    def __init__(self, matrix, xedges=None, yedges=None, zedges=None, name='', unit=''):
        if xedges is not None:
            self._matrix = np.asarray(matrix)  # dont sqeeze. trust numpys histogram functions.
        else:
            self._matrix = np.squeeze(matrix)
        self.name = name
        self.unit = unit
        self.axes = []
        self.infostring = ''
        self.infos = []
        self._label = None  # autogenerated if None
        if xedges is not None:
            self._addaxisnodes(xedges, name='x')
        elif self.dimensions > 0:
            self._addaxis((0, 1), name='x')
        if yedges is not None:
            self._addaxisnodes(yedges, name='y')
        elif self.dimensions > 1:
            self._addaxis((0, 1), name='y')
        if zedges is not None:
            self._addaxisnodes(zedges, name='z')
        elif self.dimensions > 2:
            self._addaxis((0, 1), name='z')

        # Additions due to FFT capabilities

        # self.axes_transform_state is False for axes which live in spatial domain
        # and it is True for axes which live in frequency domain
        # This assumes that fields are initially created in spatial domain.
        self.axes_transform_state = [False] * len(self.shape)

        # self.transformed_axes_origins stores the starting values of the grid
        # from before the last transform was executed, this is used to
        # recreate the correct axis interval upon inverse transform
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
        for k in ['infos', 'axes_transform_state', 'transformed_axes_origins']:
            # copy iterables one level deeper
            # but matrix is not copied!
            ret.__dict__[k] = copy.copy(self.__dict__[k])
        # create shallow copies of Axis objects
        ret.axes = [copy.copy(ret.axes[i]) for i in range(len(ret.axes))]
        return ret

    def __array__(self, dtype=None):
        '''
        will be called by numpy function in case an numpy array is needed.
        '''
        return np.asarray(self.matrix, dtype=dtype)

    # make sure that np.array() * Field() returns a Field and not a plain array
    __array_priority__ = 1

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
        ax = Axis(**kwargs, grid_node=grid_node)
        self._addaxisobj(ax)
        return

    def _addaxis(self, extent, **kwargs):
        '''
        adds a new axis that is supported by the matrix.
        '''
        matrixpts = self.shape[len(self.axes)]
        ax = Axis(**kwargs, extent=extent, n=matrixpts)
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
    def extent(self):
        '''
        returns the extents in a linearized form,
        as required by "matplotlib.pyplot.imshow".
        '''
        return np.ravel([a.extent for a in self.axes])

    @extent.setter
    def extent(self, newextent):
        '''
        sets the new extent to the specific values
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
        returns the grid spacings for all axis
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

    def conj(self):
        return self.replace_data(np.conjugate(self))

    def replace_data(self, other):
        ret = copy.copy(self)
        ret.matrix = other
        return ret

    def pad(self, pad_width, mode='constant', **kwargs):
        '''
        Pads the matrix using np.pad and takes care of the axes.
        See documentation of np.pad.

        In contrast to np.pad, pad_width may be given as integers, which will be interpreted
        as pixels, or as floats, which will be interpreted as distance along the appropriate axis.

        All other parameters are passed to np.pad unchanged.
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
        every second grid_node and averaging every second data point into one.

        if there is an odd number of grid points, the last point will
        be ignored. (that means, the extent will change by the size of
        the last grid cell)
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
        transform and applies the jacobian to the data.

        Please note that no interpolation is applied to the data, instead a non-linear
        axis grid is produced. If you want to interpolate the data to a new (linear) grid,
        use the method map_coordinates instead.

        In contrast to map_coordinates the function transform is not used to pull the new data
        points from the old grid, but is directly applied to the axis. This reverses the
        direction of the transform. In this case, in order to preserve the integral,
        it is necessary to divide by the Jacobian.

        axis: the index or name of the axis you want to apply transform to

        transform: the transformation function which takes the old coordinates as an input
        and returns the new grid

        preserve_integral: Divide by the jacobian of transform, in order to preserve the
        integral.

        jacobian_func: If given, this is expected to return the derivative of transform.
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
        The complex_mode specifies how to proceed with complex data:
         *  complex_mode = 'cartesian' - interpolate real/imag part (fastest)

         *  complex_mode = 'polar' - interpolate abs/phase
         If skimage.restoration is available, the phase will be unwrapped first (default)

         *  complex_mode = 'polar-no-unwrap' - interpolate abs/phase
         Skip unwrapping the phase, even if skimage.restoration is available

        preserve_integral: If True (the default), the data will be multiplied with the
        Jacobian determinant of the coordinate transformation such that the integral
        over the data will be preserved.

        In general, you will want to do this, because the physical unit of the new Field will
        correspond to the new axis of the Fields. Please note that Postpic, currently, does not
        automatically change the unit members of the Axis and Field objects, this you will have
        to do manually.

        There are, however, exceptions to this rule. Most prominently, if you are converting to
        polar coordinates it depends on what you are going to do with the transformed Field.
        If you intend to do a Cartesian r-theta plot or are interested in a lineout for a single
        value of theta, you do want to apply the Jacobian determinant. If you had a density in
        e.g. J/m^2 than, in polar coordinates, you want to have a density in J/m/rad.
        If you intend, on the other hand, to do a polar plot, you do not want to apply the
        Jacobian. In a polar plot, the data points are plotted with variable density which
        visually takes care of the Jacobian automatically. A polar plot of the polar data
        should look like a Cartesian plot of the original data with just a peculiar coordinate
        grid drawn over it.

        jacobian_determinant_func: a callable that returns the jacobian determinant of
        the transform. If given, this takes precedence over the following option.

        jacobian_func: a callable that returns the jacobian of the transform. If this is
        not given, the jacobian is numerically approximated.

        Additional keyword arguments are passed to scipy.ndimage.map_coordinates,
        see the documentation for that function.
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
        ret.axes = newaxes
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
        Transform the Field to new coordinates

        newaxes: The new axes of the new coordinates

        transform: a callable that takes the new coordinates as input and returns
        the old coordinates from where to sample the Field.
        It is basically the inverse of the transformation that you want to perform.
        If transform is not given, the identity will be used. This is suitable for
        simple interpolation to a new extent/shape.

        Example for cartesian -> polar:

        def T(r, theta):
            x = r*np.cos(theta)
            y = r*np.sin(theta)
            return x, y

        Note that this function actually computes the cartesian coordinates from the polar
        coordinates, but stands for transforming a field in cartesian coordinates into a
        field in polar coordinates.

        However, in order to preserve the definite integral of
        the field, it is necessary to multiply with the Jacobian determinant of T.

        $$
        \tilde{U}(r, \theta) = U(T(r, \theta)) \cdot \det
        \frac{\partial (x, y)}{\partial (r, \theta)}
        $$

        such that

        $$
        \int_V \mathop{\mathrm{d}x} \mathop{\mathrm{d}y} U(x,y) =
        \int_{T^{-1}(V)} \mathop{\mathrm{d}r}\mathop{\mathrm{d}\theta} \tilde{U}(r,\theta)\,.
        $$
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
        only keeps that part of the matrix, that belongs to newextent.
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
        removes axes that have length 1, reducing self.dimensions
        '''
        ret = copy.copy(self)
        retained_axes = [i for i in range(self.dimensions) if len(self.axes[i]) > 1]

        ret.axes = [self.axes[i] for i in retained_axes]
        ret.axes_transform_state = [self.axes_transform_state[i] for i in retained_axes]
        ret.transformed_axes_origins = [self.transformed_axes_origins[i] for i in retained_axes]

        ret._matrix = np.squeeze(ret.matrix)
        assert tuple(len(ax) for ax in ret.axes) == ret.shape
        return ret

    def transpose(self, *axes):
        '''
        transpose method equivalent to numpy.ndarray.transpose. If axes is empty, the order of the
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
        Swaps the axes `axis1` and `axis2`, equivalent to the numpy function with the same name.
        '''
        axes = list(range(self.dimensions))
        axes[axis1] = axis2
        axes[axis2] = axis1
        return self.transpose(*axes)

    def mean(self, axis=-1):
        '''
        takes the mean along the given axis.
        '''
        ret = copy.copy(self)
        if self.dimensions == 0:
            return self
        ret._matrix = np.mean(ret.matrix, axis=axis)
        ret.axes.pop(axis)
        ret.transformed_axes_origins.pop(axis)
        ret.axes_transform_state.pop(axis)

        return ret

    def _integrate_constant(self, axes):
        if not self.islinear():
            raise ValueError("Using method='constant' in integrate which is only suitable "
                             "for linear grids.")

        ret = self
        V = 1

        for axis in reversed(sorted(axes)):
            V *= ret.axes[axis].physical_length
            ret = ret.mean(axis)

        return V * ret

    def _integrate_scipy(self, axes, method):
        ret = copy.copy(self)
        for axis in reversed(sorted(axes)):
            ret._matrix = method(ret, ret.axes[axis].grid, axis=axis)
            del ret.axes[axis]

        return ret

    def integrate(self, axes=None, method=scipy.integrate.simps):
        '''
        Calculates the definite integral along the given axes.

        method: Choose the method to use. Available options:

        'constant' or any function with the same signature as scipy.integrate.simps
        '''
        if not callable(method) and method != 'constant':
            raise ValueError("Requested method {} is not supported".format(method))

        if axes is None:
            axes = range(self.dimensions)

        if not isinstance(axes, collections.Iterable):
            axes = (axes,)

        if method == 'constant':
            return self._integrate_constant(axes)
        else:
            return self._integrate_scipy(axes, method)

    def _transform_state(self, axes=None):
        """
        Returns the collective transform state of the given axes

        If all mentioned axis i have self.axes_transform_state[i]==True return True
        (All axes live in frequency domain)
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
        Automatically pad the array to a size such that computing its FFT using FFTW will be
        quick.

        The default for keyword argument `fft_padsize` is a callable, that is used to calculate
        the padded size for a given size.

        By default, this uses `fft_padsize=helper.fftw_padsize` which finds the next larger "good"
        grid size according to what the FFTW documentation says.

        However, the FFTW documentation also says:
        "(...) Transforms whose sizes are powers of 2 are especially fast."

        If you don't worry about the extra padding, you can pass
        `fft_padsize=helper.fft_padsize_power2` and this method will pad to the next power of 2.
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
            if self.transformed_axes_origins[i]:
                new_axes[i] += self.transformed_axes_origins[i] - new_axes[i][0]
        return new_axes

    def fft(self, axes=None, exponential_signs='spatial', **kwargs):
        '''
        Performs Fourier transform on any number of axes.

        The argument axis is either an integer indicating the axis to be transformed
        or a tuple giving the axes that should be transformed. Automatically determines
        forward/inverse transform. Transform is only applied if all mentioned axes are
        in the same space. If an axis is transformed twice, the origin of the axis is restored.

        exponential_signs configures the sign convention of the exponential
        exponential_signs == 'spatial':  fft using exp(-ikx), ifft using exp(ikx)
        exponential_signs == 'temporal':  fft using exp(iwt), ifft using exp(-iwt)

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
        Makes sure that the field has the given transform_states. `transform_states` might be
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
            ret.axes[i].grid_node = self.axes[i].grid_node + dx[i]

        return ret

    def shift_grid_by(self, dx, interpolation='fourier'):
        '''
        Translate the Grid by dx.
        This is useful to remove the grid stagger of field components.

        If all axis will be shifted, dx may be a list.
        Otherwise dx should be a mapping from axis to translation distance.

        The keyword-argument interpolation indicates the method to be used and
        may be one of ['linear', 'fourier'].
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

    @helper.append_doc_of(_map_coordinates)
    def topolar(self, extent=None, shape=None, angleoffset=0, **kwargs):
        '''
        Transform the Field to polar coordinates.

        This is a convenience wrapper for map_coordinates which will let you easily
        define the desired grid in polar coordinates via the arguments

        * extent,
        which should be of the form extent=(phimin, phimax, rmin, rmax) or
        extent=(phimin, phimax),
        * shape,
        which should be of the form shape=(N_phi, N_r),
        * angleoffset,
        which can be any real number and will rotate the zero-point of the angular axis.
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

    def exporttocsv(self, filename):
        if self.dimensions == 1:
            data = np.asarray(self.matrix)
            x = np.linspace(self.extent[0], self.extent[1], len(data))
            np.savetxt(filename, np.transpose([x, data]), delimiter=' ')
        elif self.dimensions == 2:
            export = np.asarray(self.matrix)
            np.savetxt(filename, export)
        else:
            raise Exception('Not Implemented')
        return

    def __str__(self):
        return '<Feld "' + self.name + '" ' + str(self.shape) + '>'

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

    @_updatename('+')
    def __iadd__(self, other):
        self.matrix += np.asarray(other)
        return self

    def __add__(self, other):
        ret = copy.copy(self)
        ret.matrix = ret.matrix + np.asarray(other)
        return ret
    __radd__ = _updatename('+', reverse=True)(__add__)
    __add__ = _updatename('+', reverse=False)(__add__)

    def __neg__(self):
        ret = copy.copy(self)
        ret.matrix = -self.matrix
        ret.name = '-' + ret.name
        return ret

    @_updatename('-')
    def __isub__(self, other):
        self.matrix -= np.asarray(other)
        return self

    @_updatename('-')
    def __sub__(self, other):
        ret = copy.copy(self)
        ret.matrix = ret.matrix - np.asarray(other)
        return ret

    @_updatename('-', reverse=True)
    def __rsub__(self, other):
        ret = copy.copy(self)
        ret.matrix = np.asarray(other) - ret.matrix
        return ret

    @_updatename('^')
    def __pow__(self, other):
        ret = copy.copy(self)
        ret.matrix = self.matrix ** np.asarray(other)
        return ret

    @_updatename('^', reverse=True)
    def __rpow__(self, other):
        ret = copy.copy(self)
        ret.matrix = np.asarray(other) ** self.matrix
        return ret

    @_updatename('*')
    def __imul__(self, other):
        self.matrix *= np.asarray(other)
        return self

    def __mul__(self, other):
        ret = copy.copy(self)
        ret.matrix = ret.matrix * np.asarray(other)
        return ret
    __rmul__ = _updatename('*', reverse=True)(__mul__)
    __mul__ = _updatename('*', reverse=False)(__mul__)

    def __abs__(self):
        ret = copy.copy(self)
        ret.matrix = np.abs(ret.matrix)
        ret.name = '|{}|'.format(ret.name)
        return ret

    @_updatename('/')
    def __itruediv__(self, other):
        self.matrix /= np.asarray(other)
        return self

    @_updatename('/')
    def __truediv__(self, other):
        ret = copy.copy(self)
        ret.matrix = ret.matrix / np.asarray(other)
        return ret

    @_updatename('/', reverse=True)
    def __rtruediv__(self, other):
        ret = copy.copy(self)
        ret.matrix = np.asarray(other) / ret.matrix
        return ret

    # python 2
    __idiv__ = __itruediv__
    __div__ = __truediv__
