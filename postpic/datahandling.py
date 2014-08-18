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

import numpy as np
import copy
from . import _const

__all__=['Field', 'Axis']

class Axis(object):
    '''
    Axis handling for a single Axis.
    '''

    def __init__(self, name='', unit=''):
        self.name = name
        self.unit = unit
        self._grid_node = np.array([])
        self._linear = None

    def islinear(self, force=False):
        """
        Checks if the axis has a linear grid.
        """
        if self._linear is None or force:
            self._linear = np.var(np.diff(self._grid_node)) < 1e-7
        return self._linear

    @property
    def grid_node(self):
        return self._grid_node

    @grid_node.setter
    def grid_node(self, value):
        gn = np.float64(value)
        if len(gn.shape) != 1:
            raise TypeError('Only 1 dimensional arrays can be assigend.')
        self._grid_node = gn
        self._linear = None

    @property
    def grid(self):
        return np.convolve(self.grid_node, np.ones(2) / 2.0, mode='valid')

    @grid.setter
    def grid(self, grid):
        gn = np.convolve(grid, np.ones(2) / 2.0, mode='full')
        gn[0] = grid[0] + (grid[0] - gn[1])
        gn[-1] = grid[-1] + (grid[-1] - gn[-2])
        self.grid_node = gn

    @property
    def extent(self):
        if len(self._grid_node) < 2:
            ret = None
        else:
            return [self._grid_node[0], self._grid_node[-1]]

    def setextent(self, extent, n):
        '''
        creates a linear grid with the given extent and n grid points
        (thus n+1 grid_node)
        '''
        if n == 1 and type(extent) is int:
            gn = np.array([extent - 0.5, extent + 0.5])
        else:
            gn = np.linspace(extent[0], extent[-1], n + 1)
        self.grid_node = gn

    def cutout(self, newextent):
        '''
        keeps the grid points within the newextent only.
        '''
        nex = np.sort(newextent)
        gnnew = [gn for gn in self.grid_node
                 if (nex[0] <= gn and gn <= nex[1])]
        self.grid_node = gnnew

    def half_resolution(self):
        '''
        removes every second grid_node.
        '''
        self.grid_node = self.grid_node[::2]

    def __len__(self):
        return len(self._grid_node) - 1

    def __str__(self):
        return '<Axis "' + str(self.name) + \
               '" (' + str(len(self)) + ' grid points)'


class Field(object):
    '''
    The Field Object carries a data matrix together with as many Axis
    Objects as the data matrix's dimensions. Additionaly the Field object
    provides any information that is necessary to plot _and_ annotate
    the plot. It will also suggest a content based filename for saving.
    '''

    def __init__(self, matrix, name='', unit=''):
        self.matrix = np.float64(np.squeeze(matrix))
        self.name = name
        self.unit = unit
        self.axes = []
        if self.dimensions > 0:
            self._addaxis((0, 1), name='x')
        if self.dimensions > 1:
            self._addaxis((0, 1), name='y')
        if self.dimensions > 2:
            self._addaxis((0, 1), name='z')

    def _addaxisobj(self, axisobj):
        '''
        uses the given axisobj as the axis obj in the given dimension.
        '''
        # check if number of grid points match
        matrixpts = self.matrix.shape[len(self.axes)]
        if matrixpts != len(axisobj):
            raise ValueError(
                'Number of Grid points in next missing Data '
                'Dimension ({:d}) has to match number of grid points of '
                'new axis ({:d})'.format(matrixpts, len(axisobj)))
        self.axes.append(axisobj)

    def _addaxis(self, extent, **kwargs):
        '''
        adds a new axis that is supported by the matrix.
        '''
        matrixpts = self.matrix.shape[len(self.axes)]
        ax = Axis(**kwargs)
        ax.setextent(extent, matrixpts)
        self._addaxisobj(ax)

    def islinear(self):
        return [a.islinear() for a in self.axes]

    @property
    def shape(self):
        return self.matrix.shape

    @property
    def grid_nodes(self):
        return [a.grid_node for a in axes]

    @property
    def grid(self):
        return [a.grid for a in axes]

    @property
    def dimensions(self):
        '''
        returns only present dimensions.
        [] and [[]] are interpreted as -1
        np.array(2) is interpreted as 0
        np.array([1,2,3]) is interpreted as 1
        and so on...
        '''
        ret = len(self.matrix.shape)  # handle self.matrix == [] seperately
        if ret == 1 and len(self.matrix) == 0:
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
        assert self.dimensions * 2 == len(newextent), \
            'size of newextent doesnt match self.dimensions * 2'
        for i in xrange(len(self.axes)):
            self.axes[i].setextent(newextent[2*i:2*i+2], self.matrix.shape[i])

    def half_resolution(self, axis):
        '''
        Halfs the resolution along the given axis by removing
        every second grid_node and averaging every second data point into one.

        if there is an odd number of grid points, the last point will
        be ignored. (that means, the extent will change by the size of
        the last grid cell)
        '''
        axis = _const.axesidentify[axis]
        self.axes[axis].half_resolution()
        n = self.matrix.ndim
        s = [slice(None), ] * n
        # ignore last grid point if self.matrix.shape[axis] is odd
        lastpt = self.matrix.shape[axis] - self.matrix.shape[axis] % 2
        # Averaging over neighboring points
        s[axis] = slice(0, lastpt, 2)
        ret = self.matrix[s]
        s[axis] = slice(1, lastpt, 2)
        ret += self.matrix[s]
        self.matrix = ret / 2.0

    def autoreduce(self, maxlen=4000):
        '''
        Reduces the Grid to a maximum length of maxlen per dimension
        by just executing half_resolution as often as necessary.
        '''
        for i in xrange(len(self.axes)):
            if len(self.axes[i]) > maxlen:
                self.half_resolution(i)
                self.autoreduce(maxlen=maxlen)
                break
        return self

    def cutout(self, newextent):
        '''
        only keeps that part of the matrix, that belongs to newextent.
        '''
        if self.dimensions() == 0:
            return
        assert self.dimensions * 2 == len(newextent), \
            'size of newextent doesnt match self.dimensions * 2'
        self.matrix = _const.cutout(self.matrix, self.extent, newextent)
        for i in xrange(len(self.axes)):
            self.axes[i].cutout(newextent[2*i:2*i+2])

    def topolar(self, extent=None, shape=None, angleoffset=0):
        '''
        remaps the current kartesian coordinates to polar coordinates
        extent should be given as extent=(phimin, phimax, rmin, rmax)
        ''''
        ret = copy.deepcopy(self)
        if extent is None:
            extent = [-np.pi, np.pi, 0, self.extent[1]]
        extent = np.asarray(extent)
        if shape is None:
            maxpt_r = np.min((np.floor(np.min(self.matrix.shape) / 2), 1000))
            shape = (1000, maxpt_r)

        extent[0:2] = extent[0:2] - angleoffset
        ret.matrix = _const.transfromxy2polar(self.matrix, self.extent,
                                              np.roll(extent, 2), shape).T
        extent[0:2] = extent[0:2] + angleoffset

        ret.extent = extent
        if ret.axes[0].name.startswith('$k_') \
           and ret.axes[1].name.startswith('$k_'):
            ret.axes[0].name = '$k_\phi$'
            ret.axes[1].name = '$|k|$'
        return ret

    def exporttocsv(self, filename):
        if self.dimensions == 1:
            data = np.asarray(self.matrix)
            x = np.linspace(self.extent[0], self.extent[1], len(data))
            np.savetxt(dateiname, np.transpose([x, data]), delimiter=' ')
        elif self.dimensions == 2:
            export = np.asarray(self.matrix)
            np.savetxt(dateiname, export)
        else:
            raise Exception('Not Implemented')

    def __str__(self):
        return '<Feld "' + self.name + '" ' + str(self.matrix.shape) + '>'

    # Operator overloading
    def __iadd__(self, other):
        if isinstance(other, Field):
            self.matrix += other.matrix
            self.name = self.name + ' + ' + other.name
        else:
            self.matrix += other
        return self

    def __add__(self, other):
        ret = copy.deepcopy(self)
        ret += other
        return ret

    def __neg__(self):
        ret = copy.deepcopy(self)
        ret.matrix *= -1
        return ret

    def __isub__(self, other):
        if isinstance(other, Field):
            self.matrix -= other.matrix
            self.name = self.name + ' - ' + other.name
        else:
            self.matrix -= other
        return self

    def __sub__(self, other):
        ret = copy.deepcopy(self)
        ret -= other
        return ret

    def __pow__(self, other):
        ret = copy.deepcopy(self)
        ret.matrix = self.matrix ** other
        return ret

    def __imul__(self, other):
        if isinstance(other, Field):
            self.matrix *= other.matrix
            self.name = self.name + ' * ' + other.name
        else:
            self.matrix *= other
        return self

    def __mul__(self, other):
        ret = copy.deepcopy(self)
        ret *= other
        return ret

    def __abs__(self):
        ret = copy.deepcopy(self)
        ret.matrix = np.abs(ret.matrix)
        return ret

    # self /= other: normalization
    def __idiv__(self, other):
        if isinstance(other, Field):
            self.matrix /= other.matrix
            self.name = self.name + ' / ' + other.name
        else:
            self.matrix /= other
        return self

    def __div__(self, other):
        ret = copy.deepcopy(self)
        ret /= other
        return ret




