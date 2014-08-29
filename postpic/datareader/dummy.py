'''
Dummy reader for creating fake simulation Data for testing purposes.

Stephan Kuschel 2014
'''

from . import Dumpreader_ifc
from . import Simulationreader_ifc
import numpy as np
from .. import _const


class Dummyreader(Dumpreader_ifc):
    '''
    Dummyreader creates fake Data for testing purposes.
    '''

    def __init__(self, dumpid):
        super(self.__class__, self).__init__(dumpid)
        # initialize fake data
        self._xdata = np.random.normal(size=dumpid)
        self._ydata = np.random.normal(size=dumpid)

    def keys(self):
        pass

    def __getitem__(self, key):
        pass

    def timestep(self):
        return self.dumpidentifier

    def time(self):
        return self.timestep() * 1e-10

    def simdimensions(self):
        return 2

    def dataE(self, axis):
        xx, yy = np.meshgrid(self.grid('y'), self.grid('x'))
        if _const.axesidentify[axis] == 0:
            ret = np.sin(self.timestep() * xx)
        elif _const.axesidentify[axis] == 1:
            ret = np.cos(xx + yy ** 2)
        elif _const.axesidentify[axis] == 2:
            ret = xx * 0
        return ret

    def dataB(self, axis):
        return 10 * self.dataE(axis)

    def grid(self, axis):
        '''
        returns the grid points for the axis specified.
        Thus only regular grids are supported currently.
        '''
        if _const.axesidentify[axis] == 0:  # x-axis
            ret = np.linspace(0, 2 * np.pi, 100)
        elif _const.axesidentify[axis] == 1:  # y-axis
            ret = np.linspace(-5, 10, 200)
        else:  # no z-axis present, since simdimensions() returns 2.
            raise IndexError('axis ' + str(axis) + ' not present.')
        return ret

    def listSpecies(self):
        return ['electron']

    def getSpecies(self, species, attrib):
        attribid = _const.attribidentify[attrib]
        if attribid == 0:  # x
            ret = self._xdata
        elif attribid == 1:  # y
            ret = self._ydata
        elif attribid == 3:  # px
            ret = self._xdata ** 2
        elif attribid == 4:  # py
            ret = self._ydata ** 2
        elif attribid == 5:  # pz
            ret = self._ydata * self._xdata
        elif attribid == 9:  # weights
            ret = np.repeat(1, len(self._xdata))
        else:
            ret = None
        return ret

    def __str__(self):
        return '<Dummyreader initialized with "' \
               + str(self.dumpidentifier) + '">'


class Dummysim(Simulationreader_ifc):

    def __init__(self, simidentifier):
        super(self.__class__, self).__init__(simidentifier)

    def __len__(self):
        return self.simidentifier

    def getDumpnumber(self, index):
        if index < len(self):
            return Dummyreader(index)
        else:
            raise IndexError()

    def __str__(self):
        return '<Dummysimulation initialized with "' \
               + str(self.simidentifier) + '">'
























