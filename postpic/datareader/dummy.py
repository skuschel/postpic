'''
Dummy reader for creating fake simulation Data for testing purposes.

Stephan Kuschel 2014
'''

from . import Dumpreader_ifc
from . import Simulationreader_ifc
import numpy as np


class Dummyreader(Dumpreader_ifc):
    '''
    Dummyreader creates fake Data for testing purposes.
    '''

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
        xx, yy = np.meshgrid(self.grid('x'), self.grid('y'))
        if self._axesidentify[axis] == 0:
            ret = np.sin(self.timestep() * xx)
        elif self._axesidentify[axis] == 1:
            ret = np.cos(xx + yy**2)
        elif self._axesidentify[axis] == 2:
            ret = xx * 0
        return ret

    def dataB(self, axis):
        return 10 * self.dataE(axis)

    def grid(self, axis):
        '''
        returns the grid points for the axis specified.
        Thus only regular grids are supported currently.
        '''
        if self._axesidentify[axis] == 0:  # x-axis
            ret = np.linspace(0, 2*np.pi, 100)
        elif self._axesidentify[axis] == 1:  # y-axis
            ret = np.linspace(-5, 10, 100)
        else:  # no z-axis present, since simdimensions() returns 2.
            raise IndexError('axis ' + str(axis) + ' not present.')
        return ret

    def listSpecies(self):
        pass

    def getSpecies(self, species, attrib):
        pass

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
























