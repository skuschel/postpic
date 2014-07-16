'''
Reader for SDF File format written by the EPOCH Code:
http://ccpforge.cse.rl.ac.uk/gf/project/epoch/
Dependecies:
sdf   The Python actual reader for sdf file format.
      It is part of the EPOCH code base and needs to be
      compiled and installed from there.
Stephan Kuschel 2014
'''

from . import Dumpreader_ifc
from . import Simulationreader_ifc
import numpy as np
import re

__all__ = ['Sdfreader', 'Visitreader']


class Sdfreader(Dumpreader_ifc):
    '''
    The Reader implementation for Data written by the EPOCH Code
    in .sdf format.
    '''

    def __init__(self, sdffile):
        '''
        Initializes the sdfreader for a specific sdffile.
        '''
        super(self.__class__, self).__init__(sdffile)
        import os.path
        import sdf
        if not os.path.isfile(sdffile):
            raise IOError('File "' + str(sdffile) + '" doesnt exist.')
        self._data = sdf.SDF(sdffile).read()

    def keys(self):
        return self._data.keys()

    def __getitem__(self, key):
        return self._data[key]

    def timestep(self):
        return self['Header']['step']

    def time(self):
        return np.float64(self['Header']['time'])

    def simdimensions(self):
        return float(re.match('Epoch(\d)d',
                     self['Header']['code_name']).group(1))

    def _returnkey2(self, key1, key2, average=False):
        key = key1 + key2
        if average:
            key = key1 + '_average' + key2
        return self[key]

    def dataE(self, axis, **kwargs):
        axsuffix = {0: 'x', 1: 'y', 2: 'z'}[self._axesidentify[axis]]
        return np.float64(self._returnkey2('Electric Field', '/E' +
                                           axsuffix, **kwargs))

    def dataB(self, axis):
        axsuffix = {0: 'x', 1: 'y', 2: 'z'}[self._axesidentify[axis]]
        return np.float64(self._returnkey2('Magnetic Field', '/B' +
                                           axsuffix, **kwargs))

    def grid(self, axis):
        axsuffix = {0: 'X', 1: 'Y', 2: 'Z'}[self._axesidentify[axis]]
        return self['Grid/Grid/' + axsuffix]

    def listSpecies(self):
        ret = []
        for key in self.keys():
            match = re.match('Particles/Px/(\w+)', key)
            if match:
                ret = np.append(ret, match.group(1))
        ret.sort()
        return ret

    def getSpecies(self, species, attrib):
        """
        Returns one of the attributes out of (x,y,z,px,py,pz,weight,ID) of
        this particle species.
        returning None means that this particle property wasnt dumped.
        Note that this is different from returning an empty list!
        """
        attribid = self._attribidentify[attrib]
        options = {9: lambda s: 'Particles/Weight/' + s,
                   0: lambda s: 'Grid/Particles/' + s + '/X',
                   1: lambda s: 'Grid/Particles/' + s + '/Y',
                   2: lambda s: 'Grid/Particles/' + s + '/Z',
                   3: lambda s: 'Particles/Px/' + s,
                   4: lambda s: 'Particles/Py/' + s,
                   5: lambda s: 'Particles/Pz/' + s,
                   10: lambda s: 'Particles/ID/' + s}
        try:
            ret = np.float64(self[options[attribid](spezies)])
        except(KeyError):
            ret = None
        return ret

    def __str__(self):
        return '<Sdfreader at "' + str(self.dumpidentifier) + '">'


class Visitreader(Simulationreader_ifc):
    '''
    Reads a series of dumps specified in a .visit file. This is specifically
    written for .visit files from the EPOCH code, but should also work for
    any other code using these files.
    '''

    def __init__(self, visitfile):
        super(self.__class__, self).__init__(visitfile)
        self.visitfile = visitfile
        import os.path
        if not os.path.isfile(visitfile):
            raise IOError('File "' + str(visitfile) + '" doesnt exist.')
        self._dumpfiles = []
        with open(visitfile) as f:
            relpath = os.path.dirname(visitfile)
            for line in f:
                self._dumpfiles.append(os.path.join(relpath,
                                       line.replace('\n', '')))

    def __len__(self):
        return len(self._dumpfiles)

    def getDumpnumber(self, index):
        from . import readDump
        return readDump(self._dumpfiles[index])

    def __str__(self):
        return '<Visitreader at "' + self.visitfile + '">'
























