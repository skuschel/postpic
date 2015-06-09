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
# Copyright Stephan Kuschel 2014, 2015
'''
.. _EPOCH: https://cfsa-pmw.warwick.ac.uk/EPOCH/epoch
.. _SDF: https://github.com/keithbennett/SDF

Reader for SDF_ File format written by the EPOCH_ Code.

Dependecies:
  - sdf: The actual python reader for the .sdf file format written in C.
    It is part of the EPOCH_ code base and needs to be
    compiled and installed from there.

Written by Stephan Kuschel 2014, 2015
'''

from . import Dumpreader_ifc
from . import Simulationreader_ifc
import numpy as np
import re
from .. import helper

__all__ = ['Sdfreader', 'Visitreader']


class Sdfreader(Dumpreader_ifc):
    '''
    The Reader implementation for Data written by the EPOCH_ Code
    in .sdf format. Written for SDF v2.2.0 or higher.
    SDF_ can be obtained without EPOCH_ from SDF_.

    Args:
      sdffile : String
        A String containing the relative Path to the .sdf file.
    '''

    def __init__(self, sdffile, **kwargs):
        super(self.__class__, self).__init__(sdffile, **kwargs)
        import os.path
        import sdf
        try:
            sdfversion = sdf.__version__
        except(AttributeError):
            sdfversion = '0.0.0'
        if sdfversion < '2.2.0':
            raise ImportError('Upgrade sdf package to 2.2.0 or higher.')
        if not os.path.isfile(sdffile):
            raise IOError('File "' + str(sdffile) + '" doesnt exist.')
        self._data = sdf.read(sdffile, dict=True)

    def keys(self):
        return self._data.keys()

    def __getitem__(self, key):
        return self._data[key]

    def getdata(self, key):
        return self[key].data

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
        return self[key].data

    def dataE(self, axis, **kwargs):
        axsuffix = {0: 'x', 1: 'y', 2: 'z'}[helper.axesidentify[axis]]
        return np.float64(self._returnkey2('Electric Field', '/E' +
                                           axsuffix, **kwargs))

    def dataB(self, axis, **kwargs):
        axsuffix = {0: 'x', 1: 'y', 2: 'z'}[helper.axesidentify[axis]]
        return np.float64(self._returnkey2('Magnetic Field', '/B' +
                                           axsuffix, **kwargs))

    def grid(self, axis):
        try:
            return self['Grid/Grid_mid'].data[helper.axesidentify[axis]]
        except(IndexError):
            raise KeyError

    def listSpecies(self):
        ret = set()
        for key in self.keys():
            match = re.match('Particles/\w+/(\w+(/\w+)?)', key)
            if match:
                ret.add(match.group(1))
        ret = list(ret)
        ret.sort()
        return ret

    def getSpecies(self, species, attrib):
        """
        Returns one of the attributes out of (x,y,z,px,py,pz,weight,ID) of
        this particle species.
        returning None means that this particle property wasnt dumped.
        Note that this is different from returning an empty list!
        """
        attribid = helper.attribidentify[attrib]
        options = {9: lambda s: self['Particles/Weight/' + s].data,
                   0: lambda s: self['Grid/Particles/' + s].data[0],
                   1: lambda s: self['Grid/Particles/' + s].data[1],
                   2: lambda s: self['Grid/Particles/' + s].data[2],
                   3: lambda s: self['Particles/Px/' + s].data,
                   4: lambda s: self['Particles/Py/' + s].data,
                   5: lambda s: self['Particles/Pz/' + s].data,
                   10: lambda s: self['Particles/ID/' + s].data}
        try:
            ret = np.float64(options[attribid](species))
        except(IndexError):
            raise KeyError
        return ret

    def getderived(self):
        '''
        Returns all Keys starting with "Derived/".
        '''
        ret = []
        for key in self._data.keys():
            r = re.match('Derived/[\w/ ]*', key)
            if r:
                ret.append(r.group(0))
        ret.sort()
        return ret

    def __str__(self):
        return '<Sdfreader at "' + str(self.dumpidentifier) + '">'


class Visitreader(Simulationreader_ifc):
    '''
    Reads a series of dumps specified in a .visit file. This is specifically
    written for .visit files from the EPOCH_ code, but should also work for
    any other code using these files.
    '''

    def __init__(self, visitfile, dumpreadercls=Sdfreader, **kwargs):
        super(self.__class__, self).__init__(visitfile, **kwargs)
        self.visitfile = visitfile
        self.dumpreadercls = dumpreadercls
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

    def getDumpreader(self, index):
        return self.dumpreadercls(self._dumpfiles[index])

    def __str__(self):
        return '<Visitreader at "' + self.visitfile + '">'
























