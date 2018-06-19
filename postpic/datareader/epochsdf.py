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
# Alexander Blinne, 2017
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
from __future__ import absolute_import, division, print_function, unicode_literals

from . import Dumpreader_ifc
from . import Simulationreader_ifc
import numpy as np
import re
from .. import helper
from warnings import warn

__all__ = ['Sdfreader', 'Visitreader']


# The default staggering of the Grid.
# Newer versions of EPOCH dump it (certainly v4.9.2)
# older versions dont. The defaults will only be used,
# if the stagger was not found in the dump.
_default_stagger = {'Electric Field/Ex': 1,
                    'Electric Field/Ey': 2,
                    'Electric Field/Ez': 4,
                    'Magnetic Field/Bx': 6,
                    'Magnetic Field/By': 5,
                    'Magnetic Field/Bz': 3,
                    'Current/Jx': 1,
                    'Current/Jy': 2,
                    'Current/Jz': 4}
_default_stagger.update({k+'_averaged': v for k, v in _default_stagger.items()})


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
        self._sdffile = sdffile
        self._sdfreader = sdf.read(sdffile, dict=True)

# --- Level 0 methods ---

    def keys(self):
        return list(self._sdfreader.keys())

    def __getitem__(self, key):
        return self._sdfreader[key]

    def dumpsize(self):
        '''
        returns the file size of the sdf file in bytes.
        '''
        import os
        return os.path.getsize(self._sdffile)

# --- Level 1 methods ---

    def data(self, key):
        return self[key].data

    def gridoffset(self, key, axis):
        axid = helper.axesidentify[axis]
        dx = self.gridspacing(key, axis)
        if hasattr(self[key], 'stagger'):
            # best case: stagger is saved
            stagger = self[key].stagger
        elif key in _default_stagger:
            stagger = _default_stagger[key]
        elif key.startswith('Derived/'):
            # c_stagger_cell_centre in EPOCH code
            stagger = 0
        else:
            warn('Stagger of "{:}" could not be found. \
                  Assuming no stagger (that is cell center).'.format(key))
            stagger = 0

        staggered = stagger & (1 << axid)
        if staggered:
            return self[key].grid_mid.data[axid][0] - dx/2.0
        else:
            return self[key].grid.data[axid][0] - dx/2.0

    def gridspacing(self, key, axis):
        axid = helper.axesidentify[axis]
        grid = self[key].grid
        extent = float(grid.extents[axid + len(grid.dims)] - grid.extents[axid])
        return extent / (grid.dims[axid]-1)

    def gridpoints(self, key, axis):
        axid = helper.axesidentify[axis]
        return self[key].dims[axid]

# --- Level 2 methods ---

    def timestep(self):
        return self['Header']['step']

    def time(self):
        return np.float64(self['Header']['time'])

    def simdimensions(self):
        return int(re.match(r'Epoch(\d)d', self['Header']['code_name']).group(1))

    def _keyE(self, component, average=False):
        axsuffix = {0: 'x', 1: 'y', 2: 'z'}[helper.axesidentify[component]]
        ret = 'Electric Field/E' + axsuffix
        if average:
            ret += '_averaged'
        return ret

    def _keyB(self, component, average=False):
        axsuffix = {0: 'x', 1: 'y', 2: 'z'}[helper.axesidentify[component]]
        ret = 'Magnetic Field/B' + axsuffix
        if average:
            ret += '_averaged'
        return ret

    def simextent(self, axis):
        '''
        Returns the extent of the actual simulation box.
        '''
        m = self['Grid/Grid']
        extents = m.extents
        dims = len(m.dims)
        axid = helper.axesidentify[axis]
        return np.array([extents[axid], extents[axid + dims]])

    def simgridpoints(self, axis):
        '''
        Returns the number of grid points of the actual simulation.
        '''
        mesh = self['Grid/Grid']
        axid = helper.axesidentify[axis]
        return mesh.dims[axid] - 1

    def listSpecies(self):
        ret = set()
        for key in list(self.keys()):
            match = re.match(r'Particles/\w+/([\w-]+(/[\w-]+)?)', key)
            if match:
                ret.add(match.group(1))
        ret = list(ret)
        ret.sort()
        return ret

    def getSpecies(self, species, attrib):
        """
        Returns one of the attributes out of (x,y,z,px,py,pz,weight,ID,mass,charge) of
        this particle species.
        raises KeyError if the requested species or property wasnt dumped.
        """
        attribid = helper.attribidentify[attrib]
        options = {9: lambda s: self['Particles/Weight/' + s].data,
                   0: lambda s: self['Grid/Particles/' + s].data[0],
                   1: lambda s: self['Grid/Particles/' + s].data[1],
                   2: lambda s: self['Grid/Particles/' + s].data[2],
                   3: lambda s: self['Particles/Px/' + s].data,
                   4: lambda s: self['Particles/Py/' + s].data,
                   5: lambda s: self['Particles/Pz/' + s].data,
                   10: lambda s: self['Particles/ID/' + s].data,
                   11: lambda s: self['Particles/Mass/' + s].data,
                   12: lambda s: self['Particles/Charge/' + s].data}
        try:
            ret = options[attribid](species)
        except(IndexError):
            raise KeyError('Attribute "{}" of species "{}" not found.'.format(attrib, species))
        return ret

    def getderived(self):
        '''
        Returns all Keys starting with "Derived/".
        '''
        ret = []
        for key in list(self.keys()):
            r = re.match(r'Derived/[\w/ ]*', key)
            if r:
                ret.append(r.group(0))
        ret.sort()
        return ret

    def __repr__(self):
        return '<Sdfreader at "{:}">'.format(self.dumpidentifier)


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
            path = os.path.dirname(os.path.abspath(visitfile))
            for line in f:
                self._dumpfiles.append(os.path.join(path,
                                                    line.replace('\n', '')))

    def __len__(self):
        return len(self._dumpfiles)

    def _getDumpreader(self, index):
        return self.dumpreadercls(self._dumpfiles[index])

    def __repr__(self):
        return '<Visitreader at "{:}" ({:} dumps)>'.format(self.visitfile, len(self))
