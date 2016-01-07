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
# Stephan Kuschel 2014
'''
The Datareader package contains methods and interfaces to read data from
any Simulation.

The basic concept consits of two different types of readers:

The Dumpreader
--------------
This has to be subclassed from Dumpreader_ifc and allows to read a single dump
created by the simulation. To identify which dump should be read its
initialized with a dumpidentifier. This dumpidentifier can be almost anything,
but in the easiest case this is the filepath pointing to a single file
containing every information about this simulation dump. With this information
the dumpreader must be able to read all data regarding this dump (which is a
lot: X, Y, Z, Px, Py, Py, weight, mass, charge, ID,.. for all particle species,
electric and magnetic fields on grid, the grid itself, mabe particle ids,...)

The Simulationreader
--------------------
This has to be subclassed from Simulationreader_ifc and allows to read a full
list of simulation dumps. Thus an alternate Name for this class could be
"Dumpsequence". This allows the code to track particles from different times
of the simulation or create plots with a time axis.

Stephan Kuschel 2014
'''
from __future__ import absolute_import, division, print_function, unicode_literals

from .datareader import *

__all__ = ['chooseCode', 'readDump', 'readSim']
__all__ += datareader.__all__

_dumpreadercls = None
_simreadercls = None


def setdumpreadercls(dumpreadercls):
    '''
    Sets the class that is used for reading dumps.
    dumpreadercls needs to be subclass of "Dumpreader_ifc".
    '''
    if issubclass(dumpreadercls, Dumpreader_ifc):
        global _dumpreadercls
        _dumpreadercls = dumpreadercls
    else:
        raise Exception('In order to set a reader class for a new file'
                        ' format it needs to be subclass of "Dumpreader_ifc"')


def setsimreadercls(simreadercls):
    '''
    Sets the class that is used for reading dumps.
    simreadercls needs to be subclass of "Simulationreader_ifc".
    '''
    if issubclass(simreadercls, Simulationreader_ifc):
        global _simreadercls
        _simreadercls = simreadercls
    else:
        raise Exception('In order to set a reader for a new file'
                        ' format it needs to be subclass of '
                        '"Simulationreader_ifc"')


def readDump(dumpidentifier, **kwargs):
    global _dumpreadercls
    if _dumpreadercls is None:
        raise Exception('Specify dumpreaderclass first.')
    return _dumpreadercls(dumpidentifier, **kwargs)


def readSim(simidentifier, **kwargs):
    global _simreadercls
    if _simreadercls is None:
        raise Exception('Specify simreaderclass first.')
    return _simreadercls(simidentifier, **kwargs)


def chooseCode(code):
    '''
    Chooses appropriate reader for the given simulation code.

    Args:
      code : string
        Possible options are:
          - "DUMMY": dummy class creating fake data.
          - "EPOCH": .sdf files written by EPOCH1D, EPOCH2D or EPOCH3D.
          - "openPMD": .h5 files written in openPMD Standard
          - "VSIM": .hdf5 files written by VSim.
    '''
    if code.lower() in ['epoch', 'epoch1d', 'epoch2d', 'epoch3d']:
        from .epochsdf import Sdfreader, Visitreader
        setdumpreadercls(Sdfreader)
        setsimreadercls(Visitreader)
    elif code.lower() in ['openpmd', 'openpmdh5']:
        from .openPMDh5 import OpenPMDreader, FileSeries
        setdumpreadercls(OpenPMDreader)
        setsimreadercls(FileSeries)
    elif code.lower() in ['vsim']:
        raise Exception('VSim reader requires update due to the interface change in '
                        'https://github.com/skuschel/postpic/commit/'
                        'c3d5b9d7afda3b3b0ebf57cd3199567a5a494803')
        from .vsimhdf5 import Hdf5reader, VSimReader
        setdumpreadercls(Hdf5reader)
        setsimreadercls(VSimReader)
    elif code.lower() in ['dummy']:
        from .dummy import Dummyreader, Dummysim
        setdumpreadercls(Dummyreader)
        setsimreadercls(Dummysim)
    else:
        raise TypeError('Code "' + str(code) + '" not recognized.')














