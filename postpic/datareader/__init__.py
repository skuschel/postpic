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
lot: X, Y, Z, Px, Py, Py, weight for all particle species, electric and
magnetic fields on grid, the grid itself, mabe particle ids,...)

The Simulationreader
--------------------
This has to be subclassed from Simulationreader_ifc and allows to read a full
list of simulation dumps. Thus an alternate Name for this class could be
"Dumpsequence". This allows the code to track particles from different times
of the simulation or create plots with a time axis.

Stephan Kuschel 2014
'''


import abc
import collections
import numpy as np


# --- Interface ---

class Dumpreader_ifc(object):
    '''
    Interface class for reading a single dump. A dump contains informations
    about the  simulation at a single timestep (Usually E- and B-Fields on
    grid + particles).

    Any Dumpreader_ifc implementation will always be initialized using a
    dumpidentifier. This dumpidentifier can be anything, that points to
    the data of a dump. In the easiest case this is just the filename of
    the dump holding all data of that timestep
    (for example .sdf file for EPOCH, .hdf5 for some other code).

    It is highly recommended to also override the __str__ function.
    '''
    __metaclass__ = abc.ABCMeta

    _axesidentify = {'X': 0, 'x': 0, 0: 0,
                     'Y': 1, 'y': 1, 1: 1,
                     'Z': 2, 'z': 2, 2: 2}
    _attribidentify = _axesidentify.copy()
    _attribidentify.update({'PX': 3, 'Px': 3, 'px': 3, 3: 3,
                            'PY': 4, 'Py': 4, 'py': 4, 4: 4,
                            'PZ': 5, 'Pz': 5, 'pz': 5, 9: 9,
                            'weight': 9, 'w': 9, 10: 10,
                            'id': 10, 'ID': 10})

    def __init__(self, dumpidentifier):
        '''
        initializes the reader object with a specific dump.
        '''
        self.dumpidentifier = dumpidentifier

    @abc.abstractmethod
    def keys(self):
        '''
        returns a list of keys, that can be used in __getitem__ to read
        any information from this dump.
        '''
        pass

    @abc.abstractmethod
    def __getitem__(self, key):
        pass

    # --- General Information ---
    @abc.abstractmethod
    def timestep(self):
        pass

    @abc.abstractmethod
    def time(self):
        pass

    @abc.abstractmethod
    def simdimensions(self):
        pass

    # --- Data on Grid ---
    @abc.abstractmethod
    def dataE(self, axis):
        pass

    @abc.abstractmethod
    def dataB(self, axis):
        pass

    @abc.abstractmethod
    def grid(self, axis):
        pass

    # --- Particle Data ---
    @abc.abstractmethod
    def listSpecies(self):
        pass

    @abc.abstractmethod
    def getSpecies(self, species, attrib):
        pass

    def __str__(self):
        return '<Dumpreader initialized with "' \
               + str(self.dumpidentifier) + '">'

    # Higher Level Functions for usability


class Simulationreader_ifc(collections.Sequence):
    '''
    Interface for reading the data of a full Simulation.

    Any Simulationreader_ifc implementation will always be initialized using a
    simidentifier. This simidentifier can be anything, that points to
    the data of multiple dumps. In the easiest case this can be the .visit
    file.

    The Simulationreader_ifc is subclass of collections.Sequence and will
    thus behave as a Sequence. The Objects in the Sequence are supposed to be
    subclassed from Dumpreader_ifc.

    It is highly recommended to also override the __str__ function.
    '''

    def __init__(self, simidentifier):
        '''
        This Function should raise a TypeError if the Reader
        dosent support the file format specified.
        '''
        self.simidentifier = simidentifier

    def __getitem__(self, key):
        if isinstance(key, slice):
            return [self[ii] for ii in xrange(*key.indices(len(self)))]
        elif isinstance(key, int):
            if key < 0:  # Handle negative indices
                key += len(self)
            if key >= len(self):
                raise IndexError("The index (%d) is out of range." % key)
            return self.getDumpnumber(key)
        else:
            raise TypeError("Invalid argument type.")

    @abc.abstractmethod
    def getDumpnumber(self, number):
        '''
        Returns the corresponding Dumpreader.
        '''
        pass

    def __str__(self):
        return '<Simulationreader initialized with "' \
               + str(self.simidentifier) + '">'

    # Higher Level Functions for usability

    def times(self):
        return np.array([s.time() for s in self])

# --- datareader package functions ---

__all__ = ['Dumpreader_ifc', 'Simulationreader_ifc', 'chooseCode',
           'readDump', 'readSim']

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


def readDump(dumpidentifier):
    global _dumpreadercls
    if _dumpreadercls is None:
        raise Exception('Specify dumpreaderclass first.')
    return _dumpreadercls(dumpidentifier)


def readSim(simidentifier):
    global _simreadercls
    if _simreadercls is None:
        raise Exception('Specify simreaderclass first.')
    return _simreadercls(simidentifier)


def chooseCode(code):
    '''
    Chooses between codes available.
    Parameters:
    ----------------
    code    The code that has produced the dumps.
            "EPOCH" - .sdf files written by EPOCH1D, EPOCH2D or EPOCH3D.
            "DUMMY" - dummy class creating fake data.
    '''
    if code in ['EPOCH', 'epoch', 'EPOCH1D', 'EPOCH2D', 'EPOCH3D']:
        from epochsdf import Sdfreader, Visitreader
        setdumpreadercls(Sdfreader)
        setsimreadercls(Visitreader)
    elif code in ['DUMMY', 'dummy']:
        from dummy import Dummyreader, Dummysim
        setdumpreadercls(Dummyreader)
        setsimreadercls(Dummysim)
    else:
        raise TypeError('Code "' + str(code) + '" not recognized.')














