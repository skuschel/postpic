from __future__ import absolute_import, division, print_function, unicode_literals

from . import Dumpreader_ifc
from . import Simulationreader_ifc
import re
import os.path
import h5py
import glob
import numpy as np
import helper 
from helper_fft import fft

__all__ = ['AMSmileiReader', 'SmileiSeries']


def _generateh5indexfile(indexfile, fnames):
    '''
    Creates a h5 index file called indexfile containing external links to all h5
    datasets which are in in the h5 filelist fnames. This is fast as datasets
    will only be externally linked within the new indexfile.
    Therefore the indexfile will also be small in size.

    Will throw an error, if any of the h5 files contain a dataset under the same name.
    '''
    if os.path.isfile(indexfile):
        # indexfile already exists. do not recreate
        return

    dirname = os.path.dirname(fnames[0])
    indexfile = os.path.join(dirname, indexfile)

    def visitf(key):
        # key is a string
        # only link if key points to a dataset. Do not link groups
        if isinstance(hf[key], h5py._hl.dataset.Dataset):
            ih[key] = h5py.ExternalLink(fname, key)

    with h5py.File(indexfile, 'w') as ih:
        for fname in fnames:
            with h5py.File(fname, 'r') as hf:
                hf.visit(visitf)


def _getindexfile(path):
    '''
    Returns the name of the index file after the file has been
    generated (File generation only if it doesnt exist)
    '''
    indexfile = os.path.join(path, '.postpic-smilei-index.h5')
    if not os.path.isfile(indexfile):
        # find all h5 files
        h5files = glob.glob(os.path.join(path, '*.h5'))
        print('generating index file "{}" from the following'
              'h5 files: {}'.format(indexfile, h5files))
        _generateh5indexfile(indexfile, h5files)
    return indexfile

class AMSmileiReader:

    def __init__(self, h5file, iteration=None):
        #super(OpenPMDreader, self).__init__(h5file)
        
        if os.path.isfile(h5file):
            self._h5 = h5py.File(h5file, 'r')
        elif os.path.isdir(h5file):
            indexfile = _getindexfile(h5file)
            self._h5 = h5py.File(indexfile, 'r')
        else:
            raise IOError('"{}" is neither a h5 file nor a directory'
                          'containing h5 files'.format(h5file))

        #<<<<<Evaluating time in both SI units and smilei units>>>>>
        self.timestep_arr = np.array(self._h5['/data/'])  #returns available iterations

        if '{:010d}'.format(iteration) not in self.timestep_arr:
            #Finding the closest integer iteration available from the user iteration
            if iteration is not None:
                int_iter = min(self.timestep_arr, key=lambda x: (abs(x - int(iteration)), x))
            else:
                int_iter = self.timestep_arr[-1]
        
            self.sim_iter = '{:010d}'.format(int_iter)   #timestep format like in the simulation dump     
        else:
            self.sim_iter = '{:010d}'.format(iteration)
        
        
        #-------------------------------------------------------------------------------------------------------------------------
        #Directly importing data files, because the field data and particle data were dumped in two different h5 files by smilei
        #And I am having confusion with the index file :(
        try:
            #field data file
            self._dataF = h5py.File('/path/Fields0.h5', 'r')['/data/{}'.format(self.sim_iter)]
        except:
            pass

        try:
            #particle data file
            self._dataP = h5py.File('/path/TrackParticlesDisordered_electron.h5', 'r')['/data/{}/particles'.format(self.sim_iter)]
        except:
            pass
        #-------------------------------------------------------------------------------------------------------------------------
        #time in SI units
        self.real_time=(self._dataF.attrs['time']) * (self._dataF.attrs['timeUnitSI'])

    def attrs(self, data):
        return data.attrs

    def __del__(self):
        del self._dataF

    # --- Level 0 methods ---

    def keys(self):
        return list(self._dataF.keys())

    def __getitem__(self, key):
        return self._dataF[key]
    
    # --- Level 1 methods ---

    #To return the number of AM modes and Field names in the dump.
    def getAMmodes(self):    
        strings=np.array(self._dataF.keys())
        max_suffix = float('-inf')
        max_suffix_string = None
        prefix_list=[]

        for s in strings:
            prefix, suffix = s.split('_mode_')
            suffix_int = int(suffix)
            if suffix_int > max_suffix:
                max_suffix = suffix_int
                max_suffix_string = s
            prefix_list.append(prefix)

        return [prefix_list, int(max_suffix_string[-1])+1]    #[field_names_preffix, number of modes]
    
    def Data(self, key, modes):
        '''
        should work with any key, that contains data, thus on every hdf5.Dataset,
        but not on hdf5.Group. Will extract the data, convert it to SI and return it
        as a numpy array of complex field data.
        '''
        if modes is None:
        
            field_array = np.array(self._dataF[key])
            field_array_shape = field_array.shape
            reshaped_array = field_array.reshape(field_array_shape[0], field_array_shape[1]//2, 2)
            complex_array = reshaped_array[:, :, 0] + 1j * reshaped_array[:, :, 1]

            return complex_array   #complex_array consist of field data in complex numbers
        
        elif modes =="all":
            array_list=[]
            for mode in range(self.getAMmodes[-1]):

                field_name = key+"_mode_"+str(mode)
                field_array = np.array(self._dataF[field_name])
                field_array_shape = field_array.shape
                reshaped_array = field_array.reshape(field_array_shape[0], field_array_shape[1]//2, 2)
                complex_array = reshaped_array[:, :, 0] + 1j * reshaped_array[:, :, 1]
                array_list.append(complex_array)
        
            mod_complex_data= np.stack(array_list,axis=0)     #Modified array of shape (Nmodes, Nx, Nr)
            return mod_complex_data
    
    #-----------MODE EXPANSION METHODS-----------

    @staticmethod
    def _modeexpansion_naiv_single(complex_data, theta=0):
    
        F = np.zeros_like(np.real(complex_data[0]))                       
        
        '''
        for m in self.modes:
            F += self.mod_data[m]*np.exp(-1j*m*self.theta)
        '''
        for m in range(getAMmodes[-1]):
            F += np.real(complex_data[m])*np.cos(m*theta)+np.imag(complex_data[m])*np.sin(m*theta)

        return F 
    
    @staticmethod
    def _modeexpansion_naiv(complex_data, theta=0):
        '''
        converts to radial data using `modeexpansion`, possibly for multiple
        theta at once.
        '''
        if np.asarray(theta).shape is ():
            # single theta
            theta = [theta]
        # multiple theta
        data = np.asarray([_modeexpansion_naiv_single(complex_data, theta=t)
                           for t in theta])
        # switch from (theta, r, z) to (r, theta, z)
        data = data.swapaxes(0, 1)
        return data
    
    @staticmethod
    def _modeexpansion_fft(complex_data, Ntheta=None):
        '''
        calculate the radialdata using an fft. This is by far the fastest
        way to do the modeexpansion.
        '''
        Nm, Nx, Nr = complex_data.shape
        Nth = (Nm+1)//2
        if Ntheta is None or Ntheta < Nth:
            Ntheta = Nth
        fd = np.empty((Nr, Ntheta, Nx), dtype=np.complex128)

        fd[:, 0, :].real = complex_data[0, :, :]
        rawdatasw = np.swapaxes(complex_data, 0, 1)
        fd[:, 1:Nth, :].real = rawdatasw[:, 1::2, :]
        fd[:, 1:Nth, :].imag = rawdatasw[:, 2::2, :]

        fd = fft.fft(fd, axis=1).real
        return fd
    
    @staticmethod
    def mode_expansion(complex_data, theta=None, Ntheta=None):
        
        Nm, Nr, Nz = complex_data.shape
        if Ntheta is not None or theta is None:
            return _modeexpansion_fft(complex_data, Ntheta=Ntheta)
        else:
            return _modeexpansion_naiv(complex_data, theta=theta)

    #----------------------------------------------------------------------------------    

    def gridoffset(self, key, axis):
        axid = helper.axesidentify[axis]
        if axid ==90 or axis=='r':
            axid=axid%90

        if "gridUnitSI" in self._dataF[key].attrs:
            attrs = self._dataF[key].attrs
        else:
            attrs = self._dataF[key].parent.attrs
        return attrs['gridGlobalOffset'][axid] * attrs['gridUnitSI']

    def gridspacing(self, key, axis):
        axid = helper.axesidentify[axis]
        if axid ==90 or axis=='r':
            axid=axid%90

        if "gridUnitSI" in self._dataF[key].attrs:
            attrs = self._dataF[key].attrs
        else:
            attrs = self._dataF[key].parent.attrs
        return attrs['gridSpacing'][axid] * attrs['gridUnitSI']
    
    #----------------Here def gridpoints(self) is missing temprorarily-----------------

    # --- Level 2 methods ---

    def timestep(self):
        return self.sim_iter

    def time(self):
        return self.real_time

    def simdimensions(self):
        '''
        the number of spatial dimensions the simulation was using.
        '''
        for k in self._simgridkeys():
            try:
                gs = self.gridspacing(k, None)
                return len(gs)
            except KeyError:
                pass
        raise KeyError('number of simdimensions could not be retrieved for {}'.format(self))

    def _keyE(self, component, **kwargs):
        # Smilei deviates from openPMD standard: Ex instead of E/x
        axsuffix = {0: 'x', 1: 'y', 2: 'z', 90: 'r', 91: 't'}[helper.axesidentify[component]]
        return 'E{}'.format(axsuffix)

    def _keyB(self, component, **kwargs):
        # Smilei deviates from openPMD standard: Bx instead of B/x
        axsuffix = {0: 'x', 1: 'y', 2: 'z', 90: 'r', 91: 't'}[helper.axesidentify[component]]
        return 'B{}'.format(axsuffix)

    def _simgridkeys(self):
        return ['El', 'Er', 'Et',
                'Bl', 'Br', 'Bt']

    def listSpecies(self):
        ret = list(self._dataP.keys())
        return ret

    def getSpecies(self, species, attrib):
        """
        Returns one of the attributes out of (x,y,z,px,py,pz,weight,ID,mass,charge) of
        this particle species.
        """
        attribid = helper.attribidentify[attrib]
        options = {9: 'particles/{}/weighting',
                   0: 'particles/{}/position/x',
                   1: 'particles/{}/position/y',
                   2: 'particles/{}/position/z',
                   3: 'particles/{}/momentum/x',
                   4: 'particles/{}/momentum/y',
                   5: 'particles/{}/momentum/z',
                   10: 'particles/{}/id',
                   11: 'particles/{}/mass',
                   12: 'particles/{}/charge'}
        optionsoffset = {0: 'particles/{}/positionOffset/x',
                         1: 'particles/{}/positionOffset/y',
                         2: 'particles/{}/positionOffset/z'}
        key = options[attribid]
        offsetkey = optionsoffset.get(attribid)
        try:
            data = self._dataP(key.format(species))
            if offsetkey is not None:
                data += self._dataP(offsetkey.format(species))
            ret = np.asarray(data, dtype=np.float64)
        except IndexError:
            raise KeyError
        return ret

    #To get the axis array.
    def getAxis(self, axis):
    
        namelist = list(self._dataF.keys())
        Nx,Nr = self._dataF[namelist[0]].shape
        Nr=Nr/2

        if axis is None:
            raise IOError("Invalid axis")
        elif len(axis)>1:
            raise IOError("Only one axis at a time")
        
        if axis == "x":      #If moving = True, the x axis data modifies according to the moving window.
            x_min = self._dataF.attrs['x_moved'] * self._dataF[namelist[0]].attrs['gridUnitSI'] 
            x_max = x_min + self.gridspacing(key=namelist[0],axis=0)*(Nx-1)
            x_axis = np.linspace(x_min, x_max, Nx-1)
            return x_axis   
        elif axis == "r":
            r_max = self.gridspacing(key=namelist[0],axis='r')*(Nr-1)
            r_axis = np.linspace(0, r_max, Nr-1)
            return r_axis
        else:
            raise IOError("Invalid axis")
    
    '''
    def getderived(self):
        
        #return all other fields dumped, except E and B.
        
        ret = []
        self['fields'].visit(ret.append)
        ret = ['fields/{}'.format(r) for r in ret if not (r.startswith('E') or r.startswith('B'))]
        ret = [r for r in ret if hasattr(self[r], 'value')]
        ret.sort()
        return ret
    '''

    def __str__(self):
        return '<OpenPMDh5reader at "' + str(self.dumpidentifier) + '">'
    

class SmileiSeries:
    '''
    Reads a time series of dumps from a given directory.

    Point this to the directory and you will get all dumps,
    but possibly containing different data.
    `simreader = SmileiSeries('path/to/simulation/')`

    Alternatively point this to a single file and you will only get
    the iterations which are present in that file:
    `simreader = SmileiSeries('path/to/simulation/Fields0.h5')`
    '''

    def __init__(self, h5file, dumpreadercls=AMSmileiReader, **kwargs):
        super(SmileiSeries, self).__init__(h5file, **kwargs)
        self.dumpreadercls = dumpreadercls
        self.h5file = h5file
        self.path = os.path.basename(h5file)
        if os.path.isfile(h5file):
            with h5py.File(h5file, 'r') as h5:
                self._dumpkeys = list(h5['data'].keys())
        elif os.path.isdir(h5file):
            indexfile = _getindexfile(h5file)
            self._h5 = h5py.File(indexfile, 'r')
            with h5py.File(indexfile, 'r') as h5:
                self._dumpkeys = list(h5['data'].keys())
        else:
            raise IOError('{} does not exist.'.format(h5file))

    def _getDumpreader(self, n):
        '''
        Do not use this method. It will be called by __getitem__.
        Use __getitem__ instead.
        '''
        return self.dumpreadercls(self.h5file, self._dumpkeys[n])

    def __len__(self):
        return len(self._dumpkeys)

    def __str__(self):
        return '<SmileiSeries based on "{}">'.format(self.simidentifier)
