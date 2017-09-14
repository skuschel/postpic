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

"""
Some global constants that are used in the code.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import numpy as np
import re
import warnings
import functools
try:
    from . import cythonfunctions as cyf
    particleshapes = cyf.shapes
except(ImportError):
    warnings.warn('cython libs could not be imported. This alters the performance'
                  'of postpic drastically. Falling back to numpy functions.')
    cyf = None
    particleshapes = [0]

warnings.filterwarnings('once', category=DeprecationWarning)
# Default values for histogramdd function
histogramdd_defs = {'shape': 2}

axesidentify = {'X': 0, 'x': 0, 0: 0,
                'Y': 1, 'y': 1, 1: 1,
                'Z': 2, 'z': 2, 2: 2,
                None: slice(None)}
attribidentify = axesidentify.copy()
attribidentify.update({'PX': 3, 'Px': 3, 'px': 3, 3: 3,
                       'PY': 4, 'Py': 4, 'py': 4, 4: 4,
                       'PZ': 5, 'Pz': 5, 'pz': 5, 9: 9,
                       'weight': 9, 'w': 9, 10: 10,
                       'id': 10, 'ID': 10,
                       'mass': 11, 'm': 11, 'Mass': 11,
                       'charge': 12, 'c': 12, 'Charge': 12, 'q': 12})


def deprecated(msg):
    '''
    Mark functions as deprecated by using this decorator.
    msg is an additioanl message that will be displayed.
    '''
    def _deprecated(func):
        d = dict(msg=msg, name=func.__name__)
        if msg is None:
            s = 'The function {name} is deprecated.'.format(**d)
        else:
            # format 2 times, so {name} can be used within msg
            s = "The function {name} is deprecated. {msg}".format(**d).format(**d)

        @functools.wraps(func)
        def ret(*args, **kwargs):
            warnings.warn(s, category=DeprecationWarning)
            return func(*args, **kwargs)
        return ret
    return _deprecated


def append_doc_of(obj):
    '''
    decorator to append the doc of `obj` to decorated object/class.
    '''
    def ret(a):
        doc = '' if a.__doc__ is None else a.__doc__
        a.__doc__ = doc + obj.__doc__
        return a
    return ret


class PhysicalConstants:
    """
    gives you some constants.
    """

    c = 299792458.0
    me = 9.109383e-31
    mass_u = me * 1836.2
    qe = 1.602176565e-19
    mu0 = np.pi * 4e-7  # N/A^2
    epsilon0 = 1 / (mu0 * c ** 2)  # 8.85419e-12 As/Vm

    @staticmethod
    def ncrit_um(lambda_um):
        '''
        Critical plasma density in particles per m^3 for a given
        wavelength lambda_um in microns.
        '''
        return 1.11e27 * 1 / (lambda_um ** 2)  # 1/m^3

    @staticmethod
    def ncrit(laslambda):
        '''
        Critical plasma density in particles per m^3 for a given
        wavelength laslambda in m.
        '''
        return PhysicalConstants.ncrit_um(laslambda * 1e6)  # 1/m^3


class SpeciesIdentifier(PhysicalConstants):
    '''
    This Class provides static methods for deriving particle properties
    from species Names. The only reason for this to be a class is that it
    can be used as a mixin.
    '''

    # unit: electronmass
    _masslist = {'electrongold': 1, 'proton': 1836.2 * 1,
                 'ionp': 1836.2, 'ion': 1836.2 * 12, 'c6': 1836.2 * 12,
                 'ionf': 1836.2 * 19, 'Palladium': 1836.2 * 106,
                 'Palladium1': 1836.2 * 106, 'Palladium2': 1836.2 * 106,
                 'Ion': 1836.2, 'Photon': 0, 'Positron': 1, 'positron': 1,
                 'gold1': 1836.2 * 197, 'gold2': 1836.2 * 197,
                 'gold3': 1836.2 * 197, 'gold4': 1836.2 * 197,
                 'gold7': 1836.2 * 197, 'gold10': 1836.2 * 197,
                 'gold20': 1836.2 * 197}

    # unit: elementary charge
    _chargelist = {'electrongold': -1, 'proton': 1,
                   'ionp': 1, 'ion': 1, 'c6': 6,
                   'ionf': 1, 'Palladium': 0,
                   'Palladium1': 1, 'Palladium2': 2,
                   'Ion': 1, 'Photon': 0, 'Positron': 1, 'positron': 1,
                   'gold1': 1, 'gold2': 2, 'gold3': 3,
                   'gold4': 4, 'gold7': 7, 'gold10': 10,
                   'gold20': 20}

    _isionlist = {'electrongold': False, 'proton': True,
                  'ionp': True, 'ion': True, 'c6': True,
                  'ionf': True, 'f9': True, 'Palladium': True,
                  'Palladium1': True, 'Palladium2': True,
                  'Ion': True, 'Photon': False, 'Positron': False,
                  'positron': False,
                  'gold1': True, 'gold2': True, 'gold3': True,
                  'gold4': True, 'gold7': True, 'gold10': True,
                  'gold20': True}

    #  unit: amu
    _masslistelement = {'H': 1, 'He': 4, 'Li': 6.9, 'C': 12, 'N': 14, 'O': 16, 'F': 19,
                        'Ne': 20.2, 'Al': 27, 'Si': 28, 'Ar': 40, 'Rb': 85.5, 'Au': 197}

    @staticmethod
    def isejected(species):
        s = species.replace('/', '')
        r = re.match(r'(ejected_)(.*)', s)
        return r is not None

    @classmethod
    def ision(cls, species):
        return cls.identifyspecies(species)['ision']

    @classmethod
    def identifyspecies(cls, species):
        """
        Returns a dictionary containing particle informations deduced from
        the species name.
        The following keys in the dictionary will always be present:
        name   species name string
        mass    kg (SI)
        charge  C (SI)
        tracer  boolean
        ejected boolean

        Valid Examples:
        Periodic Table symbol + charge state: c6, F2, H1, C6b
        ionm#c# defining mass and charge:  ionm12c2, ionc20m110
        advanced examples:
        ejected_tracer_ionc5m20b, ejected_tracer_electronx,
        ejected_c6b, tracer_proton, protonb
        """
        ret = {'tracer': False, 'ejected': False, 'name': species}
        s = species.replace('/', '_')

        # Regex for parsing ion species name.
        # See docsting for valid examples
        regex = '(?P<prae>(.*_)*)' \
                '((?P<elem>((?!El)[A-Z][a-z]?))(?P<elem_c>\d*)|' \
                '(?P<ionmc>(ionc(?P<c1>\d+)m(?P<m2>\d+)|ionm(?P<m1>\d+)c(?P<c2>\d+)))' \
                ')?' \
                '(?P<plus>(Plus)*)' \
                '(?P<electron>[Ee]le[ck])?' \
                '(?P<suffix>\w*?)$'
        r = re.match(regex, s)
        if r is None:
            raise Exception('Species ' + str(s) +
                            ' does not match regex name pattern: ' +
                            str(regex))
        regexdict = r.groupdict()
        # print(regexdict)

        # recognize anz prae and add dictionary key
        if regexdict['prae']:
            for i in regexdict['prae'].split('_'):
                key = i.replace('_', '')
                if not key == '':
                    ret[key] = True

        # Excluding patterns start here
        # 1) Name Element + charge state: C1, C6, F2, F9, Au20, Pb34a
        if regexdict['elem']:
            ret['mass'] = float(cls._masslistelement[regexdict['elem']]) * \
                1836.2 * cls.me
            if regexdict['elem_c'] == '':
                ret['charge'] = 0
            else:
                ret['charge'] = float(regexdict['elem_c']) * cls.qe
            ret['ision'] = True
        # 2) ionmc like
        elif regexdict['ionmc']:
            if regexdict['c1']:
                ret['mass'] = float(regexdict['m2']) * 1836.2 * cls.me
                ret['charge'] = float(regexdict['c1']) * cls.qe
                ret['ision'] = True

            if regexdict['c2']:
                ret['mass'] = float(regexdict['m1']) * 1836.2 * cls.me
                ret['charge'] = float(regexdict['c2']) * cls.qe
                ret['ision'] = True

        # charge may be given via plus
        if regexdict['plus']:
            charge = len(regexdict['plus'])/4 * cls.qe
            if ret['charge'] == 0:
                ret['charge'] = charge

        # Elektron can be appended to any ion name, so overwrite.
        if regexdict['electron']:
            ret['mass'] = cls.me
            ret['charge'] = -1 * cls.qe
            ret['ision'] = False

        # simply added to _masslist and _chargelist
        # this should not be used anymore
        # set only if property is not already set
        if 'mass' not in ret and regexdict['suffix'] in cls._masslist:
            ret['mass'] = float(cls._masslist[regexdict['suffix']]) * cls.me
        if 'charge' not in ret and regexdict['suffix'] in cls._chargelist:
            ret['charge'] = float(cls._chargelist[regexdict['suffix']] * cls.qe)
        if 'ision' not in ret and regexdict['suffix'] in cls._isionlist:
            ret['ision'] = cls._isionlist[regexdict['suffix']]

        if not (('mass' in ret) and ('charge' in ret) and ('ision' in ret)):
            raise Exception('species ' + species + ' not recognized.')

        return ret


# Some static functions


def cutout(m, oldextent, newextent):
    """
    cuts out a part of the matrix m that belongs to newextent if the full
    matrix corresponds to oldextent. If m has dims dimensions, then oldextent
    and newextent have to have a length of 2*dims each.
    nexextent has to be inside of oldextent!
    (this should be fixed in the future...)
    """
    import numpy as np
    dims = len(m.shape)
    assert oldextent is not newextent, 'oldextent and newextent point to the' \
                                       'same objekt(!). Get a coffe and' \
                                       'check your code again. :)'
    assert len(oldextent) / 2 == dims, \
        'dimensions of oldextent and m are wrong!'
    assert len(newextent) / 2 == dims, \
        'dimensions of newextent and m are wrong!'
    s = ()
    for dim in range(dims):
        i = 2 * dim
        thisdimmin = round((newextent[i] - oldextent[i]) /
                           (oldextent[i + 1] - oldextent[i]) * m.shape[dim])
        thisdimmax = round((newextent[i + 1] - oldextent[i]) /
                           (oldextent[i + 1] - oldextent[i]) * m.shape[dim])
        s = np.append(s, slice(thisdimmin, thisdimmax))
    if len(s) == 1:
        s = s[0]
    else:
        s = tuple(s)
    return m[s]


def transfromxy2polar(matrixxy, extentxy,
                      extentpolar, shapepolar, ashistogram=True):
    '''
    remaps a matrix matrixxy in kartesian coordinates x,y to a polar
    representation with axes r, phi.
    '''
    from scipy.ndimage.interpolation import geometric_transform
    import numpy as np

    def polar2xy(rphi):
        (r, phi) = rphi
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        return (x, y)

    def koord2index(q1q2, extent, shape):
        (q1, q2) = q1q2
        return ((q1 - extent[0]) / (extent[1] - extent[0]) * shape[0],
                (q2 - extent[2]) / (extent[3] - extent[2]) * shape[1])

    def index2koord(ij, extent, shape):
        (i, j) = ij
        return (extent[0] + i / shape[0] * (extent[1] - extent[0]),
                extent[2] + j / shape[1] * (extent[3] - extent[2]))

    def mappingxy2polar(ij, extentxy, shapexy, extentpolar, shapepolar):
        '''
        actually maps indizes of polar matrix to indices of kartesian matrix
        '''
        (i, j) = ij
        ret = polar2xy(index2koord((float(i), float(j)),
                                   extentpolar, shapepolar))
        ret = koord2index(ret, extentxy, shapexy)
        return ret

    ret = geometric_transform(matrixxy, mappingxy2polar,
                              output_shape=shapepolar,
                              extra_arguments=(extentxy, matrixxy.shape,
                                               extentpolar, shapepolar),
                              order=1)
    if ashistogram:  # volumeelement is just r
        r = np.abs(np.linspace(extentpolar[0], extentpolar[1], ret.shape[0]))
        ret = (ret.T * r).T
    return ret


def histogramdd(data, **kwargs):
    '''
    automatically chooses the histogram function to be used.
    `data` must be a tuple. Its length determines the
    dimensions of the histogram returned.
    '''
    [kwargs.setdefault(k, i) for (k, i) in list(histogramdd_defs.items())]
    if len(data) > 3:
        raise ValueError('{} is larger than the max number of dimensions '
                         'allowed (3)'.format(len(data)))
    if cyf is None:
        shape = kwargs.pop('shape', 0)
        if shape != 0:
            warnings.warn('shape {} not available without cython.'.format(shape))
        if len(data) == 1:
            h, xedges = np.histogram(data[0], **kwargs)
            return h, xedges
        if len(data) == 2:
            h, xedges, yedges = np.histogram2d(data[0], data[1], **kwargs)
            return h, xedges, yedges
        if len(data) == 3:
            h, (xe, ye, ze) = np.histogramdd(data, **kwargs)
            return h, xe, ye, ze
    else:
        if len(data) == 1:
            h, xedges = cyf.histogram(np.float64(data[0]), **kwargs)
            return h, xedges
        if len(data) == 2:
            h, xedges, yedges = cyf.histogram2d(np.float64(data[0]),
                                                np.float64(data[1]), **kwargs)
            return h, xedges, yedges
        if len(data) == 3:
            h, xe, ye, ze = cyf.histogram3d(np.float64(data[0]),
                                            np.float64(data[1]),
                                            np.float64(data[2]), **kwargs)
            return h, xe, ye, ze


def kspace(component, fields, interpolation=None, omega_func=None):
    '''
    Reconstruct the physical kspace of one polarization component
    This function basically computes one component of
        E = 0.5*(E - omega/k^2 * Cross[k, E])
    or
        B = 0.5*(B + 1/omega * Cross[k, B]).

    component must be one of ["Ex", "Ey", "Ez", "Bx", "By", "Bz"].

    The necessary fields must be given in the dict fields with keys
    chosen from ["Ex", "Ey", "Ez", "Bx", "By", "Bz"].
    Which are needed depends on the chosen component and
    the dimensionality of the fields. In 3D the following fields are necessary:

    Ex, By, Bz -> Ex
    Ey, Bx, Bz -> Ey
    Ez, Bx, By -> Ez

    Bx, Ey, Ez -> Bx
    By, Ex, Ez -> By
    Bz, Ex, Ey -> Bz

    In 2D, components which have "k_z" in front of them (see cross-product in
    equations above) are not needed.
    In 1D, components which have "k_y" or "k_z" in front of them (see
    cross-product in equations above) are not needed.

    The keyword-argument interpolation indicates whether interpolation should be
    used to remove the grid stagger. If interpolation is None, this function
    works only for non-staggered grids. Other choices for interpolation are
    "linear" and "fourier".

    The keyword-argument omega_func may be used to pass a function that will
    calculate the dispersion relation of the simulation may be given. The
    function will receive one argument that contains the k mesh.
    '''
    # target field is polfield and the other field is otherfield
    polfield = component[0]
    otherfield = 'B' if polfield == 'E' else 'E'

    # polarization axis
    polaxis = axesidentify[component[1]]

    # build a dict of the keys of the fields-dict
    field_keys = {'E': {0: 'Ex', 1: 'Ey', 2: 'Ez'},
                  'B': {0: 'Bx', 1: 'By', 2: 'Bz'}}

    # copy the polfield as a starting point for the result
    try:
        result = copy.deepcopy(fields[field_keys[polfield][polaxis]])
    except KeyError:
        raise ValueError("Required field {} not present in fields".format(component))

    # remember the origins of result's axes to compare with other fields
    result_origin = [a.grid_node[0] for a in result.axes]

    # store box size of input field
    Dx = np.array([a.grid_node[-1] - a.grid_node[0] for a in result.axes])

    # store grid spacing of input field
    dx = np.array([a.grid_node[1] - a.grid_node[0] for a in result.axes])

    # Change to frequency domain
    result.fft()

    # calculate the k mesh and k^2
    mesh = np.meshgrid(*[ax.grid for ax in result.axes], indexing='ij', sparse=True)
    k2 = sum(ki**2 for ki in mesh)

    # calculate omega, either using the vacuum expression or omega_func()
    if omega_func is None:
        omega = PhysicalConstants.c * np.sqrt(k2)
    else:
        omega = omega_func(mesh)

    # calculate the prefactor in front of the cross product
    # this will produce nan/inf in specific places, which are replaced by 0
    old_settings = np.seterr(all='ignore')
    if polfield == "E":
        prefactor = omega/k2
        prefactor[np.argwhere(np.isnan(prefactor))] = 0.0
    else:
        prefactor = -1.0/omega
        prefactor[np.argwhere(np.isinf(prefactor))] = 0.0
    np.seterr(**old_settings)

    # add/subtract the two terms of the cross-product
    # i chooses the otherfield component  (polaxis+i) % 3
    # mesh_i chooses the k-axis component (polaxis-i) % 3
    # which recreates the crossproduct
    for i in (1, 2):
        mesh_i = (polaxis-i) % 3
        if mesh_i < len(mesh):
            # copy the otherfield component, transform and reverse the grid stagger
            field_key = field_keys[otherfield][(polaxis+i) % 3]
            try:
                field = copy.deepcopy(fields[field_key])
            except KeyError as e:
                raise ValueError("Required field {} not present in fields".format(e.message))

            # remember the origin and box size of the field
            field_origin = [a.grid_node[0] for a in field.axes]
            oDx = np.array([a.grid_node[-1] - a.grid_node[0] for a in field.axes])

            # Test if all fields have the same number of grid points
            if not field.shape == result.shape:
                raise ValueError("All given Fields must have the same number of grid points. "
                                 "Field {} has a different shape than {}.".format(field_key,
                                                                                  component))

            # Test if the axes of all fields have the same lengths
            if not np.all(np.isclose(Dx, oDx)):
                raise ValueError("The axes of all given Fields must have the same length. "
                                 "Field {} has a different extent than {}.".format(field_key,
                                                                                   component))

            # Test if all fields have same grid origin...
            if interpolation is None:
                if not np.all(np.isclose(result_origin, field_origin)):
                    raise ValueError("The grids of all given Fields should have the same origin."
                                     "The origin of {} ({}) differs from the origin of {} ({})."
                                     "".format(field_key, field_origin, component, result_origin))

            # ...or at least approximately the same origin, when interpolation is activated
            else:
                grid_shift = [
                    so-to for so, to in
                    zip(result_origin, field_origin)
                ]

                if not np.all(abs(np.array(grid_shift)) < 2.*dx):
                    raise ValueError("The grids of all given Fields should have approximately the "
                                     "same origin. The origin of {} ({}) differs from the origin "
                                     "of {} ({}) by more than 2 dx."
                                     "".format(field_key, field_origin, component, result_origin))

            # linear interpolation is applied before the fft
            if interpolation == 'linear':
                field.shift_grid_by(grid_shift, interpolation='linear')

            field.fft()

            # fourier interpolation is done after the fft by applying a linear phase
            if interpolation == 'fourier':
                field._apply_linear_phase(dict(enumerate(grid_shift)))

            # add the field to the result with the appropriate prefactor
            result.matrix += (-1)**(i-1) * prefactor * mesh[mesh_i] * field.matrix

    return result

