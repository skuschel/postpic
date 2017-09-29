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
# Stephan Kuschel 2014-2017
# Alexander Blinne, 2017
"""
Some global constants that are used in the code.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import copy
import itertools
import numbers
import numpy as np
import numpy.linalg as npl
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


__all__ = ['PhysicalConstants', 'kspace_epoch_like', 'kspace',
           'kspace_propagate']


def isnotebook():
    return 'ipykernel' in sys.modules


def jupyter_client_version():
    try:
        import jupyter_client
        return jupyter_client.__version__
    except ImportError:
        return None


def _filterwarnings():
    if isnotebook():
        jver = jupyter_client_version()
        if jver:
            jmajor = int(jver.split('.')[0])
            if jmajor == 5:
                return

    warnings.filterwarnings('once', category=DeprecationWarning)


_filterwarnings()

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


def prepend_doc_of(obj):
    '''
    decorator to append the doc of `obj` to decorated object/class.
    '''
    def ret(a):
        doc = '' if a.__doc__ is None else a.__doc__
        a.__doc__ = obj.__doc__ + doc
        return a
    return ret


class float_with_name(float):
    def __new__(self, value, name):
        return float.__new__(self, value)

    def __init__(self, value, name):
        float.__init__(value)
        self.name = name


class PhysicalConstants:
    """
    gives you some constants.
    """

    c = float_with_name(299792458.0, 'c')
    me = float_with_name(9.109383e-31, 'me')
    mass_u = float_with_name(me * 1836.2, 'mu')
    qe = float_with_name(1.602176565e-19, 'qe')
    mu0 = float_with_name(np.pi * 4e-7, 'mu0')  # N/A^2
    epsilon0 = float_with_name(1 / (mu0 * c ** 2), 'eps0')  # 8.85419e-12 As/Vm

    @staticmethod
    def ncrit_um(lambda_um):
        '''
        Critical plasma density in particles per m^3 for a given
        wavelength lambda_um in microns.
        '''
        return float_with_name(1.11e27 * 1 / (lambda_um ** 2), 'ncrit_um')  # 1/m^3

    @staticmethod
    def ncrit(laslambda):
        '''
        Critical plasma density in particles per m^3 for a given
        wavelength laslambda in m.
        '''
        return float_with_name(PhysicalConstants.ncrit_um(laslambda * 1e6), 'ncrit')  # 1/m^3


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
def polar2linear(theta, r):
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    return x, y


def linear2polar(x, y):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return theta, r


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


def is_non_integer_real_number(x):
    """
    Tests if an object ix is a real number and not an integer.
    """
    return isinstance(x, numbers.Real) and not isinstance(x, numbers.Integral)


def find_nearest_index(array, value):
    """
    Gives the index i of the value array[i] which is closest to value.
    Assumes that the array is sorted.
    """
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or
                    np.fabs(value - array[idx-1]) < np.fabs(value - array[idx])):
                        return idx-1
    else:
        return idx


def omega_yee_factory(dx, dt):
    """
    Return a function omega_yee that is suitable as input for kspace.
    Pass the returned function as omega_func to kspace

    dx: a list of the grid spacings, e. g.
    dx = [ax.grid[1] - ax.grid[0] for ax in dumpreader.Ey().axes]

    dt: time step, e. g.
    dt = dumpreader.time()/dumpreader.timestep()
    """
    def omega_yee(kmesh):
        tmp = sum((np.sin(0.5 * kxi * dxi) / dxi)**2 for kxi, dxi in zip(kmesh, dx))
        omega = 2.0*np.arcsin(PhysicalConstants.c * dt * np.sqrt(tmp))/dt
        return omega
    return omega_yee


def _kspace_helper_cutfields(component, fields, extent):
    slices = fields[component]._extent_to_slices(extent)
    return {k: f[slices] for k, f in fields.items()}


def kspace_epoch_like(component, fields, extent=None, omega_func=None, align_to='B'):
    '''
    Reconstruct the physical kspace of one polarization component
    See documentation of kspace

    This will choose the alignment of the fields in a way to improve
    accuracy on EPOCH-like staggered dumps

    For the current version of EPOCH, v4.9, use the following:
    align_to == 'B' for intermediate dumps, align_to == "E" for final dumps
    '''
    polfield = component[0]
    polaxis = axesidentify[component[1]]

    # apply extent to all fields
    if extent is not None:
        fields = _kspace_helper_cutfields(component, fields, extent)

    if polfield == align_to:
        return kspace(component, fields, interpolation='linear', omega_func=omega_func)

    dx = np.array([ax.grid[1]-ax.grid[0] for ax in fields[component].axes])/2.0
    try:
        dx[polaxis] = 0
    except IndexError:
        pass

    if polfield == 'B':
        dx *= -1

    fields[component] = copy.deepcopy(fields[component])
    fields[component] = fields[component].shift_grid_by(dx, interpolation='linear')
    return kspace(component, fields, interpolation='fourier', omega_func=omega_func)


def kspace(component, fields, extent=None, interpolation=None, omega_func=None):
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

    The keyword-argument extent may be a list of values [xmin, xmax, ymin, ymax, ...]
    which denote a region of the Fields on which to execute the kspace
    reconstruction.

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

    # apply extent to all fields
    if extent is not None:
        fields = _kspace_helper_cutfields(component, fields, extent)

    # polarization axis
    polaxis = axesidentify[component[1]]

    # build a dict of the keys of the fields-dict
    field_keys = {'E': {0: 'Ex', 1: 'Ey', 2: 'Ez'},
                  'B': {0: 'Bx', 1: 'By', 2: 'Bz'}}

    # copy the polfield as a starting point for the result
    try:
        result = fields[field_keys[polfield][polaxis]]
    except KeyError:
        raise ValueError("Required field {} not present in fields".format(component))

    # remember the origins of result's axes to compare with other fields
    result_origin = [a.grid_node[0] for a in result.axes]

    # store box size of input field
    Dx = np.array([a.grid_node[-1] - a.grid_node[0] for a in result.axes])

    # store grid spacing of input field
    dx = np.array([a.grid_node[1] - a.grid_node[0] for a in result.axes])

    # Change to frequency domain
    result = result.fft()

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
    with np.errstate(invalid='ignore', divide='ignore'):
        if polfield == "E":
            prefactor = omega/k2
        else:
            prefactor = -1.0/omega

    prefactor[np.isnan(prefactor)] = 0.0
    prefactor[np.isinf(prefactor)] = 0.0

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
                field = field.shift_grid_by(grid_shift, interpolation='linear')

            field = field.fft()

            # fourier interpolation is done after the fft by applying a linear phase
            if interpolation == 'fourier':
                field = field._apply_linear_phase(dict(enumerate(grid_shift)))

            # add the field to the result with the appropriate prefactor
            result.matrix += (-1)**(i-1) * prefactor * mesh[mesh_i] * field.matrix

    return result


def linear_phase(field, dx):
    '''
    Calculates the linear phase as used in Field._apply_linear_phase and
    _kspace_propagate_generator.
    '''
    import numexpr as ne
    transform_state = field._transform_state(dx.keys())
    axes = [ax.grid for ax in field.axes]  # each axis object returns new numpy array
    for i in range(len(axes)):
        gridlen = len(axes[i])
        if transform_state is True:
            # center the axes around 0 to eliminate global phase
            axes[i] -= axes[i][gridlen//2]
        else:
            # start the axes at 0 to eliminate global phase
            axes[i] -= axes[i][0]

    # build mesh
    mesh = np.meshgrid(*axes, indexing='ij', sparse=True)

    # prepare mesh for numexpr-dict
    kdict = {'k{}'.format(i): k for i, k in enumerate(mesh)}

    # calculate linear phase
    # arg = sum([dx[i]*mesh[i] for i in dx.keys()])
    arg_expr = '+'.join('({}*k{})'.format(repr(v), i) for i, v in dx.items())

    if transform_state is True:
        exp_ikdx_expr = 'exp(1j * ({arg}))'.format(arg=arg_expr)
    else:
        exp_ikdx_expr = 'exp(-1j * ({arg}))'.format(arg=arg_expr)

    exp_ikdx = ne.evaluate(exp_ikdx_expr,
                           local_dict=kdict,
                           global_dict=None)

    return exp_ikdx


def _kspace_propagate_generator(kspace, dt, moving_window_vect=None,
                                move_window=None,
                                remove_antipropagating_waves=None,
                                yield_zeroth_step=False):
    '''
    Evolve time on a field.
    This function checks the transform_state of the field and transforms first from spatial
    domain to frequency domain if necessary. In this case the inverse transform will also
    be applied to the result before returning it. This works, however, only correctly with
    fields that are the inverse transforms of a k-space reconstruction, i.e. with complex
    fields.

    dt: time in seconds

    This function will return an infinite generator that will do arbitrary many time steps.

    If yield_zeroth_step is True, then the kspace will also be yielded after removing the
    antipropagating waves, but before the first actual step is done.

    If a vector moving_window_vect is passed to this function, which is ideally identical
    to the mean propagation direction of the field in forward time direction,
    an additional linear phase is applied in order to keep the pulse inside of the box.
    This effectively enables propagation in a moving window.
    If dt is negative, the window will actually move the opposite direction of
    moving_window_vect.
    Additionally, all modes which propagate in the opposite direction of the moving window,
    i.e. all modes for which dot(moving_window_vect, k)<0, will be deleted.

    The motion of the window can be inhibited by specifying move_window=False.
    If move_window is None, the moving window is automatically enabled if moving_window_vect
    is given.

    The deletion of the antipropagating modes can be inhibited by specifying
    remove_antipropagating_waves=False.
    If remove_antipropagating_waves is None, the deletion of the antipropagating modes
    is automatically enabled if moving_window_vect is given.
    '''
    import numexpr as ne

    transform_state = kspace._transform_state()
    if transform_state is None:
        raise ValueError("kspace must have the same transform_state on all axes. "
                         "Please make sure that either all axes 'live' in spatial domain or all "
                         "axes 'live' in frequency domain.")

    do_fft = not transform_state

    if do_fft:
        kspace = kspace.fft()

    # calculate free space dispersion relation

    # optimized version of
    # omega = PhysicalConstants.c * np.sqrt(sum(k**2 for k in kspace.meshgrid()))
    # using numexpr:
    kdict = {'k{}'.format(i): k for i, k in enumerate(kspace.meshgrid())}
    k2_expr = '+'.join('{}**2'.format(i) for i in kdict.keys())

    numexpr_vars = dict(c=PhysicalConstants.c, dt=dt)
    numexpr_vars.update(kdict)
    omega_expr = 'c*sqrt({})'.format(k2_expr)
    exp_iwt_expr = 'exp(-1j * {omega} * dt)'.format(omega=omega_expr)
    exp_iwt = ne.evaluate(exp_iwt_expr,
                          local_dict=numexpr_vars,
                          global_dict=None)

    # calculate propagation distance for the moving window
    dz = PhysicalConstants.c * dt

    # process argument moving_window_vect
    if moving_window_vect is not None:
        if len(moving_window_vect) != kspace.dimensions:
            raise ValueError("Argument moving_window_vect has the wrong length. "
                             "Please make sure that len(moving_window_vect) == kspace.dimensions.")

        moving_window_vect = np.asfarray(moving_window_vect)
        moving_window_vect /= npl.norm(moving_window_vect)
        moving_window_dict = dict(enumerate([dz*x for x in moving_window_vect]))

        if remove_antipropagating_waves is None:
            remove_antipropagating_waves = True

        if move_window is None:
            move_window = True

    # remove antipropagating waves, if requested
    if remove_antipropagating_waves:
        if moving_window_vect is None:
            raise ValueError("Missing required argument moving_window_vect.")

        # m = kspace.matrix.copy()
        # m[sum(k*dx for k, dx in zip(kspace.meshgrid(), moving_window_vect)) < 0.0] = 0.0
        # kspace = kspace.replace_data(m)
        arg_expr = '+'.join('({}*k{})'.format(repr(v), i)
                            for i, v
                            in enumerate(moving_window_vect))
        numexpr_vars = dict(kspace=kspace)
        numexpr_vars.update(kdict)
        kspace = kspace.replace_data(ne.evaluate('where({} < 0, 0, kspace)'.format(arg_expr),
                                                 local_dict=numexpr_vars,
                                                 global_dict=None))

    if yield_zeroth_step:
        if do_fft:
            yield kspace.fft()
        else:
            yield kspace

    if move_window:
        if moving_window_vect is None:
            raise ValueError("Missing required argument moving_window_vect.")
        exp_ikdx = linear_phase(kspace, moving_window_dict)

    while True:
        if move_window:
            # Apply the phase due the propagation via the dispersion relation omega
            # and apply the linear phase due to the moving window
            kspace = kspace.replace_data(ne.evaluate('kspace * exp_ikdx * exp_iwt'))

            for i in moving_window_dict.keys():
                kspace.transformed_axes_origins[i] += moving_window_dict[i]

        else:
            # Apply the phase due the propagation via the dispersion relation omega
            kspace = kspace.replace_data(ne.evaluate('kspace * exp_iwt'))

        if do_fft:
            yield kspace.fft()
        else:
            yield kspace


@prepend_doc_of(_kspace_propagate_generator)
def kspace_propagate(kspace, dt, nsteps=1, **kwargs):
    '''
    nsteps: number of steps to take

    If nsteps == 1, this function will just return the result.
    If nsteps > 1, this function will return a generator that will generate the results.
    If you want a list, just put list(...) around the return value.
    '''
    gen = _kspace_propagate_generator(kspace, dt, **kwargs)

    if nsteps == 1:
        return next(gen)

    return itertools.islice(gen, nsteps)
