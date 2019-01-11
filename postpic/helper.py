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
import collections
import numbers
import numpy as np
import scipy as sp
import numpy.linalg as npl
from ._compat import meshgrid, moveaxis, broadcast_to
import re
import warnings
import functools
import math
import numexpr as ne


__all__ = ['PhysicalConstants', 'unstagger_fields', 'kspace_epoch_like', 'kspace',
           'kspace_propagate', 'time_profile_at_plane']


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


def deprecated(msg, v='unknown'):
    '''
    Mark functions as deprecated by using this decorator.
    msg is an additioanl message that will be displayed.
    '''
    def _deprecated(func):
        d = dict(msg=msg, name=func.__name__)
        if msg is None:
            s = 'The function {name} is deprecated.'.format(**d)
        else:
            # format 2 times, so {name} can be used within {msg}
            s = "The function {name} is deprecated. {msg}".format(**d).format(**d)
        deprdoc = '''
                .. deprecated:: {}
                    {}
                '''.format(v, s)

        @functools.wraps(func)
        def ret(*args, **kwargs):
            warnings.warn(s, category=DeprecationWarning)
            return func(*args, **kwargs)
        ret.__doc__ = deprdoc if ret.__doc__ is None else ret.__doc__ + deprdoc
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


# Some static functions
def polar2linear(theta, r):
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    return x, y


def polar2linear_jac(theta, r):
    x_theta = -r*np.sin(theta)
    x_r = np.cos(theta)
    y_theta = r*np.cos(theta)
    y_r = np.sin(theta)
    return [[x_theta, x_r], [y_theta, y_r]]


def polar2linear_jacdet(theta, r):
    return r


def linear2polar(x, y):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return theta, r


def jac_det(jacobian_func):
    '''
    Calculate the determinant of the jacobian as returned by jacobian_func.
    Example:

    def polar2linear_jac(theta, r):
        x_theta = -r*np.sin(theta)
        x_r = np.cos(theta)
        y_theta = r*np.cos(theta)
        y_r = np.sin(theta)
        return [[x_theta, x_r], [y_theta, y_r]]

    det_fun = jac_det(polar2linear_jac)
    def = det_fun(theta, r)
    '''
    def fun(*coords):
        jac = jacobian_func(*coords)
        shape = np.broadcast(*coords).shape
        jacarray = np.asarray([[broadcast_to(a, shape) for a in row] for row in jac])
        jacarray = moveaxis(jacarray, [0, 1], [-2, -1])
        return abs(npl.det(jacarray))
    return fun


def islinear(grid):
    return np.all(np.isclose(grid, np.linspace(grid[0], grid[-1], len(grid))))


def monotonicity(arr, axis=-1):
    """
    Checks if an array is strictly monotonically increasing or decreasing.
    arr:  Array to be tested
    axis: axis along which monotonicality is to be tested. Like np.diff() this
          defaults to the last axis.
    Returns "1" for a strictly monotonically increasing array, "-1" for a strictly
    monotonically decreasing array and "0" for an array that is neither.
    """
    dx = np.diff(arr, axis=axis)
    if np.all(dx > 0.0):
        return 1
    if np.all(dx < 0.0):
        return -1
    return 0


def approx_jacobian(transform):
    '''
    Approximate the jacobian of the transformation given by transform.
    Example:

    def polar2linear(theta, r):
        x = r*np.cos(theta)
        y = r*np.sin(theta)
        return x, y

    polar2linear_jac_approx = approx_jacobian(polar2linear)
    jacobian = polar2linear_jac_approx(theta, r)
    '''
    def fun(*coords):
        ravel_coords = [np.ravel(c) for c in coords]

        from pkg_resources import parse_version
        if parse_version(np.__version__) < parse_version('1.13'):
            if not all(islinear(r) for r in ravel_coords):
                raise NotImplemented('Numerically approximating the Jacobian on a transform to a '
                                     'non-equidistant grid is not implemented for numpy < 1.13.')
            ravel_coords = [(r[-1]-r[0])/(len(r)-1) for r in ravel_coords]

        shape = np.broadcast(*coords).shape
        mapped_coords = transform(*coords)
        mapped_coords = [broadcast_to(c, shape) for c in mapped_coords]
        jac = [np.gradient(c, *ravel_coords) for c in mapped_coords]
        return jac
    return fun


def approx_1d_jacobian_det(transform):
    '''
    Approximate the "Jacobian determinant" of a 1d transformation, which is basically
    just the derivative.
    '''
    def fun(coords):
        epsilon = np.sqrt(np.finfo(float).eps)
        dx = epsilon + epsilon * coords
        return abs((transform(coords + dx) - transform(coords - dx)) / (2.0 * dx))
    return fun


def is_non_integer_real_number(x):
    """
    Tests if an object ix is a real number and not an integer.
    """
    return isinstance(x, numbers.Real) and not isinstance(x, numbers.Integral)


def max_frac_bounds(array, fraction):
    """
    For a 1d Array `array` this function gives indices `a`, `b` such that all values
    `array[:a]` and `array[b:]` are smaller than `fraction*max(array)`
    """
    array = np.asarray(array)
    c = np.max(array)*fraction
    i = np.nonzero(array > c)[0]
    return i[0], i[-1]+1


def product(iterable):
    """
    Calculate the cumulative product of objects from iterable.
    This uses the first object from the iterable as a starting point and thus the iterable
    must have at least one object in it, otherwise the function will fail.

    Example: product(range(n)) == math.factorial(n)
    """
    i = iter(iterable)
    p = next(i)
    for x in i:
        p = p * x
    return p


class FFTW_Pad(object):
    """
    FFTW_Pad is a class whichs objects are callables that are suitable as `fft_padsize`
    arguments to `Field.fft_autopad` and calculate optimal padding sizes for FFTW.
    """
    def __init__(self, fftsize_max=None, factors=(2, 3, 5, 7, 11, 13)):
        '''
        Calculate all 'good' sizes up to fftsize_max, using the given factors.
        While at it, make sure that at most one factor 11 or 13 exists.

        FFTW documentation says: "FFTW is best at handling sizes of the form 2^a 3^b 5^c
        7^d 11^e 13^f, where e+f is either 0 or 1, and the other exponents are arbitrary."
        '''
        self.factors = factors
        self.fftsize_max = 1 if fftsize_max is None else fftsize_max

    @property
    def fftsize_max(self):
        return self._fftsize_max

    @fftsize_max.setter
    def fftsize_max(self, fftsize_max):
        self._fftsize_max = fftsize_max

        extra_factors = [1]
        factors = list(self.factors)
        for x in (11, 13):
            if x in factors:
                extra_factors.append(x)
                factors.remove(x)

        fftsizes = []
        for extra_factor in extra_factors:
            max_powers = [int(math.log(fftsize_max/extra_factor+1, i))+1 for i in factors]
            powers_ranges = map(range, max_powers)
            fftsizes.extend(extra_factor*product(f**p for f, p in zip(factors, powers))
                            for powers
                            in itertools.product(*powers_ranges)  # build all combination of pwrs
                            )
        self.fftsizes = np.array(list(sorted(filter(lambda x: x <= fftsize_max, fftsizes))))

    def __call__(self, n):
        '''
        In the list of sizes calculated at initialization, find the next good value equal
        or larger than a given `n`.
        '''
        if n > self.fftsizes[-1]:
            # make sure values are calculated, such that a good fftsize
            # larger than n can be found. +10000 sould be safe for that purpose
            self.fftsize_max = n + 10000
        i = np.searchsorted(self.fftsizes, n)
        return self.fftsizes[i]


fftw_padsize = FFTW_Pad()


def fft_padsize_power2(n):
    return 1 << n.bit_length()


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


def omega_free(mesh):
    """
    Calculate the free space (vacuum) dispersion relation on the k-mesh `mesh`.

    `mesh`: a mesh grid of the k vector space, typically a sparse grid as provided by
    Field.meshgrid().
    """
    k2 = sum(ki**2 for ki in mesh)
    return PhysicalConstants.c * np.sqrt(k2)


def unstagger_fields(*fields, **kwargs):
    '''
    Unstagger a collection of fields.

    This functions shifts the origins of the grids of the given fields such that they coincide.
    Since the choice of the common origin is somewhat arbitrary, it might be overriden by a
    keyword-argument `origin`, as may be the interpolation `method`. See `Field.shift_grid_by`
    for available methods.
    '''
    method = kwargs.pop('method', "fourier")
    origin = kwargs.pop('origin', None)

    if not all([field.shape == fields[0].shape for field in fields]):
        raise ValueError("Fields have different shapes")

    if not all([all(field.islinear()) for field in fields]):
        raise ValueError("Fields have non-linear axes")

    spacing = [ax.spacing for ax in fields[0].axes]
    if not all([np.all(np.isclose(spacing, [ax.spacing for ax in field.axes]))
                for field in fields]):
        raise ValueError("Fields have unequal grid spacing")

    if origin is None:
        origins = np.array([[ax.grid[0] for ax in field.axes]
                            for field in fields
                            ])
        if len(fields) > 2:
            origin = np.median(origins, axis=0)
        else:
            origin = origins[0, :]

    new_fields = []
    for field in fields:
        fo = np.array([ax.grid[0] for ax in field.axes])
        if np.all(np.isclose(origin, fo)):
            new_fields.append(field)
            continue

        dx = origin-fo
        if np.any(abs(dx/spacing) > 2):
            raise ValueError('Distance of grids is larger than twice the grid spacing')

        nf = field.shift_grid_by(dx, method)
        if np.isrealobj(field):
            nf = nf.real
        new_fields.append(nf)
    return new_fields


def _kspace_helper_cutfields(component, fields, extent):
    slices = fields[component]._extent_to_slices(extent)
    return {k: f[slices] for k, f in fields.items()}


@deprecated("This function is left in postpic only for comparison. Use `kspace_epoch_like` "
            "for real work.")
def kspace_epoch_like_old(component, fields, extent=None, omega_func=omega_free, align_to='B'):
    '''
    Reconstruct the physical kspace of one polarization component
    See documentation of kspace

    This will choose the alignment of the fields in a way to improve
    accuracy on EPOCH-like staggered dumps.

    This is the old version of the function and will be removed once the new method
    is sufficiently tested

    For the current version of EPOCH, v4.9, use the following:
    align_to == 'B' for intermediate dumps, align_to == "E" for final dumps
    '''
    polfield = component[0]
    polaxis = axesidentify[component[1]]

    # apply extent to all fields
    if extent is not None:
        fields = {k: v.ensure_spatial_domain() for k, v in fields.items()}
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


def _linear_interpolation_frequency_response(dt, a=0.5):
    """
    Calculate the frequency response of a convolution with a [1-a, a] kernel, which is
    basically a linear interpolation.
    Assume a grid-step of `dt` and use a grid which `n` points.

    `dt`: physical grid-step on which linear interpolation is done
    `a`: shift distance of the linear interpolation in units of dt

    Returns a function which takes omega as an input.
    """
    def f(omega):
        return (1-a)+a*np.exp(1j*dt*omega)

    return f


def _linear_interpolation_frequency_response_on_k(lin_response_omega, k_axes, omega_func):
    """
    Remap the frequency response `lin_response_omega` from frequencies to wave-vectors.

    `lin_response_omega`: frequency response function depending on omega, e.g. output of
                          `_linear_interpolation_frequency_response`.
    `k_axes`: A list of axes objects to map the response to, e.g. Field.axes
    `omega_func`: The dispersion relation used to map k vectors to omega, e.g.
                  `omega_yee_factory(dx, dt)`.

    Returns the function f(k) as a Field object.
    """
    from . import datahandling

    kmesh = meshgrid(*[ax.grid for ax in k_axes], indexing='ij', sparse=True)

    resp_mat = abs(lin_response_omega(omega_func(kmesh)))

    lin_res_k = datahandling.Field(resp_mat, name='f', axes=k_axes)

    return lin_res_k


def kspace_epoch_like(component, fields, dt, extent=None, omega_func=omega_free, align_to='B'):
    '''
    Reconstruct the physical kspace of one polarization component
    See documentation of kspace

    This function will use special care to make sure, that the implicit linear interpolation
    introduced by Epochs half-steps will not impede the accuracy of the reconstructed k-space.
    The frequency response of the linear interpolation is modelled and removed from the
    interpolated fields.

    `dt`: time-step of the simulation, this is used to calculate the frequency response due
    to the linear interpolated half-steps

    For the current version of EPOCH, v4.9, use the following:
    align_to == 'B' for intermediate dumps, align_to == "E" for final dumps

    As of Jan 2019 the devel branch contains a change, that will modify the behaviour of final
    dumps to be the same as intermediate dumps. This change is supposed to be released with
    Epoch v5.0. See https://cfsa-pmw.warwick.ac.uk/EPOCH/epoch/issues/1896 for details. From
    Epoch v5.0 onwards, align_to should always be set to 'B'.
    '''
    polfield = component[0]
    polaxis = axesidentify[component[1]]

    main_field_key = component
    other_field_keys = list(fields.keys())
    other_field_keys.remove(main_field_key)

    # apply extent to all fields
    if extent is not None:
        fields = {k: v.ensure_spatial_domain() for k, v in fields.items()}
        fields = _kspace_helper_cutfields(component, fields, extent)

    # for k, v in fields.items():
    #     print(k, v.extent, [(a[0], a[-1]) for a in v._conjugate_grid().values()])

    fields = {k: v.ensure_frequency_domain() for k, v in fields.items()}

    lin_res = _linear_interpolation_frequency_response(dt)
    lin_res_k = _linear_interpolation_frequency_response_on_k(lin_res, fields[main_field_key].axes,
                                                              omega_func)

    if polfield != align_to:
        for c in other_field_keys:
            # print('apply lin_response to ', c, 'transform_state is',
            #       fields[c]._transform_state())
            fields[c] = fields[c] / lin_res_k
    else:
        # print('apply lin_response to ', main_field_key, 'transform_state is',
        #       fields[main_field_key]._transform_state())
        fields[main_field_key] = fields[main_field_key] / lin_res_k

    # for k, v in fields.items():
    #     print(k, v.extent, [(a[0], a[-1]) for a in v._conjugate_grid().values()])

    return kspace(component, fields, interpolation='fourier')


def kspace(component, fields, extent=None, interpolation=None, omega_func=omega_free):
    '''
    Reconstruct the physical kspace of one polarization component
    This function basically computes one component of
        E = 0.5*(E - omega/k^2 * Cross[k, B])
    or
        B = 0.5*(B + 1/omega * Cross[k, E]).

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
        fields = {k: v.ensure_spatial_domain() for k, v in fields.items()}
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

    # Change to frequency domain
    result = result.ensure_frequency_domain()

    result_spatial_grid = result._conjugate_grid()
    result_spatial_grid = [result_spatial_grid[k] for k in sorted(result_spatial_grid.keys())]

    # remember the origins of result's axes to compare with other fields
    result_origin = [g[0] for g in result_spatial_grid]

    # store box size of input field
    Dx = np.array([g[-1] - g[0] for g in result_spatial_grid])

    # store grid spacing of input field
    dx = np.array([g[1] - g[0] for g in result_spatial_grid])

    # print('result_origin', result_origin, Dx, dx)

    # calculate the k mesh and k^2
    mesh = meshgrid(*[ax.grid for ax in result.axes], indexing='ij', sparse=True)
    k2 = sum(ki**2 for ki in mesh)

    # calculate omega, either using the vacuum expression or omega_func()
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

            field_transform_state = field._transform_state()
            if field_transform_state is None:
                if interpolation == 'linear':
                    field = field.ensure_spatial_domain()
                else:
                    field = field.ensure_frequency_domain()

            field_transform_state = field._transform_state()
            if field_transform_state is True:
                field_spatial_grid = field._conjugate_grid()
                field_spatial_grid = [field_spatial_grid[k] for k in
                                      sorted(field_spatial_grid.keys())]

                # remember the origins of result's axes to compare with other fields
                field_origin = [g[0] for g in field_spatial_grid]

                # store box size of input field
                oDx = np.array([g[-1] - g[0] for g in field_spatial_grid])

                # store grid spacing of input field
                # odx = np.array([g[1] - g[0] for g in field_spatial_grid])
            else:
                field_origin = [a.grid[0] for a in field.axes]

                # remember the origin and box size of the field
                oDx = np.array([a.grid[-1] - a.grid[0] for a in field.axes])

            # Test if all fields have the same number of grid points
            if not field.shape == result.shape:
                raise ValueError("All given Fields must have the same number of grid points. "
                                 "Field {} has a different shape than {}.".format(field_key,
                                                                                  component))

            # Test if the axes of all fields have the same lengths
            if not np.all(np.isclose(Dx, oDx)):
                # print(Dx, oDx)
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
                field = field.ensure_spatial_domain().shift_grid_by(grid_shift,
                                                                    interpolation='linear')

            field = field.ensure_frequency_domain()

            # fourier interpolation is done after the fft by applying a linear phase
            if interpolation == 'fourier':
                # print('apply linear phase')
                field = field._apply_linear_phase(dict(enumerate(grid_shift)))

            # add the field to the result with the appropriate prefactor
            # result.matrix += (-1)**(i-1) * prefactor * mesh[mesh_i] * field.matrix
            mesh_mesh_i = mesh[mesh_i]
            rm = result.matrix
            fm = field.matrix
            result.matrix = ne.evaluate('rm + (-1)**(i-1) * prefactor * mesh_mesh_i * fm')

    return result


def linear_phase(field, dx):
    '''
    Calculates the linear phase as used in Field._apply_linear_phase and
    _kspace_propagate_generator.
    '''
    transform_state = field._transform_state(dx.keys())
    axes = [ax.grid.copy() for ax in field.axes]
    for i in range(len(axes)):
        gridlen = len(axes[i])
        if transform_state is True:
            # center the axes around 0 to eliminate global phase
            axes[i] -= axes[i][gridlen//2]
        else:
            # start the axes at 0 to eliminate global phase
            axes[i] -= axes[i][0]

    # build mesh
    mesh = meshgrid(*axes, indexing='ij', sparse=True)

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
                                yield_zeroth_step=False,
                                use_numexpr_in_inner_loop=True):
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
        exp_ikdx_iwt = ne.evaluate('exp_iwt * exp_ikdx')

    while True:
        if move_window:
            # Apply the phase due the propagation via the dispersion relation omega
            # and apply the linear phase due to the moving window
            if use_numexpr_in_inner_loop:
                kspace = kspace.replace_data(ne.evaluate('kspace * exp_ikdx_iwt'))
            else:
                kspace = kspace * exp_ikdx_iwt

            for i in moving_window_dict.keys():
                kspace.transformed_axes_origins[i] += moving_window_dict[i]

        else:
            # Apply the phase due the propagation via the dispersion relation omega
            if use_numexpr_in_inner_loop:
                kspace = kspace.replace_data(ne.evaluate('kspace * exp_iwt'))
            else:
                kspace = kspace * exp_iwt

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


def time_profile_at_plane(kspace_or_complex_field, axis='x', value=None, dir=1, t_input=0.0,
                          **kwargs):
    '''
    'Measure' the time-profile of the propagating `complex_field` while passing through a plane.

    The arguments `axis`, `value` and `dir` specify the plane and main propagation direction.

    `axis` specifies the axis perpendicular to the measurement plane.

    `dir=1` specifies propagation towards positive `axis`, `dir=-1` specifies the opposite
    direction of propagation.

    `value` specifies the position of the plane along `axis`. If `value=None,` a default is chosen,
    depending on `dir`.

    If `dir=-1`, the starting point of the axis is used, which lies at the 0-component of the
    inverse transform.

    If `dir=1`, the end point of the axis + one axis spacing is used, which, via periodic boundary
    conditions of the fft, also lies at the 0-component of the inverse transform.

    If the given `value` differs from these defaults, an initial propagation with moving window
    will be performed, such that the desired plane lies in the default position.

    t_input specifies the point in time at which the input field or kspace is given. This is used
    to specify the time axis of the output fields.

    For example `axis='x'` and `value=0.0` specifies the 'x=0.0' plane while `dir=1` specifies
    propagation towards positive 'x' values. The 'x' axis starts at 2e-5 and ends at 6e-5 with
    a grid spacing of 1e-6. The default value for the measurement plane would have been 6.1e-5
    so an initial backward propagation with dt = -6.1e-5/c is performed to move the pulse in front
    of the'x=0.0 plane.

    Additional `kwargs` are passed to kspace_propagate if they are not overridden by this function.
    '''
    # can't import this at top of module because this would create a circular import
    # importing here is ok, because helper and datahandling are both already interpreted
    from . import datahandling

    transform_state = kspace_or_complex_field._transform_state()
    if transform_state is None:
        raise ValueError("kspace_or_complex_field must have the same transform_state on all axes. "
                         "Please make sure that either all axes 'live' in spatial domain or all "
                         "axes 'live' in frequency domain.")

    do_fft = not transform_state

    if do_fft:
        kspace = kspace_or_complex_field.fft()
        complex_field = kspace_or_complex_field
    else:
        kspace = kspace_or_complex_field
        complex_field = kspace_or_complex_field.fft()

    # interpret axis
    axis = axesidentify[axis]
    otheraxes = list(range(kspace.dimensions))
    otheraxes.remove(axis)

    dr = complex_field.axes[axis].spacing
    dt = dr/PhysicalConstants.c

    dV = kspace.axes[axis].spacing
    N = kspace.shape[axis]
    V = dV*N
    Vk = 2*np.pi/dV
    fftnorm = np.sqrt(V/Vk) / np.sqrt(N)

    # apply the fft norm just once
    kspace = kspace * fftnorm

    # updating the kwargs for kspace_propagate
    kwargs['moving_window_vect'] = [0]*kspace.dimensions
    kwargs['moving_window_vect'][axis] = dir
    kwargs['move_window'] = True

    # only remove backwards-propagating waves:
    kwargs['nsteps'] = 1
    kwargs['yield_zeroth_step'] = False

    initial_dt = 0.0
    # do an initial propagation with moving window to align the data with the measuring plane
    if value is not None:
        if dir > 0:
            # measuring at the 0-component is just like measuring at the end of the grid + one
            # axis spacing if propagating is assumed to be in positive axis direction
            r = complex_field.axes[axis].grid[-1] + complex_field.axes[axis].spacing
        else:
            # measuring at the 0-component is just like measuring at the beginning of the grid
            # if propagating is assumed to be in negative axis direction

            r = complex_field.axes[axis].grid[0]

        # do propagating of initial_dt such that the measurement plane is at `value`
        initial_dt = dir * (value-r) / PhysicalConstants.c

    kspace = kspace_propagate(kspace, initial_dt, **kwargs)

    # setup a generator for the propagated kspaces
    kwargs['nsteps'] = len(complex_field.axes[axis])
    kwargs['move_window'] = False
    kwargs['yield_zeroth_step'] = True
    gen = kspace_propagate(kspace, dt, **kwargs)

    # initialize an empty matrix
    newmat = np.empty_like(kspace.matrix)
    slices = [slice(None)] * kspace.dimensions
    expr = 'sum(km, {})'.format(axis)
    for i, kspace_prop in enumerate(gen):
        # fill the new matrix line by line by calculating the 0-component of the inverse
        # transform after each propagation step
        slices[axis] = i
        km = kspace_prop.matrix
        # newmat[slices] = np.sum(kspace_prop.matrix, axis=axis)
        newmat[slices] = ne.evaluate(expr)

    k_transverse_tprofile = kspace.replace_data(newmat)
    t_axis = datahandling.Axis(name='t', unit='s',
                               grid=np.linspace(t_input + initial_dt,
                                                t_input + initial_dt + (N-1)*dt,
                                                N))
    k_transverse_tprofile.setaxisobj(axis, t_axis)
    k_transverse_tprofile.axes_transform_state[axis] = False
    k_transverse_tprofile.transformed_axes_origins[axis] = None

    if do_fft:
        return k_transverse_tprofile.fft(otheraxes)
    else:
        return k_transverse_tprofile
