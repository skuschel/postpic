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
# Stephan Kuschel, 2017
# Alexander Blinne, 2017
"""
Some experimental algorithms for your reference. Please note that these algorithms
are not meant to be used as is and may need adjustment in order to be applicable
to a wider range of cases.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

from . import helper

__all__ = ['kspace_propagate_adaptive']


def _kspace_propagate_adaptive_generator(field_in, axis=0,
                                         yield_zeroth_step=False):
    """
    An adaptive method to use Fourier propagation (provided by the function
    `helper.kspace_propagate`) to get far field data.
    The field is padded, propagated and automatically sliced in repeating steps.

    Note that this method is highly experimental and should not be trusted as is:
    It is merely meant as a recipe so you don't have to write your own function from scratch!

    `field_in`: input field in either spatial or frequency domain
    `axis`: The direction in which to propagate. Currently only propagation parallel to the
    positive x, y or z direction is implemented.
    `yield_zeroth_step`: boolean that determines if the initial step is also output.
    """
    transform_state = field_in._transform_state()
    if transform_state is None:
        raise ValueError("kspace must have the same transform_state on all axes. "
                         "Please make sure that either all axes 'live' in spatial domain or all "
                         "axes 'live' in frequency domain.")

    do_fft = transform_state

    if do_fft:
        complex_ex = field_in.fft()
    else:
        complex_ex = field_in

    moving_window_vect = [0]*complex_ex.dimensions
    moving_window_vect[axis] = 1

    t = 0.0
    x0 = np.mean(complex_ex.axes[axis].extent)

    if yield_zeroth_step:
        yield t, field_in

    while True:
        complex_ex = complex_ex.autoreduce()

        # print(complex_ex.axes[0].physical_length, complex_ex.axes[1].physical_length)
        lengths = [ax.physical_length for ax in complex_ex.axes]
        transv_lengths = lengths[:]
        del transv_lengths[axis]

        distance = 0.5*np.mean(transv_lengths)
        timestep = 1.0*distance / helper.PhysicalConstants.c

        # propagation distance since beginning
        x = np.mean(complex_ex.axes[axis].extent) - x0 + distance

        # box length in propagation direction
        xlen = complex_ex.axes[axis].physical_length

        # padding in prop. dir
        xpad = distance / x * (0.5*xlen)

        # pad distance in transverse directions and xpad in prop. dir
        pad = [distance] * complex_ex.dimensions
        pad[axis] = [xpad, 0.1*xpad]

        # do padding and fft autopadding
        complex_ex = complex_ex.pad(pad).fft_autopad()

        # do propagation
        complex_ex = helper.kspace_propagate(complex_ex, timestep,
                                             moving_window_vect=moving_window_vect)
        t += timestep

        # remove low field strength outer region
        complex_ex = complex_ex.autocutout(fractions=(0.01, 0.02))

        if do_fft:
            yield t, complex_ex.fft()
        else:
            yield t, complex_ex


@helper.prepend_doc_of(_kspace_propagate_adaptive_generator)
def kspace_propagate_adaptive(field_in, axis=0, t_final=None, **kwargs):
    """
    `t_final`: The time at which to stop the adaptive propagation.
    """
    gen = _kspace_propagate_adaptive_generator(field_in, axis=axis, **kwargs)
    while True:
        t, f = next(gen)
        yield t, f
        if t_final and t > t_final:
            break
