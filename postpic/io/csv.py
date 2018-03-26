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
# Stefan Tietze, 2017
# Alexander Blinne, 2017
# Stephan Kuschel, 2017
'''
The postpic.io module provides free functions for importing and exporting data.
'''

import os.path as osp

import numpy as np

from .common import _header_string


def _export_field_csv(filename, field):
    '''
    Export the data of a Field object as a CSV file.
    The extent will be given in the comments of that file.
    '''
    header = _header_string()
    data = np.asarray(field)
    header += 'Data extent: {}\n'.format(field.extent)
    np.savetxt(filename, data, header=header)
    return


def _import_field_csv(filename, delimiter=','):
    '''
    reads a .csv file using numpy.genfromtxt.

    Args:
        filename (str): Path and filename to the file to open

    Returns:
        numpy.array: the image data as numpy array converted to float64

    Author: Stephan Kuschel, 2015-2016, Alexander Blinne 2017
    '''
    from ..datahandling import Field, Axis

    ret = np.genfromtxt(filename, delimiter=delimiter)
    axes = [Axis(name=name, unit='px', grid=np.linspace(0, ret.shape[i]-1, ret.shape[i]))
            for i, name in enumerate('xy')]
    basename = osp.basename(filename)
    return Field(ret, name=basename, unit='?', axes=axes)
