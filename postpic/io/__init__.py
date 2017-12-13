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

from .csv import _export_field_csv
from .npy import _export_field_npy, _import_field_npy
from .vtk import export_scalar_vtk, export_vector_vtk

__all__ = ['export_field', 'load_field',
           'export_scalar_vtk', 'export_vector_vtk']


def load_field(filename):
    '''
    construct a new field object from file. currently, the following file
    formats are supported:
    *.npz
    '''
    if not filename.endswith('npz'):
        raise ValueError('File format of filename {0} not recognized.'.format(filename))
    return _import_field_npy(filename)


def export_field(filename, field, **kwargs):
    '''
    export Field object as a file. Format depends on the extention
    of the filename. Currently supported are:
    .npz:
        uses `numpy.savez`.
    .csv:
        uses `numpy.savetxt`.
    .vtk:
        vtk export to paraview
    '''
    if filename.endswith('npz'):
        _export_field_npy(filename, field, **kwargs)
    elif filename.endswith('csv'):
        _export_field_csv(filename, field, **kwargs)
    elif filename.endswith('vtk'):
        export_scalar_vtk(filename, field, **kwargs)
    else:
        raise ValueError('File format of filename {0} not recognized.'.format(filename))
