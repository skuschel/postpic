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

from .csv import _export_field_csv, _import_field_csv
from .npy import _export_field_npy, _import_field_npy
from .vtk import export_scalar_vtk, export_vector_vtk, export_scalars_vtk
from .image import _import_field_image

__all__ = ['export_field', 'load_field',
           'export_scalar_vtk', 'export_scalars_vtk', 'export_vector_vtk']


def load_field(filename):
    '''
    Load a field object previously stored using the `saveto` method. These are .npz files
    with a specific metadata layout.
    '''
    if not filename.lower().endswith('npz'):
        raise ValueError('File format of filename {0} not recognized.'.format(filename))
    return _import_field_npy(filename)


def import_field(filename, **kwargs):
    '''
    Construct a new field object from foreign data. Currently, the following file
    formats are supported explicitly:
    *.csv
    *.png

    All other files will be opened using Pillow.
    '''
    if filename.lower().endswith('csv'):
        return _import_field_csv(filename, **kwargs)
    else:
        # assume anything else, this will open png files with pypng and all other files
        # with pillow
        return _import_field_image(filename, **kwargs)


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
    if filename.lower().endswith('npz'):
        _export_field_npy(filename, field, **kwargs)
    elif filename.lower().endswith('csv'):
        _export_field_csv(filename, field, **kwargs)
    elif filename.lower().endswith('vtk'):
        export_scalar_vtk(filename, field, **kwargs)
    else:
        raise ValueError('File format of filename {0} not recognized.'.format(filename))
