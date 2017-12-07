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
'''
The postpic.io module provides free functions for importing and exporting data.
'''

import numpy as np


__all__ = ['export_field', 'load_field']


def load_field(filename):
    '''
    construct a new field object from file. currently, the following file
    formats are supported:
    *.npz
    '''
    if not filename.endswith('npz'):
        raise ValueError('File format of filename {0} not recognized.'.format(filename))
    return _import_field_npy(filename)


def export_field(filename, field):
    '''
    export Field object as a file. Format depends on the extention
    of the filename. Currently supported are:
    *.npz
    *.csv
    '''
    if filename.endswith('npz'):
        _export_field_npy(filename, field)
    elif filename.endswith('csv'):
        _export_field_csv(filename, field)
    else:
        raise ValueError('File format of filename {0} not recognized.'.format(filename))


def _export_field_csv(filename, field):
    '''
    Export the data of a Field object as a CSV file.
    The extent will be given in the comments of that file.
    '''
    from . import __version__
    if field.dimensions == 1:
        data = np.asarray(field.matrix)
        extent = field.extent
        header = 'Created with postpic version {0}.\nx = [{1}, {2}]'.format(
            __version__, extent[0], extent[1])
        np.savetxt(filename, data, header=header)
    elif field.dimensions == 2:
        extent = field.extent
        header = 'Created with postpic version {0}.\nx = [{1}, {2}]\ny = [{3}, {4}]'.format(
            __version__, extent[0], extent[1], extent[2], extent[3])
        data = np.asarray(field.matrix)
        np.savetxt(filename, data, header=header)
    else:
        raise Exception('Not Implemented')
    return


def _export_field_npy(filename, field, compressed=True):
    '''
    Export a Field object including all metadata and axes to a file.
    The file will be in the numpy binary format (npz).
    '''

    # collect all metadata into arrays
    # axes objects
    naxes = len(field.axes)

    length_edges = [len(ax.grid_node) for ax in field.axes]
    max_length = np.max(length_edges)

    meta_ax_edges = np.zeros([naxes, max_length])
    meta_ax_names = np.array([''] * naxes)
    meta_ax_units = np.array([''] * naxes)
    meta_ax_transform_state = np.array([False] * naxes)
    meta_ax_transformed_origins = np.array([False] * naxes)

    for nax in range(0, naxes):
        ax = field.axes[nax]
        meta_ax_edges[nax, 0:length_edges[nax]] = ax.grid_node
        meta_ax_names[nax] = str(ax.name)
        meta_ax_units[nax] = str(ax.unit)

    # field metadata
    meta_field = np.array([str(field.name), str(field.unit), str(field.label),
                           str(field.infostring)])
    meta_ax_transform_state = field.axes_transform_state
    meta_ax_transformed_origins = field.transformed_axes_origins

    # save all the data in one file
    savefunc = np.savez_compressed if compressed else np.savez
    savefunc(filename, matrix=field.matrix, meta_field=meta_field,
             meta_ax_edges=meta_ax_edges, meta_ax_names=meta_ax_names,
             meta_ax_units=meta_ax_units,
             meta_length_edges=length_edges,
             meta_ax_transform_state=meta_ax_transform_state,
             meta_ax_transformed_origins=meta_ax_transformed_origins)


def _import_field_npy(filename):
    '''
    import a field object from a file written by _export_field_npy()
    '''
    from .datahandling import Field, Axis
    import_file = np.load(filename)

    # Axes Objects
    length_edges = import_file['meta_length_edges']
    meta_ax_edges = import_file['meta_ax_edges']
    meta_ax_names = import_file['meta_ax_names']
    meta_ax_units = import_file['meta_ax_units']
    meta_ax_transform_state = import_file['meta_ax_transform_state']
    meta_ax_transformed_origins = import_file['meta_ax_transformed_origins']

    axes = []
    for nax in range(0, len(length_edges)):
        axes.append(Axis(name=meta_ax_names[nax],
                         unit=meta_ax_units[nax],
                         grid_node=meta_ax_edges[nax, 0:length_edges[nax]]))

    # field
    meta_field = import_file['meta_field']
    import_field = Field(matrix=import_file['matrix'],
                         name=meta_field[0], unit=meta_field[1],
                         axes=axes,
                         axes_transform_state=meta_ax_transform_state,
                         transformed_axes_origins=meta_ax_transformed_origins)
    import_field.label = meta_field[2]
    import_field.infostring = meta_field[3]

    return import_field


def export_scalar_vtk(filename, scalarfields):
    '''
    exports one or more 2D or 3D scalar field objects to a VTK file
    which is suitable for viewing in ParaView.
    It is assumed that all fields are defined on the same grid.
    '''
    import pyvtk
    import collections
    from .datahandling import Field
    if not isinstance(scalarfields, collections.Iterable):
        if not isinstance(scalarfields, Field):
            raise Exception('scalarfields must be one or more Field objects.')
        else:
            scalarfields = [scalarfields]

    lengths = [len(ax.grid) for ax in scalarfields[0].axes]
    increments = [ax.spacing for ax in scalarfields[0].axes]
    if scalarfields[0].dimensions == 3:
        starts = [scalarfields[0].extent[0], scalarfields[0].extent[2], scalarfields[0].extent[4]]
    elif scalarfields[0].dimensions == 2:
        starts = [scalarfields[0].extent[0], scalarfields[0].extent[2], 0.0]
        lengths.append(1)
        increments.append(1)
    else:
        raise Exception('Only 2D or 3D fields are supported.')

    grid = pyvtk.StructuredPoints(dimensions=lengths, origin=starts, spacing=increments)

    scalar_list = []
    for f in scalarfields:
        scalar_list.append(pyvtk.Scalars(scalars=np.asarray(f).T.flatten(), name=f.name))
    pointData = pyvtk.PointData(*scalar_list)

    vtk = pyvtk.VtkData(grid, pointData)
    vtk.tofile(filename)

    return


def export_vector_vtk(filename, fieldX, fieldY, fieldZ, name=''):
    '''
    exports a vector field to a VTK file suitable for viewing in ParaView.
    Exactly three 3D fields are expected, which will form the X, Y and Z component
    of the vector field.
    '''
    import pyvtk
    from .datahandling import Field

    if not isinstance(fieldX, Field):
        raise Exception('fieldX must be a Field object.')
    if not isinstance(fieldY, Field):
        raise Exception('fieldY must be a Field object.')
    if not isinstance(fieldZ, Field):
        raise Exception('fieldZ must be a Field object.')
    if not fieldX.dimensions == 3:
        raise Exception('fieldX must be a 3D field.')
    if not fieldY.dimensions == 3:
        raise Exception('fieldY must be a 3D field.')
    if not fieldZ.dimensions == 3:
        raise Exception('fieldZ must be a 3D field.')

    lengths = [len(ax.grid) for ax in fieldX.axes]
    increments = [ax.spacing for ax in fieldX.axes]
    starts = [fieldX.extent[0], fieldX.extent[2], fieldX.extent[4]]
    grid = pyvtk.StructuredPoints(dimensions=lengths, origin=starts, spacing=increments)

    if name == '':
        name = fieldX[0].name

    vectors_help = []

    for zidx in range(0, lengths[2]):
        for yidx in range(0, lengths[1]):
            for xidx in range(0, lengths[0]):
                vectors_help.append([np.asarray(fieldX)[xidx, yidx, zidx],
                                     np.asarray(fieldY)[xidx, yidx, zidx],
                                     np.asarray(fieldZ)[xidx, yidx, zidx]])
    pointData = pyvtk.PointData(pyvtk.Vectors(vectors=vectors_help, name=name))
    vtk = pyvtk.VtkData(grid, pointData)
    vtk.tofile(filename)
