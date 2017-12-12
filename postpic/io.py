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

import numpy as np


__all__ = ['export_field', 'load_field',
           'export_scalar_vtk', 'export_vector_vtk']


def _header_string():
    '''
    creates a header string for general export information.
    '''
    from . import __version__
    import datetime
    now = str(datetime.datetime.now())
    ret = '''
    This file was written by postpic {v:}
    --- the open-source particle-in-cell post-processor. ---
    https://github.com/skuschel/postpic

    written on {now:}\n
    '''
    return ret.format(v=__version__, now=now)


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
             meta_ax_transformed_origins=meta_ax_transformed_origins,
             header=_header_string())


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


def export_scalar_vtk(filename, scalarfield):
    '''
    exports one 2D or 3D scalar field object to a VTK file
    which is suitable for viewing in ParaView.
    It is assumed that all fields are defined on the same grid.
    '''
    import pyvtk
    import collections
    from .datahandling import Field
    if not isinstance(scalarfield, Field):
        raise Exception('scalarfield must be one or more Field objects.')
    if scalarfield.dimensions == 1:
        raise ValueError('Cannot export 1D Field.')

    scalarfield = scalarfield.atleast_nd(3)
    if all(ax.islinear() for ax in scalarfield.axes):
        lengths = [len(ax) for ax in scalarfield.axes]
        increments = [ax.spacing for ax in scalarfield.axes]
        starts = [ax.grid[0] for ax in scalarfield.axes]
        grid = pyvtk.StructuredPoints(dimensions=lengths, origin=starts, spacing=increments)
    else:
        grid = pyvtk.RectilinearGrid(*scalarfield.grid)

    grid = pyvtk.StructuredPoints(dimensions=lengths, origin=starts, spacing=increments)

    scalar_list = pyvtk.Scalars(scalars=np.ravel(scalarfield, order='F'), name=scalarfield.name)
    pointData = pyvtk.PointData(scalar_list)

    vtk = pyvtk.VtkData(grid, pointData)
    vtk.tofile(filename, 'binary')

    return


def _make_vectors_help(*fields):
    return np.stack((np.ravel(f, order='F') for f in fields), axis=-1)


def export_vector_vtk(filename, *fields, **kwargs):
    '''
    exports a vector field to a VTK file suitable for viewing in ParaView.
    Three 3D fields are expected, which will form the X, Y and Z component
    of the vector field. If less than tree fields are given, the missing components
    will be assumed to be zero.
    '''
    import pyvtk
    from .datahandling import Field

    name = kwargs.pop('name', '')
    fields = [field if isinstance(field, Field) else Field(field) for field in fields]
    shape = fields[0].shape

    if not all(shape == field.shape for field in fields):
        raise ValueError("All fields must have the same shape")

    if len(fields) > 3:
        raise ValueError("Too many fields")

    while len(fields) < 3:
        fields.append(fields[0].replace_data(np.zeros_like(fields[0])))

    if len(shape) > 3:
        raise ValueError("Fields have to many axes")

    fields = [f.atleast_nd(3) for f in fields]

    if all(ax.islinear() for ax in fields[0].axes):
        lengths = [len(ax) for ax in fields[0].axes]
        increments = [ax.spacing for ax in fields[0].axes]
        starts = [ax.grid[0] for ax in fields[0].axes]
        grid = pyvtk.StructuredPoints(dimensions=lengths, origin=starts, spacing=increments)
    else:
        grid = pyvtk.RectilinearGrid(*fields[0].grid)

    if name == '':
        name = fields[0].name

    vectors_help = _make_vectors_help(*[np.asarray(f) for f in fields])

    pointData = pyvtk.PointData(pyvtk.Vectors(vectors=vectors_help, name=name))
    vtk = pyvtk.VtkData(grid, pointData)
    vtk.tofile(filename, 'binary')
