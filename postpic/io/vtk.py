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

import warnings

import numpy as np

from ..helper import product


class DataSet(object):
    '''
    Superclass to represent different vtkDataSets
    '''
    pass


class StructuredPoints(DataSet):
    '''
    Class to represent a vtkStructuredPoints
    '''
    def __init__(self, dimensions, origin, spacing):
        if len(dimensions) != 3 or len(origin) != 3 or len(spacing) != 3:
            raise ValueError('All arguments must have len(...) == 3.')
        self.dimensions = dimensions
        self.origin = origin
        self.spacing = spacing

    @classmethod
    def from_field(cls, field):
        if field.dimensions != 3:
            raise ValueError("Field must have three dimensions.")
        dimensions = [len(ax) for ax in field.axes]
        origin = [ax.grid[0] for ax in field.axes]
        spacing = [ax.spacing for ax in field.axes]
        return cls(dimensions, origin, spacing)

    def tofile(self, vtk):
        vtk.file.write(b'DATASET STRUCTURED_POINTS\n')
        vtk.file.write('DIMENSIONS {} {} {}\n'.format(*self.dimensions).encode('ascii'))
        vtk.file.write('ORIGIN {} {} {}\n'.format(*self.origin).encode('ascii'))
        vtk.file.write('SPACING {} {} {}\n'.format(*self.spacing).encode('ascii'))


class RectilinearGrid(DataSet):
    '''
    Class to represent a vtkRectilinearGrid
    '''
    def __init__(self, grid):
        if len(grid) != 3:
            raise ValueError('Grid must have three axes')
        self.grid = grid

    @classmethod
    def from_field(cls, field):
        return cls(field.grid)

    def tofile(self, vtk):
        vtk.file.write(b'DATASET RECTILINEAR_GRID\n')
        vtk.file.write('DIMENSIONS {} {} {}\n'.format(*(len(g) for g in self.grid))
                       .encode('ascii'))
        for axname, axis in zip('XYZ', self.grid):
            vtk.file.write('{}_COORDINATES {} {}\n'.format(axname, len(axis), vtk.type)
                           .encode('ascii'))
            try:
                # ndarray.tobytes() was introduced in numpy 1.9
                axis = axis.astype(vtk.dtype).tobytes()
            except AttributeError:
                # workaround for numpy <1.9, but only works in python 3
                axis = axis.astype(vtk.dtype).data.tobytes()
            vtk.file.write(axis)
            vtk.file.write(b'\n')


class VtkFile(object):
    '''
    Class used to write a .vtk file.
    Used by `VtkData`.
    '''
    def __init__(self, fname, type='double', mode='binary'):
        self.fname = fname

        self.mode = mode
        if self.mode != 'binary':
            raise NotImplemented("Only 'mode'=='binary' is implemented.")

        self.type = type
        if type == 'double':
            self.dtype = np.dtype('>f8')
        elif type == 'float':
            self.dtype = np.dtype('>f4')
        else:
            raise ValueError("Invalid 'type', must be 'float' or 'double'.")

    def __enter__(self):
        self.file = open(self.fname, 'wb')
        self.file.write(b'# vtk DataFile Version 2.0\n')
        self.file.write(b'PostPic exported data\n')
        self.file.write('{}\n'.format(self.mode.upper()).encode('ascii'))
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.file.close()


class VtkData(object):
    '''
    Class to represent the data that should be written to a .vtk file.
    Uses `VtkFile`.
    '''
    def __init__(self, dataset, *data):
        self.dataset = dataset
        if not isinstance(dataset, DataSet):
            raise TypeError("dataset must be of type DataSet")

        self.data = data
        if not all(isinstance(d, Data) for d in data):
            raise TypeError("all elements of data must be of type Data")

    def tofile(self, fname, type='double', mode='binary'):
        with VtkFile(fname, mode=mode, type=type) as vtk:
            self.dataset.tofile(vtk)
            for d in self.data:
                d.tofile(vtk)


class Data(object):
    '''
    Superclass to represent the attributed data associated with a `DataSet`.
    '''
    def __init__(self, arraydata):
        self.arraydata = arraydata

    def tofile(self, vtk):
        self.arraydata.tofile(vtk)


class PointData(Data):
    '''
    PointData associated with a `DataSet`
    '''
    def tofile(self, vtk):
        vtk.file.write('POINT_DATA {}\n'.format(len(self.arraydata)).encode('ascii'))
        super(PointData, self).tofile(vtk)


class CellData(Data):
    '''
    CellData associated with a `DataSet`
    '''
    def tofile(self, vtk):
        vtk.file.write('CELL_DATA {}\n'.format(len(self.arraydata)).encode('ascii'))
        super(PointData, self).tofile(vtk)


class ArrayData(object):
    '''
    Superclass to represent different kinds of data that can be attributed to Points or Cells and
    are given as an iterable of Fields
    '''
    def __init__(self, *fields, **kwargs):
        self.fields = fields
        self.name = kwargs.pop('name', getattr(fields[0], 'name', ''))

    def transform_data(self, dtype):
        data = np.vstack((np.ravel(f, order='F') for f in self.fields))
        data = data.ravel(order='F').astype(dtype)
        return data

    def tofile(self, vtk):
        data = self.transform_data(vtk.dtype)
        try:
            # ndarray.tobytes() was introduced in numpy 1.9
            data = data.tobytes()
        except AttributeError:
            # workaround for numpy <1.9, but only works in python 3
            data = data.data.tobytes()
        vtk.file.write(data)

    def __len__(self):
        return product(self.fields[0].shape)


class Vectors(ArrayData):
    '''
    Class to represent Vectors
    '''
    def __init__(self, *fields, **kwargs):
        if len(fields) != 3:
            raise ValueError('A Vector must have three components')
        super(Vectors, self).__init__(*fields, **kwargs)

    def tofile(self, vtk):
        vtk.file.write('VECTORS {} {}\n'.format(self.name, vtk.type).encode('ascii'))
        super(Vectors, self).tofile(vtk)


class Scalars(ArrayData):
    '''
    Class to represent a collection of Scalars
    '''
    def __init__(self, *fields, **kwargs):
        if len(fields) > 4:
            raise ValueError('Vtk supports up to 4 Scalars in one DataSet')
        super(Scalars, self).__init__(*fields, **kwargs)

    def tofile(self, vtk):
        vtk.file.write('SCALARS {} {} {}\n'.format(self.name, vtk.type, len(self.fields))
                       .encode('ascii'))
        vtk.file.write(b'LOOKUP_TABLE default\n')
        super(Scalars, self).tofile(vtk)


def _export_arraydata_vtk(filename, *fields, **kwargs):
    '''
    Generic method to export multiple `fields` into `filename` as a classic vtk file.
    Keyword arguments may be:

    kwargs['kind']: one of `Vectors` or `Scalars`
    kwargs['name']: Name of the exported dataset
    kwargs['type']: 'float' or 'double'
    kwargs['unstagger']: True if fields should be automatically unstaggered if appropriate
    kwargs['skip_axes_check']: True if the check that fields for having the same axes should be
                               skipped
    '''
    if kwargs.pop('unstagger', True):
        from ..helper import unstagger_fields
        try:
            fields = unstagger_fields(*fields)
        except ValueError as e:
            warnings.warn('Could not unstagger fields, {}'.format(e))
            pass

    if not kwargs.pop('skip_axes_check', False) \
            and not all([field.axes == fields[0].axes for field in fields[1:]]):
        raise ValueError('All fields must have the same axes.')

    if all([ax.islinear() for ax in fields[0].axes]):
        dataset = StructuredPoints.from_field(fields[0])
    else:
        dataset = RectilinearGrid.from_field(fields[0])

    arraydata = kwargs.pop('kind', Vectors)(*fields, name=kwargs.pop('name', None))
    data = PointData(arraydata)
    vtkdata = VtkData(dataset, data)
    vtkdata.tofile(filename, type=kwargs.pop('type', 'float'))


def export_scalar_vtk(filename, scalarfield, **kwargs):
    '''
    exports one 2D or 3D scalar field object to a VTK file
    which is suitable for viewing in ParaView.
    It is assumed that all fields are defined on the same grid.
    '''
    _export_scalars_vtk(filename, scalarfield, **kwargs)


def export_vector_vtk(filename, *fields, **kwargs):
    '''
    exports a vector field to a VTK file suitable for viewing in ParaView.
    Three 3D fields are expected, which will form the X, Y and Z component
    of the vector field. If less than tree fields are given, the missing components
    will be assumed to be zero.
    '''
    from ..datahandling import Field

    fields = [field if isinstance(field, Field) else Field(field) for field in fields]

    if len(fields) > 3:
        raise ValueError("Too many fields")

    while len(fields) < 3:
        fields.append(fields[0].replace_data(np.zeros_like(fields[0])))

    if len(fields[0].shape) > 3:
        raise ValueError("Fields have to many axes")

    fields = [f.atleast_nd(3) for f in fields]

    kwargs['kind'] = Vectors
    _export_arraydata_vtk(filename, *fields, **kwargs)


def _export_scalars_vtk(filename, *fields, **kwargs):
    '''
    exports a set of scalar fields to a VTK file suitable for viewing in ParaView.
    Up to four fields may be given
    '''
    from ..datahandling import Field

    fields = [field if isinstance(field, Field) else Field(field) for field in fields]

    if len(fields) > 4:
        raise ValueError("Too many fields")

    if len(fields[0].shape) > 3:
        raise ValueError("Fields have to many axes")

    fields = [f.atleast_nd(3) for f in fields]

    kwargs['kind'] = Scalars
    _export_arraydata_vtk(filename, *fields, **kwargs)
