
import numpy as np


def export_scalar_vtk(filename, scalarfield):
    '''
    exports one 2D or 3D scalar field object to a VTK file
    which is suitable for viewing in ParaView.
    It is assumed that all fields are defined on the same grid.
    '''
    import pyvtk
    import collections
    from ..datahandling import Field
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
    return np.dstack((np.ravel(f, order='F') for f in fields))


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
