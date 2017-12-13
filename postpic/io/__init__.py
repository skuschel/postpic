
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
