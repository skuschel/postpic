
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
