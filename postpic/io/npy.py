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

from .common import _header_string


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
    from ..datahandling import Field, Axis
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
                         axes_transform_state=meta_ax_transform_state.tolist(),
                         transformed_axes_origins=meta_ax_transformed_origins.tolist())
    import_field.label = meta_field[2]
    import_field.infostring = meta_field[3]

    return import_field
