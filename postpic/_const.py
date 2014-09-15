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

"""
Some global constants that are used in the code.
"""


axesidentify = {'X': 0, 'x': 0, 0: 0,
                'Y': 1, 'y': 1, 1: 1,
                'Z': 2, 'z': 2, 2: 2}
attribidentify = axesidentify.copy()
attribidentify.update({'PX': 3, 'Px': 3, 'px': 3, 3: 3,
                       'PY': 4, 'Py': 4, 'py': 4, 4: 4,
                       'PZ': 5, 'Pz': 5, 'pz': 5, 9: 9,
                       'weight': 9, 'w': 9, 10: 10,
                       'id': 10, 'ID': 10})

# Some static functions


def cutout(m, oldextent, newextent):
    """
    cuts out a part of the matrix m that belongs to newextent if the full
    matrix corresponds to oldextent. If m has dims dimensions, then oldextent
    and newextent have to have a length of 2*dims each.
    nexextent has to be inside of oldextent!
    (this should be fixed in the future...)
    """
    import numpy as np
    dims = len(m.shape)
    assert oldextent is not newextent, 'oldextent and newextent point to the' \
                                       'same objekt(!). Get a coffe and' \
                                       'check your code again. :)'
    assert len(oldextent) / 2 == dims, \
        'dimensions of oldextent and m are wrong!'
    assert len(newextent) / 2 == dims, \
        'dimensions of newextent and m are wrong!'
    s = ()
    for dim in range(dims):
        i = 2 * dim
        thisdimmin = round((newextent[i] - oldextent[i])
                           / (oldextent[i + 1] - oldextent[i]) * m.shape[dim])
        thisdimmax = round((newextent[i + 1] - oldextent[i])
                           / (oldextent[i + 1] - oldextent[i]) * m.shape[dim])
        s = np.append(s, slice(thisdimmin, thisdimmax))
    if len(s) == 1:
        s = s[0]
    else:
        s = tuple(s)
    return m[s]


def transfromxy2polar(matrixxy, extentxy,
                      extentpolar, shapepolar, ashistogram=True):
    '''
    remaps a matrix matrixxy in kartesian coordinates x,y to a polar
    representation with axes r, phi.
    '''
    from scipy.ndimage.interpolation import geometric_transform
    import numpy as np

    def polar2xy((r, phi)):
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        return (x, y)

    def koord2index((q1, q2), extent, shape):
        return ((q1 - extent[0]) / (extent[1] - extent[0]) * shape[0],
                (q2 - extent[2]) / (extent[3] - extent[2]) * shape[1])

    def index2koord((i, j), extent, shape):
        return (extent[0] + i / shape[0] * (extent[1] - extent[0]),
                extent[2] + j / shape[1] * (extent[3] - extent[2]))

    def mappingxy2polar((i, j), extentxy, shapexy, extentpolar, shapepolar):
        '''
        actually maps indizes of polar matrix to indices of kartesian matrix
        '''
        ret = polar2xy(index2koord((float(i), float(j)),
                       extentpolar, shapepolar))
        ret = koord2index(ret, extentxy, shapexy)
        return ret

    ret = geometric_transform(matrixxy, mappingxy2polar,
                              output_shape=shapepolar,
                              extra_arguments=(extentxy, matrixxy.shape,
                                               extentpolar, shapepolar),
                              order=1)
    if ashistogram:  # volumeelement is just r
        r = np.abs(np.linspace(extentpolar[0], extentpolar[1], ret.shape[0]))
        ret = (ret.T * r).T
    return ret





