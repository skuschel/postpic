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
# Alexander Blinne, 2017
# Stephan Kuschel, 2017
'''
The postpic.io subpackage provides free functions for importing and exporting data.
'''

import os.path as osp

import numpy as np


def _import_field_image(filename, hotpixelremove=False):
    '''

    '''
    from ..datahandling import Field, Axis
    if filename.lower().endswith('png'):
        data = _readpng(filename)
    else:
        data = _read_image_pil(filename)

    if hotpixelremove:
        import scipy.ndimage
        if data.ndim == 3 and data.shape[2] < 5:
            # assume multichannel image file like rgb or rgba
            for i in range(data.shape[2]):
                data[..., i] = scipy.ndimage.morphology.grey_opening(data[..., i], size=(3, 3))
            else:
                data = scipy.ndimage.morphology.grey_opening(data, size=(3, 3))

    # image data are usually in y-major order, but postpic Fields assume x-major order
    # and rows are stored from top to bottom while y axes coordinate grows from bottom to top
    data = np.moveaxis(data, 0, 1)[:, ::-1, ...]

    axes = []
    for i, (name, axlen) in enumerate(zip(['x', 'y', 'channel'], data.shape)):
        ax = Axis(name=name, unit='px' if i < 2 else '',
                  grid=np.linspace(0, axlen-1, axlen))
        axes.append(ax)

    basename = osp.basename(filename)
    return Field(data, unit='counts', name=basename, axes=axes)


def _readpng(filename):
    '''
    Reads a png file and returns appropriate count vales, even if a bit depth
    other than 8 or 16 is used. An example this might be needed is having a
    12-bit png recorded from a 12-bit camera using LabViews IMAQ toolset.
    In this case the PIL (python image library) fails to retrieve the
    original count values.
    Copied from https://gist.github.com/skuschel/87960f78c7a4a42eb042

    Args:
        filename (str): Path and filename to the file to open

    kwargs:
        hotpixelremove (bool): Removes hotpixels if True. (Default: True)

    Returns:
        numpy.array: the png data as numpy array converted to float64

    Author: Stephan Kuschel, 2015-2016
    '''
    import png  # pypng
    import scipy.misc as sm
    import numpy as np
    meta = png.Reader(filename)
    meta.preamble()
    ret = sm.imread(filename)
    if meta.sbit is not None:
        significant_bits = ord(meta.sbit)
        ret >>= 16 - significant_bits
    # else: 8 bit image, no need to modify data
    return np.float64(ret)


def _read_image_pil(filename):
    '''
    reads an image file using PIL.Image.open.

    Args:
        filename (str): Path and filename to the file to open

    Returns:
        numpy.array: the image data as numpy array converted to float64

    Author: Stephan Kuschel, 2015-2016
    '''
    from PIL import Image
    im = Image.open(filename)
    data = np.array(im, dtype=np.float64)
    return data
