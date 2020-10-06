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
# Stephan Kuschel 2019
# Alexander Blinne, 2017-2019
"""
Helper for fft functions.

If available, pyffw will be used to speed up calculations efficiently.
"""
import sys

if sys.version[0] == '2':
    import functools32 as functools
else:
    import functools


__all__ = ['fft']


class _fft:

    try:
        import psutil
        nproc = psutil.cpu_count(logical=False)
    except ImportError:
        try:
            import os
            nproc = os.cpu_count()
        except AttributeError:
            import multiprocessing
            nproc = multiprocessing.cpu_count()

    try:
        # pyfftw is, in most situations, faster than numpys fft,
        # although pyfftw will benefit from multithreading only on very large arrays
        # on a 720x240x240 3D transform multithreading still doesn't give a large benefit
        # benchmarks of a 720x240x240 transform of real data on a Intel(R) Xeon(R) CPU
        # E5-1620 v4 @ 3.50GHz:
        # numpy.fft: 3.6 seconds
        # pyfftw, nproc=4: first transform 2.2s, further transforms 1.8s
        # pyfftw, nproc=1: first transform 3.4s, further transforms 2.8s
        # Try to import pyFFTW's numpy_fft interface
        import pyfftw.interfaces.cache as fftw_cache
        import pyfftw.interfaces.numpy_fft as fftw
        fftw_cache.enable()
        fft_module = fftw
        # workaround for
        # https://github.com/pyFFTW/pyFFTW/issues/135
        # also, scaling is bad, so 1 process wont hurt too much
        nproc = 1
        fft_kwargs = dict(planner_effort='FFTW_ESTIMATE', threads=nproc)
    except ImportError:
        # pyFFTW is not available, just import numpys fft
        import numpy.fft as fft_module
        fft_kwargs = dict()

    _fft_functions = ['fft', 'ifft', 'fft2', 'ifft2', 'fftn',
                      'ifftn', 'rfft', 'irfft', 'rfft2', 'irfft2', 'irfftn',
                      'hfft', 'ihfft']

    @classmethod
    def _get_defaultkwargf(cls, name):
        wrapped = getattr(cls.fft_module, name)

        @functools.wraps(wrapped)
        def ret(*args, **kwargs):
            kws = cls.fft_kwargs.copy()
            kws.update(kwargs)
            return wrapped(*args, **kws)
        return ret

    def __getattr__(self, attr):
        return getattr(self.fft_module, attr)

    def __init__(self):
        for fftf in self._fft_functions:
            setattr(self, fftf, self._get_defaultkwargf(fftf))


fft = _fft()
