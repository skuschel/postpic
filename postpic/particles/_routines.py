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
# Stephan Kuschel 2017
"""
Particle related functions.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

from . import _particlestogrid as ptg
from ..helper import PhysicalConstants
import re

particleshapes = ptg.shapes

# Default values for histogramdd function
histogramdd_defs = {'shape': 2}

__all__ = ['histogramdd', 'SpeciesIdentifier']


def histogramdd(data, **kwargs):
    '''
    automatically chooses the histogram function to be used.
    `data` must be a tuple. Its length determines the
    dimensions of the histogram returned.
    '''
    [kwargs.setdefault(k, i) for (k, i) in list(histogramdd_defs.items())]
    if isinstance(data, np.ndarray):
        if len(data.shape) == 1:  # [1,2,3]
            h, xedges = ptg.histogram(np.float64(data), **kwargs)
            return h, xedges
    if len(data) == 1:  # ([1,2,3],)
        h, xedges = ptg.histogram(np.float64(data[0]), **kwargs)
        return h, xedges
    if len(data) == 2:  # [[1,2,3], [4,5,6]]
        h, xedges, yedges = ptg.histogram2d(np.float64(data[0]),
                                            np.float64(data[1]), **kwargs)
        return h, xedges, yedges
    if len(data) == 3:  # [[1,2,3], [4,5,6], [7,8,9]]
        h, xe, ye, ze = ptg.histogram3d(np.float64(data[0]),
                                        np.float64(data[1]),
                                        np.float64(data[2]), **kwargs)
        return h, xe, ye, ze
    else:
        raise ValueError('Data with len {:} not supported. Maximum is 3D data.'.format(len(data)))


class SpeciesIdentifier(PhysicalConstants):
    '''
    This Class provides static methods for deriving particle properties
    from species Names. The only reason for this to be a class is that it
    can be used as a mixin.
    '''

    def _specdict(mass, charge, ision):
        return dict(mass=mass * PhysicalConstants.me,
                    charge=charge * PhysicalConstants.qe,
                    ision=ision)

    # if in default, use this
    _defaults = {'electrongold': _specdict(1, -1, False),
                 'proton': _specdict(1836.2 * 1, 1, True),
                 'Proton': _specdict(1836.2 * 1, 1, True),
                 'ionp': _specdict(1836.2, 1, True),
                 'ion': _specdict(1836.2 * 12, 1, True),
                 'c6': _specdict(1836.2 * 12, 1, True),
                 'ionf': _specdict(1836.2 * 19, 1, True),
                 'Palladium': _specdict(1836.2 * 106, 0, True),
                 'Palladium1': _specdict(1836.2 * 106, 1, True),
                 'Palladium2': _specdict(1836.2 * 106, 2, True),
                 'Ion': _specdict(1836.2, 1, True),
                 'Photon': _specdict(0, 0, False),
                 'Positron': _specdict(1, 1, False),
                 'positron': _specdict(1, 1, False),
                 'gold1': _specdict(1836.2 * 197, 1, True),
                 'gold3': _specdict(1836.2 * 197, 3, True),
                 'gold4': _specdict(1836.2 * 197, 4, True),
                 'gold2': _specdict(1836.2 * 197, 2, True),
                 'gold7': _specdict(1836.2 * 197, 7, True),
                 'gold10': _specdict(1836.2 * 197, 10, True),
                 'gold20': _specdict(1836.2 * 197, 20, True)}

    #  unit: amu
    _masslistelement = {'H': 1, 'He': 4,
                        'Li': 6.9, 'C': 12, 'N': 14, 'O': 16, 'F': 19, 'Ne': 20.2,
                        'Na': 23, 'Al': 27, 'Si': 28, 'S': 32, 'Cl': 35.5, 'Ar': 40,
                        'Ti': 47.9, 'Cr': 52, 'Fe': 55.8, 'Cu': 63.5, 'Zn': 65.4, 'Kr': 83.8,
                        'Rb': 85.5, 'Zr': 91.2, 'Pd': 106.4, 'Ag': 107.8, 'Sn': 118.7,
                        'Xe': 131.3,
                        'W': 183.8, 'Pt': 195, 'Au': 197, 'Hg': 200.6, 'Pb': 207.2}

    @staticmethod
    def isejected(species):
        s = species.replace('/', '')
        r = re.match(r'(ejected_)(.*)', s)
        return r is not None

    @classmethod
    def ision(cls, species):
        return cls.identifyspecies(species)['ision']

    @classmethod
    def identifyspecies(cls, species):
        """
        Returns a dictionary containing particle informations deduced from
        the species name.
        The following keys in the dictionary will always be present:
        name   species name string
        mass    kg (SI)
        charge  C (SI)
        tracer  boolean
        ejected boolean

        Valid Examples:
        Periodic Table symbol + charge state: c6, F2, H1, C6b
        ionm#c# defining mass and charge:  ionm12c2, ionc20m110
        advanced examples:
        ejected_tracer_ionc5m20b, ejected_tracer_electronx,
        ejected_c6b, tracer_proton, protonb
        """
        ret = {'tracer': False, 'ejected': False, 'name': species}
        s = species.replace('/', '_')

        if s in cls._defaults:
            # if species name is found in cls._defaults use it and
            # return result.
            ret.update(cls._defaults[s])
            return ret

        # Regex for parsing ion species name.
        # See docsting for valid examples
        regex = '(?P<prae>(.*_)*)' \
                '((?P<elem>((?!El)[A-Z][a-z]?))(?P<elem_c>\d*)|' \
                '(?P<ionmc>(ionc(?P<c1>\d+)m(?P<m2>\d+)|ionm(?P<m1>\d+)c(?P<c2>\d+)))' \
                ')?' \
                '(?P<plus>(Plus)*)' \
                '(?P<electron>[Ee]le[ck])?' \
                '(?P<suffix>\w*?)$'
        r = re.match(regex, s)
        if r is None:
            raise Exception('Species ' + str(s) +
                            ' does not match regex name pattern: ' +
                            str(regex))
        regexdict = r.groupdict()
        # print(regexdict)

        # recognize anz prae and add dictionary key
        if regexdict['prae']:
            for i in regexdict['prae'].split('_'):
                key = i.replace('_', '')
                if not key == '':
                    ret[key] = True

        # Excluding patterns start here
        # 1) Name Element + charge state: C1, C6, F2, F9, Au20, Pb34a
        if regexdict['elem']:
            ret['mass'] = float(cls._masslistelement[regexdict['elem']]) * \
                1836.2 * cls.me
            if regexdict['elem_c'] == '':
                ret['charge'] = 0
            else:
                ret['charge'] = float(regexdict['elem_c']) * cls.qe
            ret['ision'] = True
        # 2) ionmc like
        elif regexdict['ionmc']:
            if regexdict['c1']:
                ret['mass'] = float(regexdict['m2']) * 1836.2 * cls.me
                ret['charge'] = float(regexdict['c1']) * cls.qe
                ret['ision'] = True

            if regexdict['c2']:
                ret['mass'] = float(regexdict['m1']) * 1836.2 * cls.me
                ret['charge'] = float(regexdict['c2']) * cls.qe
                ret['ision'] = True

        # charge may be given via plus
        if regexdict['plus']:
            charge = len(regexdict['plus'])/4 * cls.qe
            if ret['charge'] == 0:
                ret['charge'] = charge

        # Elektron can be appended to any ion name, so overwrite.
        if regexdict['electron']:
            ret['mass'] = cls.me
            ret['charge'] = -1 * cls.qe
            ret['ision'] = False

        if not (('mass' in ret) and ('charge' in ret) and ('ision' in ret)):
            raise Exception('species ' + species + ' not recognized.')

        return ret
