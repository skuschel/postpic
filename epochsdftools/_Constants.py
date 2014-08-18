"""
Konstanten und statische Methoden
"""

import numpy as np


__all__ = ['_Constants']

class _Constants:
    """
    Stellt Konstatanten fuer dieses Modul bereit.
    """
    _c = 299792458.0
    _me = 9.109383e-31
    _qe = 1.602176565e-19
    _mu0 = np.pi * 4e-7  # N/A^2
    _epsilon0 = 1 / (_mu0 * _c ** 2)  # 8.85419e-12 As/Vm

    _axisoptions = {'X':slice(0, 2), 'Y':slice(2, 4), 'Z':slice(4, 6), None:slice(None),
            'x':slice(0, 2), 'y':slice(2, 4), 'z':slice(4, 6)}
    _axisoptionseinzel = {'X':slice(0, 1), 'Y':slice(1, 2), 'Z':slice(2, 3), None:slice(None), 'x':slice(0, 1), 'y':slice(1, 2), 'z':slice(2, 3)}
    _axesidentify = {'X':0, 'x':0, 0:0, 'Y':1, 'y':1, 1:1, 'Z':2, 'z':2, 2:2}
    _poptsidentify = _axesidentify.copy()
    _poptsidentify.update({'px':3, 'Px':3, 'py':4, 'Py':4, 'pz':5, 'Pz':5, 'weight':9, 'w':9, 'id':10, 'ID':10})

    @staticmethod
    def ncrit_um(lambda_um):
        return 1.11e27 * 1 / (lambda_um ** 2)  # 1/m^3

    @staticmethod
    def ncrit(laslambda):
        return _Constants.ncrit_um(laslambda * 1e6)  # 1/m^3

    @staticmethod


    # Transformation: polarkoordinaten --> kartesische Koordinaten,
    # (i,j), Indizes des Bildes in Polardarstellung --> Indizes des Bildes in kartesischer Darstellung
    # damit das Bild transformiert wird: kartesischen Koordinaten --> polarkoordinaten
    @staticmethod



    @staticmethod
    def _constructsavename(name=None):
        import os
        fname = 'tmp'
        if name:
            fname = name
        f = os.path.join(os.getcwd(), 'datasave', fname)
        if not os.path.exists(os.path.dirname(f)):
            os.makedirs(os.path.dirname(f))
        return f

    @staticmethod
    def saveidlist(idlist, name=None):
        '''
        saves an list of particle IDs inside the sdfdir.
        name can be given in addition to change the output filename
        '''
        import pickle
        with open(_Constants._constructsavename(name), 'w') as f:
            pickle.dump(idlist, f)

    @staticmethod
    def loadidlist(name=None):
        '''
        saves an list of particle IDs inside the sdfdir.
        name can be given in addition to change the output filename
        '''
        import pickle
        with open(_Constants._constructsavename(name), 'r') as f:
            ret = pickle.load(f)
        return ret
