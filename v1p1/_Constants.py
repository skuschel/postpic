"""
Konstanten und statische Methoden
"""

import numpy as np


__all__=['_Constants']

class _Constants:
    """
    Stellt Konstatanten fuer dieses Modul bereit.
    """
    _c = 299792458.0
    _me = 9.109383e-31
    _qe = 1.602176565e-19
    _mu0 = np.pi*4e-7 #N/A^2
    _epsilon0 = 1 / (_mu0 * _c**2) #8.85419e-12 As/Vm

    _axisoptions = {'X':slice(0,2), 'Y':slice(2,4), 'Z':slice(4,6), None:slice(None),
            'x':slice(0,2), 'y':slice(2,4), 'z':slice(4,6)}    
    _axisoptionseinzel = {'X':slice(0,1), 'Y':slice(1,2), 'Z':slice(2,3), None:slice(None), 'x':slice(0,1), 'y':slice(1,2), 'z':slice(2,3)} 
    _axesidentify = {'X':0, 'x':0, 0:0,'Y':1, 'y':1, 1:1,'Z':2, 'z':2, 2:2}   
    _poptsidentify = _axesidentify.copy()
    _poptsidentify.update({'px':3,'Px':3,'py':4,'Py':4,'pz':5,'Pz':5,'weight':9,'w':9})

    @staticmethod
    def ncrit_um(lambda_um):
        return 1.11e27*1/(lambda_um**2) #1/m^3

    @staticmethod
    def ncrit(laslambda):
        return _Constants.ncrit_um(laslambda * 1e6) #1/m^3
        
    @staticmethod
    def datenausschnitt(m, oldextent, newextent):
        """
        Schneidet aus einer Datenmatrix m den Teil heraus, der newextent entspricht, wenn die gesamte Matrix zuvor oldextent entsprochen hat. Hat m insgeamt dims dimensionen, dann muessen oldextent und newextent die Laenge 2*dims haben.
        Einschraenkung: newextent muss innerhalb von oldextent liegen
        
        """
        dims = len(m.shape)
        assert not oldextent is newextent, 'oldextent und newextent zeigen auf dasselbe Objekt(!). Da hat wohl jemand beim Programmieren geschlafen :)'
        assert len(oldextent) / 2 == dims, 'Dimensionen von m und oldextent falsch!'
        assert len(newextent) / 2 == dims, 'Dimensionen von m und newextent falsch!'
        s=()
        for dim in range(dims):
            i=2*dim
            thisdimmin = round((newextent[i]-oldextent[i])/(oldextent[i + 1]-oldextent[i])*m.shape[dim])
            thisdimmax = round((newextent[i + 1]-oldextent[i])/(oldextent[i + 1]-oldextent[i])*m.shape[dim])
            s = np.append(s, slice(thisdimmin, thisdimmax))
        if len(s) == 1:
            s=s[0]
        else:
            s=tuple(s)
        return m[s]

    # Transformation: polarkoordinaten --> kartesische Koordinaten,
    # (i,j), Indizes des Bildes in Polardarstellung --> Indizes des Bildes in kartesischer Darstellung
    # damit das Bild transformiert wird: kartesischen Koordinaten --> polarkoordinaten
    @staticmethod
    def transfromxy2polar(matrixxy, extentxy, extentpolar, shapepolar, ashistogram=True):
        """transformiert eine Matrix in kartesischer Darstellung (matrixxy, Achsen x,y) in eine Matrix in polardarstellung (Achsen r,phi)."""
        from scipy import ndimage
        def polar2xy((r, phi)):
            x = r * np.cos(phi)
            y = r * np.sin(phi)
            return (x,y)
        def koord2index((q1,q2), extent, shape):
            return ( (q1-extent[0])/(extent[1]-extent[0])*shape[0], (q2-extent[2])/(extent[3]-extent[2])*shape[1] )
        def index2koord((i,j), extent, shape):
            return ( extent[0] + i/shape[0] * (extent[1]-extent[0]), extent[2] + j/shape[1] * (extent[3]-extent[2]) )
        def mappingxy2polar((i,j), extentxy, shapexy, extentpolar, shapepolar):
            """actually maps indizes of polar matrix to indices of kartesian matrix"""
            ret = polar2xy(index2koord((float(i), float(j)), extentpolar, shapepolar))
            ret = koord2index(ret, extentxy, shapexy)
            return ret
        ret = ndimage.interpolation.geometric_transform(matrixxy, mappingxy2polar, output_shape=shapepolar, extra_arguments=(extentxy, matrixxy.shape, extentpolar, shapepolar), order=1)
        if ashistogram: #Volumenelement ist einfach nur r
            ret = (ret.T * np.abs(np.linspace(extentpolar[0], extentpolar[1], ret.shape[0]))).T
        return ret
        
        
        
        
