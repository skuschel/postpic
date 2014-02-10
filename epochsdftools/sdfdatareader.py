"""
Stellt Klassen und Methoden zum abstracten Datenzugriff mit einem Dump oder mit einer Liste von Dumps zur verfuegung. Jeglicher Datenzugriff aus anderen Programmteilen muss ueber Klassen dieses Moduls erfolgen.
"""


import re
import os
import sys
import numpy as np
from _Interfaces import PlotDescriptor
from _Constants import _Constants
from . import Feld, ParticleAnalyzer, FieldAnalyzer


__all__ = ['OutputAnalyzer', 'SDFAnalyzer']

class OutputAnalyzer(PlotDescriptor):
    """
    Sammelt Informationen ueber eine gesamte output Serie, definiert durch eine .visit Datei.
    """

    def __init__(self, visitfile, lasnm=None):
        self.visitfile = visitfile
        import os.path
        if not os.path.isfile(visitfile):
            raise IOError('File ' + str(visitfile) + ' doesnt exist.')
        self.lasnm = lasnm
        self.projektname = os.path.basename(os.getcwd())
        if lasnm:
            print visitfile + ": lambda_0 (nm) = " + str(lasnm)
        else:
            print "WARNING: Laserwellenlaenge nicht gegeben. Einige Plots stehen nicht zur Verfuegung."
        self.sdffiles = []
        with open(visitfile) as f:
            relpath = os.path.dirname(visitfile)
            for line in f:
                self.sdffiles.append(os.path.join(relpath, line.replace('\n', '')))

    def __str__(self):
        return '<OutputAnalyzer at ' + self.visitfile + ' using lambda0=' + str(self.lasnm) + 'nm>'

    def __len__(self):
        """
        Anzahl der beinhaltenden Dumps.
        """
        return len(self.sdffiles)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return [self[i] for i in xrange(*key.indices(len(self)))]
        elif isinstance(key, int):
            return SDFAnalyzer(self.sdffiles[key], lasnm=self.lasnm, printinfo=False)
        else:
            raise TypeError

    # PlotDescriptor Interface
    def getprojektname(self):
        return self.projektname
    def getprojektname2(self):
        return os.path.basename(self.visitfile)
    def getprojektname3(self):
        return ''
    def getprojektname4(self):
        return ''
    def getlasnm(self):
        return self.lasnm

    # High Level
    def species(self, ejected='ignore'):
        """
        Gibt eine Liste aller an der Simulation beteiligten Species aus
        ejected='ignore' gibt an, wie ejected particles behandelt werden (falls vorhanden). Optionen sind
        'all' - normale und ejected particles werden ausgegeben
        'only' - nur ejected particles werden ausgegeben
        'ignore' - ejected particles werden nicht mit ausgegeben.
        """
        ret = set()
        for dump in self:
            ret |= set(dump.species(ejected=ejected))
        ret = list(ret)
        ret.sort()
        return ret

    def getparticleanalyzercollect(self, species, ndumps=sys.maxint):
        """
        Gibt einen einzelnen Particle Analzer zurueck der alle Teilchen des letzten Dumps und alle ejected Particles der vorhergehenden Dumps einer einzelnen Spezies beinhaltet.
        ndumps=n verwendet nur die ersten n+1 dumps aus dieser visit file, so alsob die Simulation danach abgebrochen worden waere. 
        Achtung: Bei ndumps=15 ist mit fortlaufender nummerierung (0000.sdf, 0001.sdf,..., 0014.sdf) also <ndumps-1>.sdf die letzte sdf-datei.
        """
        sl = slice(0, ndumps + 1)
        sdfas = self[sl]
        pa = sdfas[-1].getparticleanalyzer(species)
        nparticles = len(pa)
        species = 'ejected_' + species.replace('/', '')
        for sdfa in sdfas:
            pa += sdfa.getparticleanalyzer(species)
        if len(pa) == nparticles:
            print('WARNING: No ejected Particles collected. Did you forget to dump ejected particles?')
        return pa

    def maptoSDFAnalyzer(self, f, *args, **kwargs):
        return [getattr(s, f)(*args, **kwargs) for s in self]

    def getparticleanalyzerlist(self, *speciess):
        return [s.getparticleanalyzer(*speciess) for s in self]

    def getfieldanalyzerlist(self):
        return [s.getfieldanalyzer() for s in self]

    def times(self, unit=''):
        # Alle folgenden Zeilen sind aequivalent
        # return self.maptoSDFAnalyzer('time')
        return [s.time(unit) for s in self]
        # return map(SDFAnalyzer.time, self)

    def _createtimeseries_simple(self, f, unit=''):
        """
        einfach implementiert.
        Parallele Alternative schreiben!
        """
        try:
            # try to override 1D Histogram bin default to improve quality
            ret = Feld.factorystack(*tuple([f(sdfa, optargsh={'bins':800}) for sdfa in self]))
        except TypeError:
            ret = Feld.factorystack(*tuple([f(sdfa) for sdfa in self]))
        ret.addaxis('Time', unit)
        ret.setgrid_node_fromgrid(-1, self.times(unit))
        ret.zusatz = ''
        return ret

    def hasID(self):
        return self[0].hasID()

    def createtimeseries(self, f):
        """
        Erzeugt eine Zeitserie.
        f ist eine Funktion, die einen SDFAnalyzer als Argument nimmt und ein Feld (0D oder 1D) als Rueckgabewert hat. Diese Felder werden dann zu mehreren Felder kombiniert, mit der Zeit (genauer: den Dumps) auf einer weiteren Achse.
        If f is created by any histogram, createtimeseries improves the quality by trying to override the 'bins' setting. In this case make sure to forward **kwargs through f.
        """
        return self._createtimeseries_simple(f)







class SDFAnalyzer(PlotDescriptor, _Constants):
    """
    Liest die sdfdatei ein und gibt und berechnet grundlegende Informationen.
    """

    def __init__(self, dateiname, lasnm=None, printinfo=True):
        if printinfo:
            print("-------------- " + dateiname + " --------------")
        import sdf
        self.dateiname = dateiname
        import os.path
        if not os.path.isfile(dateiname):
            raise IOError('File ' + str(dateiname) + ' doesnt exist.')
        self._data = sdf.SDF(dateiname).read()
        self.dumpname = os.path.basename(dateiname).replace('.sdf', '')
        self.projektname = os.path.basename(os.getcwd())
        self.header = self._data['Header']
        self.simdimensions = float(re.match('Epoch(\d)d', self.header['code_name']).group(1))
        self.lasnm = lasnm
        # Simextent bestimmen
        self._simextent = np.real([self._data['Grid/Grid_node/X'][0], self._data['Grid/Grid_node/X'][-1]])
        self.simgridpoints = [np.real(self._data['Grid/Grid_node/X']).shape[0] - 1]
        if self.simdimensions > 1:
            self.simgridpoints = np.append(self.simgridpoints, np.real(self._data['Grid/Grid_node/Y']).shape[0] - 1)
            self._simextent = np.append(self._simextent, np.float64([self._data['Grid/Grid_node/Y'][0], self._data['Grid/Grid_node/Y'][-1]]))
        if self.simdimensions > 2:
            self.simgridpoints = np.append(self.simgridpoints, np.real(self._data['Grid/Grid_node/Z']).shape[0] - 1)
            self._simextent = np.append(self._simextent, np.float64([self._data['Grid/Grid_node/Z'][0], self._data['Grid/Grid_node/Z'][-1]]))
        if lasnm:
            if printinfo:
                print "lambda_0 (nm) = " + str(lasnm)
                print self.species(ejected='all')
        else:
            print "WARNING: Laserwellenlaenge nicht gegeben. Einige Plots stehen nicht zur Verfuegung."



    def __str__(self):
        return '<SDFAnalyzer at ' + self.dateiname + ' using lambda0=' + str(self.lasnm) + 'nm>'

    # Plot Descriptor Funktionen
    def getprojektname(self):
        return self.projektname
    def getprojektname2(self):
        """
        genauere Beschreibung des Projekts, z.B. dumpname
        """
        return self.dumpname
    def getprojektname3(self):
        """
        noch genauere Beschreibung des Projekts, z.B. Zeitschritt
        """
        return self.time()
    def getprojektname4(self):
        return self.header['step']
    def getlasnm(self):
        return self.lasnm


    def time(self, unit=''):
        t = self.header['time']
        if unit == '':
            return t
        elif unit == 'f' or unit == 'fs':
            return t * 10 ** 15
        else:
            raise Exception('Unit ' + str(unit) + ' unbekannt')


    # High Level
    def species(self, ejected='ignore'):
        """
        Gibt eine Liste aller an der Simulation beteiligten Species aus
        ejected='ignore' gibt an, wie ejected particles behandelt werden (falls vorhanden). Optionen sind
        'all' - normale und ejected particles werden ausgegeben
        'only' - nur ejected particles werden ausgegeben
        'ignore' - ejected particles werden nicht mit ausgegeben.
        """
        ret = []
        for key in self._data.keys():
            match = re.match('Particles/Px/(\w+)', key)
            if match:
                if ejected == 'all' or (ParticleAnalyzer.isejected(match.group(1)) == (ejected == 'only')):
                    ret = np.append(ret, match.group(1))
        ret.sort()
        return ret

    def hasSpecies(self, species):
        return species in self.species(ejected='all')

    def ions(self, ejected='ignore'):
        """
        Gibt eine Liste aller an der Simulation beteiligten Ionen aus
        ejected='ignore' gibt an, wie ejected particles behandelt werden (falls vorhanden). Optionen sind
        'all' - normale und ejected particles werden einzeln ausgegeben
        'only' - nur ejected particles werden ausgegeben
        'ignore' - ejected particles werden nicht mit ausgegeben.
        """
        ret = []
        for species in self.species(ejected=ejected):
            if ParticleAnalyzer.ision(species):
                ret.append(species)
        return ret

    def nonions(self, ejected='ignore'):
        """
        Gibt eine Liste aller an der Simulation beteiligten nicht-Ionen aus
        ejected='ignore' gibt an, wie ejected particles behandelt werden (falls vorhanden). Optionen sind
        'all' - vorhandene und ejected particles werden ausgegeben
        'only' - nur ejected particles werden ausgegeben
        'ignore' - ejected particles werden nicht mit ausgegeben.
        """
        ret = []
        for species in self.species(ejected=ejected):
            if not ParticleAnalyzer.ision(species):
                ret.append(species)
        return ret

    def getfieldanalyzer(self):
        return FieldAnalyzer(self, lasnm=self.lasnm)

    def getparticleanalyzer(self, species, ejected='ignore'):
        """
        Gibt einen Particle Analyzer einer einzelnen Spezies oder aller ionen/nicht-ionen aus.
        ejected='ignore' gibt an, wie ejected particle species behandelt werden (falls vorhanden). Optionen sind
        'all' - vorhandene und ejected particles werden einzeln ausgegeben / bei anforderung einer einzelnen spezies werden diese ebenfalls hinzugefuegt
        'only' - nur ejected particles werden ausgegeben
        'ignore' - ejected particles werden nicht mit ausgegeben.
        """
        if species == 'ions':
            return ParticleAnalyzer(self, *self.ions(ejected=ejected))
        elif species == 'nonions':
            return ParticleAnalyzer(self, *self.nonions(ejected=ejected))
        else:
            return ParticleAnalyzer(self, species)

    # low-level
    def getderived(self):
        """Gibt alle Keys zurueck die mit "Derived/" beginnen. Diese sollten direkt an data(key) weitergegeben werden koennen, um die Daten zu erhalten."""
        ret = []
        for key in self._data.keys():
            r = re.match('Derived/[\w/ ]*', key)
            if r:
                ret.append(r.group(0))
        ret.sort()
        return ret

    def data(self, key):
        """Gibt die Daten zu diesem Key zurueck (in float64)"""
        assert self._data.has_key(key), 'Den Key ' + key + ' gibts nicht!'
        return np.float64(self._data[key])

    def _returnkey2(self, key1, key2, average=False):
        key = key1 + key2
        if average:
            key = key1 + '_average' + key2
        return self.data(key)

    def dataE(self, axis, **kwargs):
        axsuffix = {0:'x', 1:'y', 2:'z'}[_Constants._axesidentify[axis]]
        return self._returnkey2('Electric Field', '/E' + axsuffix, **kwargs)

    def dataB(self, axis, **kwargs):
        axsuffix = {0:'x', 1:'y', 2:'z'}[_Constants._axesidentify[axis]]
        return self._returnkey2('Magnetic Field', '/B' + axsuffix, **kwargs)


    def simextent(self):
        return self._simextent

    def grid(self, axis):
        """axis wird erkannt sofern es in _Constants._axesidentify ist"""
        axsuffix = {0:'X', 1:'Y', 2:'Z'}[_Constants._axesidentify[axis]]
        return self._data['Grid/Grid/' + axsuffix]

    def grid_node(self, axis):
        """axis wird erkannt sofern es in _Constants._axesidentify ist"""
        axsuffix = {0:'X', 1:'Y', 2:'Z'}[_Constants._axesidentify[axis]]
        return self._data['Grid/Grid_node/' + axsuffix]

    def getSpecies(self, spezies, attrib):
        """
        Gibt das Attribut (x,y,z,px,py,pz,weight,ID) dieser Teilchenspezies zurueck.
        returning None means that this particle property wasnt dumped. Note that this is different from returning an empty list.
        """
        attribid = _Constants._poptsidentify[attrib]
        options = {9:lambda s: 'Particles/Weight/' + s,
            0:lambda s: 'Grid/Particles/' + s + '/X',
            1:lambda s: 'Grid/Particles/' + s + '/Y',
            2:lambda s: 'Grid/Particles/' + s + '/Z',
            3:lambda s: 'Particles/Px/' + s,
            4:lambda s: 'Particles/Py/' + s,
            5:lambda s: 'Particles/Pz/' + s,
            10:lambda s:'Particles/ID/' + s}
        try:
            ret = self._data[options[attribid](spezies)]
        except(KeyError):
            ret = None
        return ret

    def hasID(self):
        return self.getSpecies(self.species()[0], 'ID') is not None


