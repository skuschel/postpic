"""
+-------------------+
|   EPOCHSDFTOOLS   |
+-------------------+

Stephan Kuschel, 20130525
last update: 130911


UPDATE HISTORY:
---------------------
131108: Bumped to v1.1.0: Plot Descriptor hinzugefuegt, inkompatibel mit vorhergehenden Versionen, weil SDFPlots.sdfanalyzer nicht mehr existiert.
131107: OutputAnalyzer hinzugefuegt
131106: lineoutx und lineouty als Option zum 2D Feld Plot hinzugefuegt. Ekennung fuer "ejected_" Particles hinzugefuegt. 
130911: createHistogram2D rangex und rangy hinzugefuegt.
130813: SDFPlots.funcformatter geupdated.
130726: CSV export fuer 1d Felder hinzugefuegt. Kann direkt beim plot mit savecsv=True aktiviert werden. Volumenelement bei angleoffaxis entfernt.
130712: ein haufen Bugfixes.
130702: ParticleAnalyzer benutzt jetzt intern den SingleSpeciesAnalyzer. ParticleAnalyzer Objekte koennen nun addiert werden um unterschiedliche Sets von Teilchen zu erstellen. SingleSpeciesAnalyzer erkennt nun Masse und Ladung der Teilchen, SingleSpeciesAnalyzer.getmass wurde ersetzt durch SingleSpeciesAnalyzer.retrieveparticleinfo. ParticleAnalyzer nimmt nun auch mehrere Spezies direkt entgegen. Elektronen werden erkannt und der ParticleAnalyzer weiss, ob ein Teilchen ein ion ist oder nicht.
130701: Viele tools zum Plotten und erstellen der Histogramme hinzugefuegt. Feld-Klasse hinzugefuegt um Datenformat zu vereinheitlichen.
130628: numpy als np importiert. sdfanalyzer hinzugefuegt.
130619: FieldAnalyzer begonnen
130618: Konversion zu float64 ergaenzt. Verhindert Rechenfehler, falls Daten nur mit single precision geschrieben wurden.
130610: erkennt automatisch die Anzahl der Dimensionen in der Simulation, sowie ionenspezies vom typ ionc22m70
"""

__version__='1.1.0'

import numpy as np
import os
import re
import copy

import matplotlib; matplotlib.use('Agg') #Bilder auch ohne X11 rendern
import matplotlib.pyplot as plt
#from matplotlib.ticker import FuncFormatter
from matplotlib.colors import LinearSegmentedColormap




class PlotDescriptor():
    """
    Stellt Allgemeine Informationen zum Beschriften der Plots bereit. Wird von Allen Analyzer Klassen Implementiert.
    """

    def __init__(self):
        pass
    
    def getprojektname(self):
        raise NotImplementedError
        
    def getprojektname2(self):
        """
        genauere Beschreibung des Projekts, z.B. dumpname
        """
        raise NotImplementedError
        
    def getprojektname3(self):
        """
        noch genauere Beschreibung des Projekts, z.B. Zeit
        """
        raise NotImplementedError
    
    def getprojektname4(self):
        """
        z.B. Zeitschritt
        """
        raise NotImplementedError

    def getlasnm(self):
        raise NotImplementedError





class OutputAnalyzer(PlotDescriptor):
    """
    Sammelt Informationen ueber eine gesamte output Serie, definiert durch eine .visit Datei.
    """

    def __init__(self, visitfile, lasnm=None):
        self.visitfile = visitfile
        self.lasnm = lasnm
        self.projektname = os.path.basename(os.getcwd())
        if lasnm:
            print visitfile + ": lambda_0 (nm) = " + str(lasnm)
        else:
            print "WARNING: Laserwellenlaenge nicht gegeben. Einige Plots stehen nicht zur Verfuegung."
        self.sdffiles = []
        with open(visitfile) as f:
            relpath = os.path.dirname(visitfile)
            if relpath == '':
                relpath = '.'
            relpath = relpath + '/'
            for line in f:
                self.sdffiles.append(relpath + line.replace('\n',''))
    
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
            
    #PlotDescriptor Interface
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
        

    def getparticleanalyzercollect(self, species):
        """
        Gibt einen einzelnen Particle Analzer zurueck der alle Teilchen des letzten Dumps und alle ejected Particles der vorhergehenden Dumps einer einzelnen Spezies beinhaltet.
        """
        pa = self[-1].getparticleanalyzer(species)
        species = '/ejected_' + species.replace('/','')
        for sdfa in self:
            pa += sdfa.getparticleanalyzer(species)
        return pa
        
    def maptoSDFAnalyzer(self, f, *args, **kwargs):
        return [getattr(s, f)(*args, **kwargs) for s in self]
               
    def getparticleanalyzerlist(self, *speciess):
        return [s.getparticleanalyzer(*speciess) for s in self]

    def getfieldanalyzerlist(self):
        return [s.getfieldanalyzer() for s in self]
        
    def times(self, unit=''):
        #Alle folgenden Zeilen sind aequivalent
        #return self.maptoSDFAnalyzer('time')  
        return [s.time(unit) for s in self]    
        #return map(SDFAnalyzer.time, self)
        
    def _createtimeseries_simple(self, f, unit=''):
        """
        einfach implementiert, aber richtig.
        Parallele Alternative schreiben!
        """
        retm = []
        for sdfa in self:  ### Parallelisieren!
            feld = f(sdfa)
            retm.append(feld.matrix)
        feld.matrix = np.array(retm)
        feld.addaxis('Time', unit)
        feld.setgrid_node_fromgrid(-1, self.times(unit))
        #feld.extent.tolist().append([self[0].time('fs'), self[-1].time('fs')])
        return feld
        
    def createtimeseries(self, f):
        """
        Erzeugt eine Zeitserie.
        f ist eine Funktion, die einen SDFAnalyzer als Argument nimmt und ein Feld (0D oder 1D) als Rueckgabewert hat. Diese Felder werden dann zu mehreren Felder kombiniert, mit der Zeit (genauer: den Dumps) auf einer weiteren Achse.
        """
        return self._createtimeseries_simple(f)


class SDFAnalyzer(PlotDescriptor):
    """
    Liest die sdfdatei ein und gibt und berechnet grundlegende Informationen.
    """

    def __init__(self, dateiname, lasnm=None, printinfo=True):
        if printinfo:
            print("-------------- " + dateiname + " --------------")
        import sdf
        self.dateiname = dateiname
        self.data = sdf.SDF(dateiname).read()
        self.dumpname = os.path.basename(dateiname).replace('.sdf','')
        self.projektname = os.path.basename(os.getcwd())
        self.header = self.data['Header']
        self.simdimensions = float(re.match('Epoch(\d)d', self.header['code_name']).group(1))
        self.lasnm = lasnm
        if lasnm:
            if printinfo:
                print "lambda_0 (nm) = " + str(lasnm)
                print self.species(ejected='all')
        else:
            print "WARNING: Laserwellenlaenge nicht gegeben. Einige Plots stehen nicht zur Verfuegung."

        
    def __str__(self):
        return '<SDFAnalyzer at ' + self.dateiname + ' using lambda0=' + str(self.lasnm) + 'nm>' 

    #Plot Descriptor Funktionen    
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
        if unit=='':
            return t
        elif unit=='f' or unit=='fs':
            return t * 10**15   
        else:
            raise Exception('Unit ' + str(unit) + ' unbekannt')
            
            

    def species(self, ejected='ignore'):
        """
        Gibt eine Liste aller an der Simulation beteiligten Species aus
        ejected='ignore' gibt an, wie ejected particles behandelt werden (falls vorhanden). Optionen sind
        'all' - normale und ejected particles werden ausgegeben
        'only' - nur ejected particles werden ausgegeben
        'ignore' - ejected particles werden nicht mit ausgegeben.
        """
        ret = []
        for key in self.data.keys():
            match = re.match('Particles/Px(/\w+)', key)
            if match:
                if ejected=='all' or (ParticleAnalyzer.isejected(match.group(1)) == (ejected=='only')):
                    ret = np.append(ret, match.group(1))
        ret.sort()
        return ret
     
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
        return FieldAnalyzer(self.data, lasnm=self.lasnm)

    def getparticleanalyzer(self, species, ejected='ignore'):
        """
        Gibt einen Particle Analyzer einer einzelnen Spezies oder aller ionen/nicht-ionen aus.
        ejected='ignore' gibt an, wie ejected particle species behandelt werden (falls vorhanden). Optionen sind
        'all' - vorhandene und ejected particles werden einzeln ausgegeben / bei anforderung einer einzelnen spezies werden diese ebenfalls hinzugefuegt
        'only' - nur ejected particles werden ausgegeben
        'ignore' - ejected particles werden nicht mit ausgegeben.
        """
        if species == 'ions':
            return ParticleAnalyzer(self.data, *self.ions(ejected=ejected))
        elif species == 'nonions':
            return ParticleAnalyzer(self.data, *self.nonions(ejected=ejected))
        else:
            return ParticleAnalyzer(self.data, species)

    
    def getderived(self):
        """Gibt alle Keys zurueck die mit "Derived/" beginnen"""
        ret = []
        for key in self.data.keys():
            r = re.match('Derived/[\w/ ]*', key)
            if r:
                ret.append(r.group(0))
        ret.sort()
        return ret






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
        

        


class _SingleSpeciesAnalyzer(_Constants):
    """
    Wird ausschliesslich von der ParticleAnalyzer Klasse benutzt.
    Oberste Prioritaet haben die Spezies, die in masslist und chargelist eingetragen sind.
    """
    masslist = {'electrongold':1, 'proton':1836*1,
        'ionp':1836,'ion':1836*12,'c6':1836*12, 
        'ionf':1836*19,'Palladium':1836*106, 
        'Palladium1':1836*106, 'Palladium2':1836*106, 
        'Ion':1836, 'Photon':0, 'Positron':1, 'positron':1,
        'gold1':1836*197,'gold2':1836*197,'gold3':1836*197,
        'gold4':1836*197,'gold7':1836*197,'gold10':1836*197,
        'gold20':1836*197} #unit: electronmass
    chargelist = {'electrongold':-1, 'proton':1,
        'ionp':1,'ion':1, 'c6':6, 
        'ionf':1,'Palladium':0, 
        'Palladium1':1, 'Palladium2':2, 
        'Ion':1, 'Photon':0, 'Positron':1, 'positron':1,
        'gold1':1,'gold2':2,'gold3':3,
        'gold4':4,'gold7':7,'gold10':10,
        'gold20':20} #unit: elementary charge
        
    isionlist = {'electrongold':False, 'proton':True,
        'ionp':True,'ion':True, 'c6':True, 
        'ionf':True,'f9':True,'Palladium':True, 
        'Palladium1':True, 'Palladium2':True, 
        'Ion':True, 'Photon':False, 'Positron':False, 'positron':False,
        'gold1':True,'gold2':True,'gold3':True,
        'gold4':True,'gold7':True,'gold10':True,
        'gold20':True} 
    
    masslistelement = {'H':1, 'He':4, 'C':12, 'N':14,'O':16, 'F':19, 
        'Ne': 20.2, 'Al':27, 'Si':28, 'Ar':40, 'Au':197} #unit: amu for convenience
        
    @staticmethod
    def isejected(species):
        s = species.replace('/','')
        r = re.match(r'(ejected_)(.*)', s)
        return not r==None
        
    @staticmethod    
    def retrieveparticleinfo(species):
        """
        Returns a dictionary contining particle informations. 
        mass in kg (SI)
        charge in C (SI)
        """
        import re
        ret = {}
        s = species.replace('/','')
        #Evtl. vorangestelltes "ejected_" entfernen
        r = re.match(r'(ejected_)?(.*)', s)
        s = r.group(2)
            
        #Name ist Elementsymbol und Ladungszustand, Bsp: C1, C6, F2, F9, Au20, Pb34a
        r = re.match('([A-Za-z]+)(\d+)([a-z]*)', s)
        if r:
            if _SingleSpeciesAnalyzer.masslistelement.has_key(r.group(1)):
                ret.update({'mass' : float(_SingleSpeciesAnalyzer.masslistelement[r.group(1)]) * 1836.2 * _Constants._me})
                ret.update({'charge' : float(r.group(2)) * _Constants._qe})
                ret.update({'ision' : True})
                return ret
                
        #Elektron identifizieren
        r = re.match('[Ee]le[ck]tron\d*', s)
        if r:
            ret.update({'mass' : _Constants._me})
            ret.update({'charge' : -1})
            ret.update({'ision' : False})
            return ret
        
        # --- seltener Bloedsinn. Sollte nicht besser nicht verwendet werden
        #Name ist ion mit charge (in Elementarladungen) und mass (in amu), Bsp: ionc1m1, ionc20m110,...
        r = re.match('ionc(\d+)m(\d+)', s)
        if r != None:
            ret.update({'mass' : float(r.group(2)) * 1836.2 * _Constants._me})
            ret.update({'charge' : float(r.group(1)) * 1836.2 * _Constants._qe})
            ret.update({'ision' : True})
        r = re.match('ionm(\d+)c(\d+)', s)
        if r != None:
            ret.update({'mass' : float(r.group(1)) * 1836.2 * _Constants._me})
            ret.update({'charge' : float(r.group(2)) * 1836.2 * _Constants._qe})
            ret.update({'ision' : True})
                
        # einzeln in Liste masslist und chargelist
        if _SingleSpeciesAnalyzer.masslist.has_key(s):
            ret.update({'mass': float(_SingleSpeciesAnalyzer.masslist[s]) * _Constants._me})
        if _SingleSpeciesAnalyzer.chargelist.has_key(s):
            ret.update({'charge': float(_SingleSpeciesAnalyzer.chargelist[s] * _Constants._qe)})
        if _SingleSpeciesAnalyzer.isionlist.has_key(s):
            ret.update({'ision': _SingleSpeciesAnalyzer.isionlist[s]})
        
        assert ret.has_key('mass') & ret.has_key('charge'), 'Masse/Ladung der Spezies ' + species + ' nicht gefunden.'
        return ret
        

    def __init__(self, data, species):
        self.species = species
        self.speciesexists = True
        if not data.has_key('Particles/Weight'+species):
            self.speciesexists = False
            return
        self.data = data
        self._weightdata = data['Particles/Weight' + species]
        self._Xdata = data['Grid/Particles' + species + '/X']
        self._Ydata = np.array([0,1])
        self._Zdata = np.array([0,1])
        self.simdimensions = 1
        if data.has_key('Grid/Particles'+species+'/Y'):
            self._Ydata = data['Grid/Particles'+species+'/Y']
            self.simdimensions = 2
        if data.has_key('Grid/Particles'+species+'/Z'):
            self._Zdata = data['Grid/Particles'+species+'/Z']
            self.simdimensions = 3
        self._Pxdata = data['Particles/Px'+species]
        self._Pydata = data['Particles/Py'+species]
        self._Pzdata = data['Particles/Pz'+species]
        self._particleinfo = self.retrieveparticleinfo(species) 
        self._mass = self._particleinfo['mass'] #SI
        self._charge = self._particleinfo['charge'] #SI
        self.compresslog=[]
            

    def compress(self, condition, name='unknown condition'):
        """
        In Anlehnung an numpy.compress.  Zusatzlich, kann ein name angegeben werden, der fortlaufend in compresslog gespeichert wird.
        Bsp.:
        cfintospectrometer = lambda x: x.angle_offaxis() < 30e-3 
        cfintospectrometer.name = '< 30mrad offaxis'
        pa.compress(cfintospectrometer(pa), name=cfintospectrometer.name)
        """
        assert self._weightdata.shape[0] == condition.shape[0], 'condition hat die falsche Laenge'
        self._weightdata = np.compress(condition, self._weightdata)
        self._Xdata = np.compress(condition, self._Xdata)
        if self.simdimensions > 1:
            self._Ydata = np.compress(condition, self._Ydata)
        if self.simdimensions > 2:
            self._Zdata = np.compress(condition, self._Zdata)
        self._Pxdata = np.compress(condition, self._Pxdata)
        self._Pydata = np.compress(condition, self._Pydata)
        self._Pzdata = np.compress(condition, self._Pzdata)
        self.compresslog = np.append(self.compresslog, name)

    def uncompress(self):
        """
        Verwirft alle Einschraenkungen (insbesondere durch compress). Reinitialisiert das Objekt.
        """
        self.__init__(self.data, self.species)

        
    # --- Stellt ausschliesslich GRUNDLEGENDE funktionen bereit
    
    def weight(self): # np.float64(np.array([4.3])) == 4.3 fuehrt sonst zu Fehler
        return np.asfarray(self._weightdata, dtype='float64')
    def mass(self): #SI 
        return np.repeat(self._mass, self.weight().shape[0])
    def charge(self): #SI
        return np.repeat(self._charge, self.weight().shape[0])
    def Px(self):
        return np.float64(self._Pxdata)
    def Py(self):
        return np.float64(self._Pydata)
    def Pz(self):
        return np.float64(self._Pzdata)
    def X(self):
        return np.float64(self._Xdata)
    def Y(self):
        return np.float64(self._Ydata)
    def Z(self):
        return np.float64(self._Zdata)        
        
        
        

        
class ParticleAnalyzer(_Constants):
    """
    Hat die Gleiche Funktionilitaet wie SingleSpeciesAnalyzer, jedoch koennen mehrere ParticleAnalyzer addiert werden, um die Gesamtheit der Teilchen auszuwaehlen.
    """
    
    @staticmethod
    def ision(species):
        return _SingleSpeciesAnalyzer.retrieveparticleinfo(species)['ision']
        
    @staticmethod
    def isejected(species):
        return _SingleSpeciesAnalyzer.isejected(species)
        
    
    def __init__(self, data, *speciess):
        #self.data = data
        self._speciess = speciess
        self.species = ''
        self._ssas = []
        self.simdimensions = None
        for s in speciess:
            ssa = _SingleSpeciesAnalyzer(data,s)
            if ssa.speciesexists:
                self.species = self.species + s
                self._ssas.append(ssa)
                self.simdimensions = self._ssas[0].simdimensions
        self._compresslog = []

        #set default extents
        self.simextent = np.real([data['Grid/Grid_node/X'][0], data['Grid/Grid_node/X'][-1]])
        self.simgridpoints = [np.real(data['Grid/Grid_node/X']).shape[0] - 1]
        self.X.__func__.extent=self.simextent[0:2]
        self.X.__func__.gridpoints=self.simgridpoints[0]
        self.X_um.__func__.extent=self.simextent[0:2] * 1e6
        self.X_um.__func__.gridpoints=self.simgridpoints[0]
        if self.simdimensions > 1:
            self.simextent = np.append(self.simextent, np.real([data['Grid/Grid_node/Y'][0], data['Grid/Grid_node/Y'][-1]]))
            self.simgridpoints = np.append(self.simgridpoints, np.real(data['Grid/Grid_node/Y']).shape[0] - 1)
            self.Y.__func__.extent=self.simextent[2:4]
            self.Y.__func__.gridpoints=self.simgridpoints[1]
            self.Y_um.__func__.extent=self.simextent[2:4] * 1e6
            self.Y_um.__func__.gridpoints=self.simgridpoints[1]
        if self.simdimensions > 2:
            self.simextent = np.append(self.simextent, np.real([data['Grid/Grid_node/Z'][0], data['Grid/Grid_node/Z'][-1]]))
            self.simgridpoints = np.append(self.simgridpoints, np.real(data['Grid/Grid_node/Z']).shape[0] - 1)
            self.Z.__func__.extent=self.simextent[4:6]
            self.Z.__func__.gridpoints=self.simgridpoints[2]
            self.Z_um.__func__.extent=self.simextent[4:6] * 1e6
            self.Z_um.__func__.gridpoints=self.simgridpoints[2]
        self.angle_xy.__func__.extent=np.real([-np.pi, np.pi])
        self.angle_yz.__func__.extent=np.real([-np.pi, np.pi])
        self.angle_zx.__func__.extent=np.real([-np.pi, np.pi])
        self.angle_offaxis.__func__.extent=np.real([0, np.pi])
        
    def __str__(self):
        return '<ParticleAnalyzer including ' + str(self._speciess) +'>'
        
    def __len__(self):
        """
        identisch zu self.N()
        """
        return self.N()
        
    # --- Funktionen, um ParticleAnalyzer zu kombinieren
        
    def append(self, other):
        if len(self._ssas) > 0:
            self._ssas = copy.deepcopy(self._ssas) + copy.deepcopy(other._ssas)
            self.species = self.species + other.species
        return self
        
    def __add__(self, other): # self + other
        ret = copy.copy(self)
        ret.append(other)
        return ret
        
    def __iadd__(self, other): # self += other
       return self.append(other)
       
        
    # --- nur GRUNDLEGENDE Funktionen auf SingleSpeciesAnalyzer abbilden 
    
    def _funcabbilden(self, func):
        ret = np.array([])
        for ssa in self._ssas:
            a = getattr(ssa, func)()
            ret = np.append(ret, a)
        return ret
    
    def _weight(self):
        return self._funcabbilden('weight')
        
    def _mass(self): #SI
        return self._funcabbilden('mass')
    
    def _charge(self): #SI
        return self._funcabbilden('charge')
        
    def _Px(self):
        return self._funcabbilden('Px')
        
    def _Py(self):
        return self._funcabbilden('Py')
        
    def _Pz(self):
        return self._funcabbilden('Pz')
        
    def _X(self):
        return self._funcabbilden('X')
        
    def _Y(self):
        return self._funcabbilden('Y')
        
    def _Z(self):
        return self._funcabbilden('Z')
        
    def compress(self, condition, name='unknown condition'):
        i = 0
        for ssa in self._ssas:
            n = ssa.weight().shape[0]
            ssa.compress(condition[i:i+n], name=name)
            i += n
        self._compresslog = np.append(self._compresslog, name)
        
    # --- Hilfsfunktionen
    
    def compressfn(self, conditionf, name='unknown condition'):
        if hasattr(conditionf, 'name'):
            name = conditionf.name
        self.compress(conditionf(self), name=name)
        
    def uncompress(self):
        self._compresslog=[]
        for s in self._ssas:
            s.uncompress()
        #self.__init__(self.data, *self._speciess)
        
    def _mass_u(self):
        return self._mass() / self._me / 1836.2
    
    def _Eruhe(self):
        return self._mass() * self._c**2
        
    def getcompresslog(self):
        ret = {'all': self._compresslog}
        for ssa in self._ssas:
            ret.update({ssa.species: ssa.compresslog})
        return ret

    def N(self):
        return self._weight().shape[0]
    
    # --- Skalarfunktionen. Ordnen jedem Teilchen ein Skalar zu.
    
    def weight(self):
        return self._weight()
    weight.name='Particle weight'
    weight.unit=''
    def Px(self):
        return np.self._Px()
    Px.unit=''
    Px.name='Px'
    def Py(self):
        return self._Py()
    Py.unit=''
    Py.name='Py'
    def Pz(self):
        return self._Pz()
    Pz.unit=''
    Pz.name='Pz'
    def P(self):
        return np.sqrt(self._Px()**2 + self._Py()**2 + self._Pz()**2)
    P.unit=''
    P.name='P'
    def X(self):
        return self._X()
    X.unit='m'
    X.name='X'
    def X_um(self):
        return self._X() * 1e6
    X_um.unit='$\mu m$'
    X_um.name='X'
    def Y(self):
        return self._Y()
    Y.unit='m'
    Y.name='Y'
    def Y_um(self):
        return self._Y() * 1e6
    Y_um.unit='$\mu m$'
    Y_um.name='Y'
    def Z(self):
        return self._Z()
    Z.unit='m'
    Z.name='Z'
    def Z_um(self):
        return self._Z() * 1e6
    Z_um.unit='$\mu m$'
    Z_um.name='Z'
    def beta(self):
        return np.sqrt(self.gamma()**2 - 1)/self.gamma()
    beta.unit=r'$\beta$'
    beta.name='beta'
    def V(self):
        return self._c * self.beta()
    V.unit='m/s'
    V.name='V'
    def gamma(self):
        return np.sqrt(1 + (self._Px()**2 + self._Py()**2 + self._Pz()**2)/(self._mass() * self._c)**2)
    gamma.unit=r'$\gamma$'
    gamma.name='gamma'
    def Ekin(self):
        return (self.gamma() - 1) * self._Eruhe()
    Ekin.unit='J'
    Ekin.name='Ekin'
    def Ekin_MeV(self):
        return self.Ekin() / self._qe / 1e6
    Ekin_MeV.unit='MeV'
    Ekin_MeV.name='Ekin'
    def Ekin_MeV_amu(self):
        return self.Ekin_MeV() / self._mass_u()
    Ekin_MeV_amu.unit='MeV / amu'
    Ekin_MeV_amu.name='Ekin / amu'
    def Ekin_keV(self):
        return self.Ekin() / self._qe / 1e3
    Ekin_keV.unit='keV'
    Ekin_keV.name='Ekin'
    def Ekin_keV_amu(self):
        return self.Ekin_keV() / self._mass_u()
    Ekin_keV_amu.unit='keV / amu'
    Ekin_keV_amu.name='Ekin / amu'
    def angle_xy(self):
        return np.arctan2(self._Py(), self._Px())
    angle_xy.unit='rad'
    angle_xy.name='anglexy'    
    def angle_yz(self):
        return np.arctan2(self._Pz(), self._Py())
    angle_yz.unit='rad'
    angle_yz.name='angleyz'
    def angle_zx(self):
        return np.arctan2(self._Px(), self._Pz())
    angle_zx.unit='rad'
    angle_zx.name='anglezx'
    def angle_offaxis(self):
        return np.arccos(self._Px() / (self.P() + 1e-300))
    angle_offaxis.volumenelement = lambda theta: 1#/np.sin(theta)
    angle_offaxis.unit='rad'
    angle_offaxis.name='angleoffaxis'


    # ---- Hilfen zum erstellen des Histogramms ---
    

    def createHistgram1d(self, scalarfx, optargsh={'bins':300}, simextent=False, simgrid=False):
        if simgrid:
            simextent = True
        #Falls alle Teilchen aussortiert wurden, z.B. durch ConditionFunctions
        if len(scalarfx(self)) == 0: 
            return [0], [0]
        rangex = [np.min(scalarfx(self)), np.max(scalarfx(self))]
        if simextent:
            if hasattr(scalarfx, 'extent'):
                rangex = scalarfx.extent
        if simgrid:
            if hasattr(scalarfx, 'gridpoints'):
                optargsh['bins'] = scalarfx.gridpoints
        h, edges = np.histogram(scalarfx(self), weights=self.weight(), range=rangex, **optargsh)
        h = h/np.diff(edges) #um auf Teilchen pro xunit zu kommen
        if hasattr(scalarfx, 'volumenelement'):
            h = h * scalarfx.volumenelement(np.convolve(edges, [0.5, 0.5], 'valid'))
        x = np.convolve(edges, [0.5,0.5], 'valid')
        return h, x


    def createHistgram2d(self, scalarfx, scalarfy, optargsh={'bins':[500, 500]}, simextent=False, simgrid=False, rangex=None, rangey=None):
        """
        simgrid=True erzwingt, dass Ortsachsen dasselbe Grid zugeordnet wird wie in der Simulation. Bedingt simextent=True
        simextent=True erzwingt, dass Ortsachsen sich ueber den gleichen Bereich erstrecken, wie in der Simulation.
        """
        if simgrid:
            simextent = True
	### TODO: Falls rangex oder rangy gegeben ist, ist die Gesamtteilchenzahl falsch berechnet, weil die Teilchen die ausserhalb des sichtbaren Bereiches liegen mitgezaehlt werden.
        if rangex is None:
		rangex = [np.min(scalarfx(self)), np.max(scalarfx(self)) + 1e-7]
        if rangey is None:
        	rangey = [np.min(scalarfy(self)), np.max(scalarfy(self)) + 1e-7]
        if simextent:
            if hasattr(scalarfx, 'extent'):
                rangex = scalarfx.extent
            if hasattr(scalarfy, 'extent'):
                rangey = scalarfy.extent
        if simgrid:
            if hasattr(scalarfx, 'gridpoints'):
                optargsh['bins'][0] = scalarfx.gridpoints
            if hasattr(scalarfy, 'gridpoints'):
                optargsh['bins'][1] = scalarfy.gridpoints
        h, xedges, yedges = np.histogram2d(scalarfx(self), scalarfy(self), weights=self.weight(), range=[rangex, rangey], **optargsh)
        h = h / (xedges[1]-xedges[0]) / (yedges[1] - yedges[0])
        if hasattr(scalarfy, 'volumenelement'):
            h = h * scalarfy.volumenelement(np.convolve(yedges, [0.5, 0.5], 'valid'))
        if hasattr(scalarfx, 'volumenelement'):
            h = (h.T * scalarfx.volumenelement(np.convolve(xedges, [0.5, 0.5], 'valid'))).T
        extent = np.array([xedges[0], xedges[-1], yedges[0], yedges[-1]])
        return h, xedges, yedges, extent


    def createHistgramFeld1d(self, scalarfx, optargsh={'bins':300}, simextent=False, simgrid=False, name='distfn', title=None):
        h, x = self.createHistgram1d(scalarfx, simextent=simextent, simgrid=simgrid, optargsh=optargsh)
        ret = Feld(h)
        #ret.extent = np.array([x[0], x[-1]])
        ret.setgrid_node(0, x)        
        ret.name = name
        ret.name2 = ret.name2 + self.species
        ret.label = self.species
        if title:
            ret.name = title
        if hasattr(scalarfx, 'unit'):
            ret.axesunits = [scalarfx.unit]
        if hasattr(scalarfx, 'name'):
            ret.axesnames = [scalarfx.name]
        ret.textcond = self.getcompresslog()['all']
        ret.zusatz = self.N()
        return ret


   # def createHistgramFeld2d(self, scalarfx, scalarfy, optargsh={'bins':[500, 500]}, simextent=False, simgrid=False, name='distfn', title=None):
    def createHistgramFeld2d(self, scalarfx, scalarfy, **kwargs):
	name = 'distfn'
	title = None
	if kwargs.has_key('name'):
		name = kwargs.pop('name')
	if kwargs.has_key('title'):
		title = kwargs.pop('title')
        h, xedges, yedges, extent = self.createHistgram2d(scalarfx, scalarfy, **kwargs)
        ret = Feld(h)
        #ret.extent = extent
        ret.setgrid_node(0, xedges)
        ret.setgrid_node(1, yedges)
        ret.name = name + self.species
        ret.name2 = ret.name2 + self.species
        if title:
            ret.name = title
        ret.axesunits = [scalarfx.unit, scalarfy.unit]
        ret.axesnames = [scalarfx.name, scalarfy.name]
        ret.zusatz = "%.0f particles" % self.N()
        ret.textcond = self.getcompresslog()['all']    
        return ret
        
    def createFeld(self, *scalarf, **kwargs):
        if self.simdimensions == None:
            return None
        if len(scalarf) == 1:
            return self.createHistgramFeld1d(*scalarf, **kwargs)
        elif len(scalarf) == 2:
            return self.createHistgramFeld2d(*scalarf, **kwargs)
        else:
            raise Exception('createFeld kann nur 1 oder 2 Skalarfunktionen entgegennehmen')





class FieldAnalyzer(_Constants):
    

    def __init__(self, data, lasnm=None):
        self.data=data
        self.lasnm=lasnm
        if lasnm:
            self.k0 = 2 * np.pi / (lasnm * 1e-9)
        else:
            self.k0 = None
        self.simdimensions = 1
        self._simextent = np.float64([data['Grid/Grid_node/X'][0], data['Grid/Grid_node/X'][-1]])
        self._simgridpoints = np.float64([len(data['Grid/Grid_node/X'])-1])
        if data.has_key('Grid/Grid_node/Y'):
            self._simextent = np.append(self._simextent, np.float64([data['Grid/Grid_node/Y'][0], data['Grid/Grid_node/Y'][-1]]))
            self._simgridpoints = np.append(self._simgridpoints, np.float64([len(data['Grid/Grid_node/Y'])-1]))
            self.simdimensions = 2
        if data.has_key('Grid/Grid_node/Z'):
            self._simextent = np.append(self._simextent, np.float64([data['Grid/Grid_node/Z'][0], data['Grid/Grid_node/Z'][-1]]))
            self._simgridpoints = np.append(self._simgridpoints, np.float64([len(data['Grid/Grid_node/Z'])-1]))
            self.simdimensions = 3
        self._extent = self._simextent.copy() #Variable definiert den ausgeschnittenen Bereich
    
    def datenausschnitt_bound(self, m):
        return self.datenausschnitt(m, self._simextent, self._extent)

    def getsimextent(self, axis=None):    
        return self._simextent.copy()[self._axisoptions[axis]]

    def getextent(self, axis=None):
        return self._extent.copy()[self._axisoptions[axis]]

    def getsimgridpoints(self, axis=None):
        return self._simgridpoints.copy()[self._axisoptionseinzel[axis]]
    
    def getsimdomainsize(self, axis=None):
        return np.diff(self.getsimextent(axis))[0::2]

    def getspatialresolution(self, axis=None):
        return self.getsimdomainsize() / self.getsimgridpoints()

    def setextent(self, newextent, axis=None):
        self._extent[self._axisoptions[axis]] = newextent
        
    def setspacialtofield(self, field):
        """
        Fuegt dem Feld alle Informationen uber das rauemliche Grid hinzu.
        """
        field.setallaxesspacial()
        field.setgrid_node(0, self.data['Grid/Grid_node/X'])
        if self.data.has_key('Grid/Grid_node/Y'):
            field.setgrid_node(1, self.data['Grid/Grid_node/Y'])
        if self.data.has_key('Grid/Grid_node/Z'):
            field.setgrid_node(2, self.data['Grid/Grid_node/Z'])
        return None


    # --- Return functions for basic data layer
    
    # -- very basic --
    
    def _returnkey(self, key):
        assert self.data.has_key(key), 'Den Key ' + key + ' gibts nicht!'
        return self.datenausschnitt_bound(np.float64(self.data[key]))
        
    def _returnkey2(self, key1, key2, average=False):
        key = key1 + key2
        if average:
            key = key1 + '_average' + key2
        return self._returnkey(key)
    
    # -- just basic --
    
    def _Ex(self, **kwargs):
        return self._returnkey2('Electric Field', '/Ex', **kwargs)
    def _Ey(self, **kwargs):
        return self._returnkey2('Electric Field', '/Ey', **kwargs)
    def _Ez(self, **kwargs):
        return self._returnkey2('Electric Field', '/Ez', **kwargs)
    def _Bx(self, **kwargs):
        return self._returnkey2('Magnetic Field', '/Bx', **kwargs)
    def _By(self, **kwargs):
        return self._returnkey2('Magnetic Field', '/By', **kwargs)
    def _Bz(self, **kwargs):
        return self._returnkey2('Magnetic Field', '/Bz', **kwargs)


    # --- Alle Funktionen geben ein Objekt vom Typ Feld zurueck

    # allgemein ueber dem Ort auftragen. Insbesondere fuer Derived/*
    def createfeldfromkey(self, key):
        ret = Feld(self._returnkey(key));
        ret.name = key
        self.setspacialtofield(ret)
        return ret

    def createfelderfromkeys(self, *keys):
        ret = ()
        for key in keys:
            ret+=(self.createfeldfromkey(key),)
        return ret


    # jetzt alle einzeln    
    def Ex(self, **kwargs):
        ret = Feld(self._Ex(**kwargs))
        ret.unit='V/m'
        ret.name='Ex'
        self.setspacialtofield(ret)
        return ret
        
    def Ey(self, **kwargs):
        ret = Feld(self._Ey(**kwargs))
        ret.unit='V/m'
        ret.name='Ey'
        self.setspacialtofield(ret)
        return ret
        
    def Ez(self, **kwargs):
        ret = Feld(self._Ez(**kwargs))
        ret.unit='V/m'
        ret.name='Ez'
        self.setspacialtofield(ret)
        return ret

    def Bx(self, **kwargs):
        ret = Feld(self._Bx(**kwargs))
        ret.unit='T'
        ret.name='Bx'
        self.setspacialtofield(ret)
        return ret

    def By(self, **kwargs):
        ret = Feld(self._By(**kwargs))
        ret.unit='T'
        ret.name='By'
        self.setspacialtofield(ret)
        return ret
        
    def Bz(self, **kwargs):
        ret = Feld(self._Bz(**kwargs))
        ret.unit='T'
        ret.name='Bz'
        self.setspacialtofield(ret)
        return ret



    # --- spezielle Funktionen
    
    def energydensityE(self, **kwargs):
        ret = Feld( 0.5 * self._epsilon0 * (self._Ex(**kwargs)**2 + self._Ey(**kwargs)**2 + self._Ez(**kwargs)**2) )
        ret.unit='J/m^3'
        ret.name='Energy Density Electric-Field'
        self.setspacialtofield(ret)
        return ret
        
    def energydensityM(self, **kwargs):
        ret = Feld( 0.5 / self._mu0 * (self._Bx(**kwargs)**2 + self._By(**kwargs)**2 + self._Bz(**kwargs)**2) )
        ret.unit='J/m^3'
        ret.name='Energy Density Magnetic-Field'
        self.setspacialtofield(ret)
        return ret

    def energydensityEM(self, **kwargs):
        ret = Feld( 0.5 * self._epsilon0 * (self._Ex(**kwargs)**2 + self._Ey(**kwargs)**2 + self._Ez(**kwargs)**2) \
             + 0.5 / self._mu0 * (self._Bx(**kwargs)**2 + self._By(**kwargs)**2 + self._Bz(**kwargs)**2) )
        ret.unit='J/m^3'
        ret.name='Energy Density EM-Field'
        self.setspacialtofield(ret)
        return ret
        
    # --- Spektren
    
    def spectrumEx(self, axis=0):
        if self.k0 == None:
            ret = Feld(None)
            print 'WARNING: lasnm not given. Spectrum will not be calculated.'
        else:
            rfftaxes = np.roll((0,1),axis)
            ret = Feld( 0.5 * self._epsilon0 * abs(np.fft.fftshift(np.fft.rfft2(self._Ex(),axes=rfftaxes), axes=axis))**2 )
        ret.unit='?'
        ret.name='Spectrum Ex'
        ret.setallaxes(name=[r'$k_x$', r'$k_y$', r'$k_z$'], unit=['','',''])
        extent = np.zeros(2*self.simdimensions)
        extent[1::2] = np.pi / self.getspatialresolution()
        if self.k0:
            ret.setallaxes(name=[r'$k_x / k_0$', r'$k_y / k_0$', r'$k_z / k_0$'], unit=['$\lambda_0 =$' + str(self.lasnm) + 'nm','',''])
            extent[1::2] = extent[1::2] / self.k0
        mittel = np.mean(extent[(0+2*axis):(2+2*axis)])
        extent[0+2*axis] = -2*mittel
        extent[1+2*axis] = 2*mittel
        ret.setgrid_node_fromextent(extent)
        return ret
        
    def spectrumBz(self, axis=0):
        if self.k0 == None:
            ret = Feld(None)
            print 'WARNING: lasnm not given. Spectrum will not be calculated.'
        else:
            rfftaxes = np.roll((0,1),axis)
            ret = Feld( 0.5 / self._mu0 * abs(np.fft.fftshift(np.fft.rfft2(self._Bz(), axes=rfftaxes), axes=axis))**2 )
        ret.unit='?'
        ret.name='Spectrum Bz'
        ret.setallaxes(name=[r'$k_x$', r'$k_y$', r'$k_z$'], unit=['','',''])
        extent = np.zeros(2*self.simdimensions)
        extent[1::2] = np.pi / self.getspatialresolution()
        if self.k0:
            ret.setallaxes(name=[r'$k_x / k_0$', r'$k_y / k_0$', r'$k_z / k_0$'], unit=['$\lambda_0 =$' + str(self.lasnm) + 'nm','',''])
            extent[1::2] = extent[1::2] / self.k0
        mittel = np.mean(extent[(0+2*axis):(2+2*axis)])
        extent[0+2*axis] = -2*mittel
        extent[1+2*axis] = 2*mittel
        ret.setgrid_node_fromextent(extent)
        return ret




## ----- Hilfsfunktionen zum Plotten und speichern -----


class Feld(_Constants):
    """
    Repraesentiert ein Feld, das spaeter dirket geplottet werden kann.
    """
        
    def __init__(self, matrix):
        self.matrix=np.array(matrix)
        self.grid_nodes = []
        self.axesnames = []
        self.axesunits = []
        for i in xrange(self.dimensions()):
            self.addaxis()
        self.name='unbekannt'
        self.name2=''
        self.label=''
        self.unit=None           
        self.zusatz = ''
        self.textcond = ''
        self._grid_nodes_linear = True #None bedeutet unbekannt

    def __str__(self):
        return '<Feld '+self.name +' '+str(self.matrix.shape)+'>'
        
    def addaxis(self, name='', unit='?'):
        self.axesnames.append(name)
        self.axesunits.append(unit)
        self.grid_nodes.append(np.array([0,1]))
        return self
    
    def extent(self):
        ret = []
        for traeger in self.grid_nodes:
            ret.append(traeger[0])
            ret.append(traeger[-1])
        return ret
        
    def grid_nodes_linear(self, force=True):
        """
        Testet, ob die grid_nodes linear verteielt sind. Ist das der Fall, reicht es auch nur mit extent() zu plotten.
        Ausgabe ist die Liste der Varianzen der Grid Node Abstaende.
        """
        if self._grid_nodes_linear == None or force:
            self._grid_nodes_linear = all([np.var(np.diff(gn)) < 1e-7 for gn in self.grid_nodes ])
        return self._grid_nodes_linear 
            
        
        
    def setgrid_node(self, axis, grid_node):
        if axis < self.dimensions():
            self.grid_nodes[axis] = np.float64(grid_node)
            self._grid_nodes_linear = None
        return self
        
    def setgrid_node_fromextent(self, extent):
        """
        Errechnet eigene Werte fuer grid_node, falls nur der extent gegeben wurde.
        Vereinfacht Kompatibilitaet, aber es ist empfohlen setgrid_node(self, axis, grid_node) direkt aufzurufen. 
        """
        assert not len(extent)%2, 'len(extent) ist kein Vielfaches von 2.' 
        for dim in xrange(self.dimensions()):
            self.setgrid_node(dim, np.linspace(extent[2*dim], extent[2*dim + 1], self.matrix.shape[dim] + 1))
        return self
        
    def setgrid_node_fromgrid(self, axis, grid):
        """
        grid_node beinhaltet die Kanten des Grids. In 1D gehoert also zu einem Datenfeld der Laenge 1000 ein grid_node Vektor der Laenge 1001.
        grid beinhaltet die Positionen. In 1D gehoert also zu einem Datenfeld der Laenge 1000 ein grid Vektor der Laenge 1000.
        """
        gn = np.convolve(grid, np.ones(2)/2.0, mode='full')
        gn[0] = grid[0] + 2 * (grid[0] - gn[1])
        gn[-1] = grid[-1] + 2 * (grid[-1] - gn[-2])
        return self.setgrid_node(axis, gn)
        
    def ausschnitt(self, ausschnitt):
        if self.dimensions() == 0:
            return
        if self.extent != None:
            raise Exception('extent kann nicht geaendert werden, wenn der aktuelle extent unbekannt ist.')
        self.matrix = _Constants.datenausschnitt(self.matrix, self.extent, ausschnitt)
        self.extent = ausschnitt
    
    def dimensions(self):
        return len(self.matrix.shape)
        
    def savename(self):
        return self.name + ' ' + self.name2

    def mikro(self):
        #self.grid_nodes *= 1e6
        map(lambda x: x*1e6, self.grid_nodes)
        self.axesunits = ['$\mu $'+x for x in self.axesunits]
        return self        
        
    def grid(self):
        """
        Creates lists containing X and Y coordinates of the data (in 2D case). Those can be directly parsed to matplotlib.pyplot.pcolormesh
        
        Also von grid_node (Laenge N+1)  auf grid (Laenge N) konvertieren. 
        """
        return tuple([np.convolve(gn, np.ones(2)/2.0, mode='valid') for gn in self.grid_nodes])

    def setallaxes(self, name=None, unit=None):
        def setlist(arg):
            if isinstance(arg,list):
                return [arg[dim] for dim in xrange(self.dimensions())]
            else:
                return [arg for dim in xrange(self.dimensions())]
        if name:
              self.axesnames = setlist(name)
        if unit:
            self.axesunits = setlist(unit)
        return self
        
    def setallaxesspacial(self):
        """
        Alle (vorhandenen) Achsen werden zu Raumachsen.
        """
        self.setallaxes(name=['X','Y','Z'], unit=[r'$m$', r'$m$', r'$m$'])

    def mean(self, axis=-1):
        if self.dimensions() == 0:
            return self
        self.matrix=np.mean(self.matrix, axis=axis)
        self.axesunits.pop(axis)
        self.axesnames.pop(axis)
        if self.extent() != None:
            self.grid_nodes.pop(axis)
            #self.extent = np.delete(self.extent(), [2*axis, 2*axis+1])
        return self

    def topolar(self, extent=None, shape=None, angleoffset=0):
        """Transformiert die Aktuelle Darstellung in Polardarstellung. extent und shape = None bedeutet automatisch.
extent=(phimin, phimax, rmin, rmax)"""
        ret = copy.deepcopy(self)
        if extent==None:
            extent=[-np.pi, np.pi,0, self.extent()[1]]
        extent=np.asarray(extent)
        if shape==None:
            shape = (1000, np.min((np.floor(np.min(self.matrix.shape) / 2), 1000)) )
        
        extent[0:2]=extent[0:2] - angleoffset
        ret.matrix = self.transfromxy2polar(self.matrix, self.extent(), np.roll(extent,2), shape).T
        extent[0:2]=extent[0:2] + angleoffset
    
        ret.setgrid_nodefromextent(extent)    
        if ret.axesnames[0].startswith('$k_') and ret.axesnames[1].startswith('$k_'):
            ret.axesnames[0]='$k_\phi$'
            ret.axesnames[1]='$|k|$'
        return ret

    def exporttocsv(self, dateiname):
        if self.dimensions() == 1:
            data = np.asarray(self.matrix)
            x = np.linspace(self.extent()[0], self.extent()[1], len(data))
            np.savetxt(dateiname, np.transpose([x, data]), delimiter=' ')
        elif self.dimensions() == 2:
            export = np.asarray(self.matrix)
            np.savetxt(dateiname, export)
        else:
            raise Exception('Not Implemented')
        
    def __add__(self, other):
        ret = copy.deepcopy(self)
        ret.matrix = self.matrix + other.matrix
        return ret


    def __pow__(self, other):
        ret = copy.deepcopy(self)
        ret.matrix = self.matrix**other
        return ret





class SDFPlots(_Constants):

    #@staticmethod
    #def axesformat(x, pos): #'The two args are the value and tick position'
    #    return '%.3f' % x #1.1float
    #@staticmethod
    #def axesformatexp(x, pos): #'The two args are the value and tick position'
    #    return '%.1e' % x 
    axesformatterx = matplotlib.ticker.ScalarFormatter()
    axesformatterx.set_powerlimits((-2,3))
    axesformattery = matplotlib.ticker.ScalarFormatter()
    axesformattery.set_powerlimits((-2,3))
 
    efeldcdict={'red': ((0,0,0),(1,1,1)),
        'green': ((0,0,0),(1,0,0)),
        'blue':    ((0,1,1),(1,0,0)),
        'alpha': ((0,1,1),(0.5,0,0),(1,1,1))}
    lsc=LinearSegmentedColormap('EFeld',efeldcdict,1024)
    plt.register_cmap(cmap=lsc, name='EFeld', data=efeldcdict)    


    def __init__(self, plotdescriptor, outdir=None):
        self.lasnm = plotdescriptor.getlasnm()
        self.outdir = outdir
        self.plotdescriptor = plotdescriptor        
        self._savenamesused = []
        if self.outdir == None:
            print 'Kein Ausgabeverzeichnis angegeben. Es werden keine Plots gespeichert.'

    def setzetext(self, titel, extray='', belowtime='', textcond=''):
        #plt.title(titel)
        plt.figtext(0.5 , 0.97, titel, ha='center', fontsize=14)
        if not self.plotdescriptor == None:
            plt.figtext(0.03, 0.965, self.plotdescriptor.getprojektname(), horizontalalignment='left')
            nexttext = str(self.plotdescriptor.getprojektname2())
            if isinstance(self.plotdescriptor.getprojektname4(), float):
                nexttext += ", step: %.0f" % (self.plotdescriptor.getprojektname4())
            elif isinstance(self.plotdescriptor.getprojektname4(), str):
                nexttext += " " + self.plotdescriptor.getprojektname4()
            else:
                pass
            plt.figtext(0.03, 0.94, nexttext, horizontalalignment='left')
            nexttext = ''
            if isinstance(self.plotdescriptor.getprojektname3(), float):
                nexttext += "%.1f fs" % (1e15*self.plotdescriptor.getprojektname3())
            elif isinstance(self.plotdescriptor.getprojektname3(), str):
                nexttext += " " + self.plotdescriptor.getprojektname3()
            else:
                pass
            plt.figtext(0.92, 0.96, nexttext, horizontalalignment='right')    
        plt.figtext(0.92, 0.93, belowtime, ha='right')
        plt.figtext(0.04, 0.5, extray, horizontalalignment='center', va='center', rotation='vertical')
        if textcond != None and textcond != []:
            plt.figtext(0.5, 0.87, textcond, ha='center')
    
    def setzetextfeld(self, feld, zusatz=None):
        if zusatz==None:
            zusatz=feld.zusatz
        if feld.dimensions() == 2:
            self.setzetext(feld.savename(), belowtime=zusatz, textcond=feld.textcond)
        else:
            self.setzetext(feld.savename(), belowtime=zusatz, textcond=feld.textcond) #oder nur feld.name, damit die spezies nicht mit angegeben wird?

    def symmetrisiereclim(self):
        """
        symmetrisiert die clim so, dass bei der colorbar auf jeden Fall eine 0 in der Mitte ist.
        """
        bound=max(abs(np.asarray(plt.gci().get_clim())))
        plt.clim(-bound,bound)

    def plotspeichern(self,key, dpi=150, facecolor=(1,1,1,0.01)):
	#Die sind zwar schoen, aber machen manchmal Probleme, wenn alle Werte null sind...
        plt.gca().xaxis.set_major_formatter(SDFPlots.axesformatterx);
        plt.gca().yaxis.set_major_formatter(SDFPlots.axesformattery);
        plt.gcf().set_size_inches(9,7)
        savename=self._savename(key)
        if savename != None:
            plt.savefig(savename, dpi=dpi, facecolor=facecolor, transparent=True);
            plt.close()


    def _savename(self, key):
        if self.outdir == None:
            return None
        name = self.outdir + self.plotdescriptor.getprojektname2()+'_' + str(len(self._savenamesused)) + '_' + key.replace('/','_').replace(' ','')+'_'+self.plotdescriptor.getprojektname()
        nametmp = name + '_%d'
        i = 0
        while name in self._savenamesused:
            i = i + 1
            name = nametmp % i
        self._savenamesused.append(name)
        #print name
        return name + '.png' 

    def lastsavename(self):
        """Gibt den zuletzt verwendeten Savename zurueck. zieht einen neuen, falls es keinen letzten gab."""
        if len(self._savenamesused) == 0:
            return self._savename('lastsavename')
        else:
            return self._savenamesused[-1]


    def _plotFeld1d(self, feld, log10plot=True, saveandclose=True, xlim=None, clim=None, scaletight=None, name=None, majorgrid=False, savecsv=False):
        assert feld.dimensions() == 1, 'Feld muss genau eine Dimension haben.'
        if name:
            feld.name2 = name
        plt.plot(np.linspace(feld.extent()[0], feld.extent()[1], len(feld.matrix)), feld.matrix, label=feld.label)
        if log10plot and ((feld.matrix < 0).sum() == 0) and (feld.matrix.sum() > 0):
            plt.yscale('log')
            #plt.gca().yaxis.set_major_formatter(FuncFormatter(SDFPlots.axesformatexp));
        if feld.axesunits[0] == '':
            plt.xlabel(feld.axesnames[0])
        else:
            plt.xlabel(feld.axesnames[0] + '[' + feld.axesunits[0] + ']')
        plt.autoscale(tight=scaletight)
        if xlim != None:
            plt.xlim(xlim)
        if clim != None:
            plt.ylim(clim)
        if majorgrid:
            plt.grid(b=True, which='major', linestyle='--')
        if saveandclose:
            self.plotspeichern(feld.savename())
            if savecsv:
                feld.exporttocsv(self.lastsavename() + '.csv')

            
    def plotFelder1d(self, *felder, **kwargs):
        kwargs.update({'saveandclose': False})
        zusatz = []
        for feld in felder:
            if feld.dimensions() == 0:
                continue
            zusatz.append(feld.zusatz)
            self._plotFeld1d(feld, **kwargs)
        self.setzetextfeld(feld, zusatz=zusatz)
        plt.legend()
        self.plotspeichern(feld.savename())
        if kwargs.has_key('savecsv') and kwargs['savecsv']:
            for feld in felder:
                if feld.dimensions() == 0:
                    continue
                feld.exporttocsv(self.lastsavename() + feld.label + '.csv')


    #add lineouts to 2D-plots
    @staticmethod
    def _addxlineout(ax0, m, extent, log10=False):
        ax = ax0.twinx()
        l = m.mean(axis=0)        
        if log10:
            l = np.log10(l)
        x = np.linspace(extent[0], extent[1], len(l))
        ax.plot(x,l, 'k', lw=1)
        ax.autoscale(tight=True)
        return ax
    
    @staticmethod
    def _addylineout(ax0, m, extent, log10=False):
        ax = ax0.twiny()
        #ax.set_xlim(ax.get_xlim()[::-1])
        l = m.mean(axis=1)
        if log10:
            l = np.log10(l)
        x = np.linspace(extent[2], extent[3], len(l))
        ax.plot(l,x, 'k', lw=1)
        ax.autoscale(tight=True)
        #ax.spines['top'].set_position(('axes',0.9))
        return ax

    def plotFeld2d(self, feld, log10plot=True, interpolation='none', contourlevels=np.array([]), saveandclose=True, xlim=None, ylim=None, clim=None, scaletight=None, name='', majorgrid=False, savecsv=False, lineoutx=False, lineouty=False):
        assert feld.dimensions() == 2, 'Feld muss genau 2 Dimensionen haben.'
        fig, ax0 = plt.subplots()
        if log10plot & ((feld.matrix < 0).sum() == 0) & (feld.matrix.sum() > 0):
            if feld.grid_nodes_linear() and True:
                plt.imshow(np.log10(feld.matrix.T), origin='lower', aspect='auto', extent=feld.extent(), cmap='jet',interpolation=interpolation) 
            else:
                print 'using pcolormesh'
                x, y = feld.grid()
                plt.pcolormesh(x,y, np.log10(feld.matrix.T), cmap='jet')
            plt.colorbar(format='%3.1f')
            if clim:
                plt.clim(clim)
        else:
            log10plot = False
            plt.imshow(feld.matrix.T, aspect='auto', origin='lower', extent=feld.extent(), cmap='EFeld', interpolation=interpolation)
            if clim:
                plt.clim(clim)
            self.symmetrisiereclim()
            plt.colorbar(format='%6.0e')
        if contourlevels.size != 0: #Einzelne Konturlinie(n) plotten
            plt.contour(feld.matrix.T, contourlevels, hold='on', extent=feld.extent())
        if feld.axesunits[0] == '':
            plt.xlabel(feld.axesnames[0])
        else:
            plt.xlabel(feld.axesnames[0] + '[' + feld.axesunits[0] + ']')
        if feld.axesunits[1] == '':
            plt.ylabel(feld.axesnames[1])
        else:
            plt.ylabel(feld.axesnames[1] + '[' + feld.axesunits[1] + ']')
        plt.autoscale(tight=scaletight)
        if xlim != None:
            plt.xlim(xlim)
        if ylim != None:
            plt.ylim(ylim)
        if name != None:
            feld.name2 = name            
        self.setzetextfeld(feld)
        if majorgrid:
            plt.grid(b=True, which='major', linestyle='--')
        if lineoutx:
            self._addxlineout(ax0, feld.matrix.T, feld.extent(), log10=log10plot)
        if lineouty:
            self._addylineout(ax0, feld.matrix.T, feld.extent(), log10=log10plot)
        if saveandclose:
            self.plotspeichern(feld.savename())
            if savecsv:
                feld.exporttocsv(self.lastsavename() + '.csv')


    def plotFeld(self, feld, **kwargs):
        if feld == None:
            return self._skipplot()
        elif feld.dimensions() == 0:
            return self._skipplot()
        elif feld.dimensions() == 1:
            return self.plotFelder1d(feld, **kwargs)
        elif feld.dimensions() == 2:
            return self.plotFeld2d(feld, **kwargs)
        else:
            raise Exception('plotFeld kann nur 1 oder 2 dimensionale Felder plotten.')

    def _skipplot(self):
        print 'Skipped Plot: ' + self._savename('')
        return
    
        
    def plotFelder(self, *felder, **kwargs):
        for feld in felder:
            self.plotFeld(feld, **kwargs)


    def plotallderived(self, sdfanalyzer):
        fa = sdfanalyzer.getfieldanalyzer()
        felder = fa.createfelderfromkeys(*sdfanalyzer.getderived())
        self.plotFelder(*felder)
        return 0
















