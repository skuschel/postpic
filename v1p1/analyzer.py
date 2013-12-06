"""
Die Analyzer Klassen erkennen die Spezies und berechnen aus den gedumpten Daten gewuenschte Groessen wie Energie und Richtung von Teilchen oder Spektren aus Feldern. Achsenbeschriftungen werden ebenfalls hier richtig eingefuegt.
"""


from . import *
import copy
import re
from _Interfaces import *
from _Constants import *
from feld import *

__all__=['ParticleAnalyzer', 'FieldAnalyzer']

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
        

    def __init__(self, sdfanalyzer, species):
        self.species = species
        self.speciesexists = True
        if not sdfanalyzer.hasSpecies(species):
            self.speciesexists = False
            return
        self.sdfanalyzer = sdfanalyzer
        self._weightdata = sdfanalyzer.getSpecies(species, 'weight')
        self._Xdata = sdfanalyzer.getSpecies(species, 'x')
        self._Ydata = np.array([0,1])
        self._Zdata = np.array([0,1])
        self.simdimensions = sdfanalyzer.simdimensions
        if self.simdimensions > 1:
            self._Ydata = sdfanalyzer.getSpecies(species, 'y')
        if self.simdimensions > 2:
            self._Zdata = sdfanalyzer.getSpecies(species,'z')
        self._Pxdata = sdfanalyzer.getSpecies(species,'px')
        self._Pydata = sdfanalyzer.getSpecies(species,'py')
        self._Pzdata = sdfanalyzer.getSpecies(species,'pz')
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
        self.__init__(self.sdfanalyzer, self.species)

        
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
        
    
    def __init__(self, sdfanalyzer, *speciess):
        self._speciess = speciess
        self.species = ''
        self._ssas = []
        self.simdimensions = None
        for s in speciess:
            ssa = _SingleSpeciesAnalyzer(sdfanalyzer,s)
            if ssa.speciesexists:
                self.species = self.species + s
                self._ssas.append(ssa)
                self.simdimensions = self._ssas[0].simdimensions
        self._compresslog = []

        self.simextent = sdfanalyzer.simextent()
        self.simgridpoints = sdfanalyzer.simgridpoints
        self.X.__func__.extent=self.simextent[0:2]
        self.X.__func__.gridpoints=self.simgridpoints[0]
        self.X_um.__func__.extent=self.simextent[0:2] * 1e6
        self.X_um.__func__.gridpoints=self.simgridpoints[0]
        if self.simdimensions > 1:
            self.Y.__func__.extent=self.simextent[2:4]
            self.Y.__func__.gridpoints=self.simgridpoints[1]
            self.Y_um.__func__.extent=self.simextent[2:4] * 1e6
            self.Y_um.__func__.gridpoints=self.simgridpoints[1]
        if self.simdimensions > 2:
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
    

    def __init__(self, sdfanalyzer, lasnm=None):
        self.sdfanalyzer=sdfanalyzer
        self.lasnm=lasnm
        if lasnm:
            self.k0 = 2 * np.pi / (lasnm * 1e-9)
        else:
            self.k0 = None
        self.simdimensions = sdfanalyzer.simdimensions
        self._simextent = sdfanalyzer.simextent()
        self._simgridpoints = sdfanalyzer.simgridpoints
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
        field.setgrid_node(0, self.sdfanalyzer.grid_node('x'))
        if self.simdimensions > 1:
            field.setgrid_node(1, self.sdfanalyzer.grid_node('y'))
        if self.simdimensions > 2:
            field.setgrid_node(2, self.sdfanalyzer.grid_node('z'))
        return None


    # --- Return functions for basic data layer
    
    # -- basic --
    #**kwargs ist z.B. average=True
    def _Ex(self, **kwargs):
        return self.sdfanalyzer.dataE('x', **kwargs)
    def _Ey(self, **kwargs):
        return self.sdfanalyzer.dataE('y', **kwargs)
    def _Ez(self, **kwargs):
        return self.sdfanalyzer.dataE('z', **kwargs)
    def _Bx(self, **kwargs):
        return self.sdfanalyzer.dataB('x', **kwargs)
    def _By(self, **kwargs):
        return self.sdfanalyzer.dataB('y', **kwargs)
    def _Bz(self, **kwargs):
        return self.sdfanalyzer.dataB('z', **kwargs)


    # --- Alle Funktionen geben ein Objekt vom Typ Feld zurueck

    # allgemein ueber dem Ort auftragen. Insbesondere fuer Derived/*
    def createfeldfromkey(self, key):
        ret = Feld(self.sdfanalyzer.data(key));
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

