"""
Plottet und speichert Felder, die die Analyzer Klassen generiert haben.
"""



from . import *
import matplotlib; matplotlib.use('Agg') #Bilder auch ohne X11 rendern
import matplotlib.pyplot as plt
from _Constants import *


__all__=['SDFPlots']

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
 
    from matplotlib.colors import LinearSegmentedColormap
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
        if log10plot and ((feld.matrix < 0).sum() == 0) and not (feld.matrix.sum() < 0):
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
        if log10plot and ((feld.matrix < 0).sum() == 0) and not (feld.matrix.sum() < 0):
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




