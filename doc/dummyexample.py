#!/usr/bin/env python2

import numpy as np
import postpic as pp


pp.datareader.chooseCode('dummy')

dr = pp.datareader.readDump(3e5)  # Dummyreader takes a float as argument, not a string.
plotter = pp.plotting.plottercls(dr, outdir='', autosave=True)
fa = pp.analyzer.FieldAnalyzer(dr)

from postpic.analyzer import ParticleAnalyzer as PA

pas = []
for s in dr.listSpecies():
    pas.append(PA(dr, s))

if True:
    plotter.plotField(fa.Ex())
    plotter.plotField(fa.Ey())
    plotter.plotField(fa.Ez())
    plotter.plotField(fa.energydensityEM())

    #Number Density and related
    optargsh={'bins': [300,300]}
    for pa in pas:
        #include full simgrid
        nd = pa.createField(PA.X, PA.Y, optargsh=optargsh,simextent=True)
        ekin = pa.createField(PA.X, PA.Y, weights=PA.Ekin_MeV, optargsh=optargsh, simextent=True)
        plotter.plotField(nd, name='NumberDensity')
        #plotter.plotFeld(ekin, name='Kin Energy (MeV)')
        plotter.plotField(ekin / nd, name='Avg Kin Energy (MeV)')
        
        #lower resolution
        plotter.plotField(pa.createField(PA.X, PA.Y, optargsh=optargsh))
        plotter.plotField(pa.createField(PA.X, PA.P, optargsh=optargsh))
        
        #high resolution
        plotter.plotField(pa.createField(PA.X, PA.Y, optargsh={'bins': [1000,1000]}))
        plotter.plotField(pa.createField(PA.X, PA.P, optargsh={'bins': [1000,1000]}))

        def p_r(pa):
            return np.sqrt(pa.Px()**2 + pa.Py()**2) / pa.P()
        p_r.unit=''
        p_r.name='$\sqrt{P_x^2 + P_y^2} / P$'
        def r(pa):
            return np.sqrt(pa.X()**2 + pa.Y()**2)
        r.unit='$m$'
        r.name='r'
        plotter.plotField(pa.createField(r, p_r, optargsh={'bins':[100,200]}))


