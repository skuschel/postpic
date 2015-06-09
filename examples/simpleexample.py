#!/usr/bin/env python2
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
# Copyright Stephan Kuschel 2015
#

import numpy as np
import postpic as pp

# postpic will use matplotlib for plotting. Changing matplotlibs backend
# to "Agg" makes it possible to save plots without a display attached.
# This is necessary to run this example within the "run-tests" script
# on travis-ci.
import matplotlib; matplotlib.use('Agg')


# choose the dummy reader. This reader will create fake data for testing.
pp.chooseCode('dummy')

dr = pp.readDump(3e5)  # Dummyreader takes a float as argument, not a string.
# set and create directory for pictures.
savedir = '_examplepictures/'
import os
if not os.path.exists(savedir):
    os.mkdir(savedir)

# initialze the plotter object.
# project name will be prepended to all output names
plotter = pp.plotting.plottercls(dr, outdir=savedir, autosave=True, project='simpleexample')

# we will need a refrence to the MultiSpecies quite often
from postpic import MultiSpecies as MS

# create MultiSpecies Object for every particle species that exists.
pas = [MS(dr, s) for s in dr.listSpecies()]

if True:
    # Plot Data from the FieldAnalyzer fa. This is very simple: every line creates one plot
    plotter.plotField(dr.Ex())  # plot 0
    plotter.plotField(dr.Ey())  # plot 1
    plotter.plotField(dr.Ez())  # plot 2
    plotter.plotField(dr.energydensityEM())  # plot 3

    # Using the MultiSpecies requires an additional step:
    # 1) The MultiSpecies.createField method will be used to create a Field object
    # with choosen particle scalars on every axis
    # 2) Plot the Field object
    optargsh={'bins': [300,300]}
    for pa in pas:
        # create a Field object nd holding the number density
        nd = pa.createField(MS.X, MS.Y, optargsh=optargsh,simextent=True)
        # plot the Field object nd
        plotter.plotField(nd, name='NumberDensity')   # plot 4
        # more advanced: create a field holding the total kinetic energy on grid
        ekin = pa.createField(MS.X, MS.Y, weights=MS.Ekin_MeV, optargsh=optargsh, simextent=True)
        # The Field objectes can be used for calculations. Here we use this to
        # calculate the average kinetic energy on grid and plot
        plotter.plotField(ekin / nd, name='Avg Kin Energy (MeV)')  # plot 5

        # use optargsh to force lower resolution
        # plot number density
        plotter.plotField(pa.createField(MS.X, MS.Y, optargsh=optargsh), lineoutx=True, lineouty=True)  # plot 6
        # plot phase space
        plotter.plotField(pa.createField(MS.X, MS.P, optargsh=optargsh))  # plot 7
        plotter.plotField(pa.createField(MS.X, MS.gamma, optargsh=optargsh))  # plot 8
        plotter.plotField(pa.createField(MS.X, MS.beta, optargsh=optargsh))  # plot 9

        # same with high resolution
        plotter.plotField(pa.createField(MS.X, MS.Y, optargsh={'bins': [1000,1000]}))  # plot 10
        plotter.plotField(pa.createField(MS.X, MS.P, optargsh={'bins': [1000,1000]}))  # plot 11

        # advanced: postpic has already defined a lot of particle scalars as Px, Py, Pz, P, X, Y, Z, gamma, beta, Ekin, Ekin_MeV, Ekin_MeV_amu, ... but if needed you can also define your own particle scalar on the fly.
        # In case its regularly used it should be added to postpic. If you dont know how, just let us know about your own useful particle scalar by email or adding an issue at
        # https://github.com/skuschel/postpic/issues

        # define your own particle scalar: p_r = sqrt(px**2 + py**2)/p
        def p_r(ms):
            return np.sqrt(ms.Px()**2 + ms.Py()**2) / ms.P()
        # add unit and name for automatic labeling when plotted with plotField method
        p_r.unit=''
        p_r.name='$\sqrt{P_x^2 + P_y^2} / P$'
        # define another own particle scalar: r = sqrt(x**2 + y**2)
        def r(ms):
            return np.sqrt(ms.X()**2 + ms.Y()**2)
        r.unit='m'
        r.name='r'
        # use the plotter with the particle scalars defined above.
        plotter.plotField(pa.createField(r, p_r, optargsh={'bins':[400,400]}))  # plot 12

        # choose particles by their properies
        def cf(ms):
            return ms.X() > 0.0  # only use particles with x > 0.0
        cf.name = 'x>0'
        pa.compressfn(cf)
        # plot 13, compare with plot 10
        plotter.plotField(pa.createField(MS.X, MS.Y, optargsh={'bins': [1000,1000]}))
        # plot 14, compare with plot 12
        plotter.plotField(pa.createField(r, p_r, optargsh={'bins':[400,400]}))

