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

'''
This is a demonstration file to show the differences between various particles
shapes used.
'''

import numpy as np
import postpic as pp

# postpic will use matplotlib for plotting. Changing matplotlibs backend
# to "Agg" makes it possible to save plots without a display attached.
# This is necessary to run this example within the "run-tests" script
# on travis-ci.
import matplotlib; matplotlib.use('Agg')


# choose the dummy reader. This reader will create fake data for testing.
pp.chooseCode('dummy')

# Create a dummy reader with 300 particles, not initialized with a seed and use
# uniform distribution
dr = pp.readDump(300, seed=None, randfunc=np.random.random)
# set and create directory for pictures.
savedir = '_examplepictures/'
import os
if not os.path.exists(savedir):
    os.mkdir(savedir)

# initialze the plotter object.
# project name will be prepended to all output names
plotter = pp.plotting.plottercls(dr, outdir=savedir, autosave=True, project='particleshapedemo')

# we will need a refrence to the ParticleAnalyzer quite often
from postpic import ParticleAnalyzer as PA

# create ParticleAnalyzer for every particle species that exists.
pas = [PA(dr, s) for s in dr.listSpecies()]


# --- 1D ---
if True:
        pa = pas[0]
        plotargs = {'ylim': (0,1600), 'log10plot': False}

        # 1 particle per cell
        plotter.plotField(pa.createField(PA.X, optargsh={'bins': 300, 'order': 0}, title='1ppc_order0', rangex=(0,1)), **plotargs)
        plotter.plotField(pa.createField(PA.X, optargsh={'bins': 300, 'order': 1}, title='1ppc_order1', rangex=(0,1)), **plotargs)

        # 3 particles per cell
        plotter.plotField(pa.createField(PA.X, optargsh={'bins': 100, 'order': 0}, title='3ppc_order0', rangex=(0,1)), **plotargs)
        plotter.plotField(pa.createField(PA.X, optargsh={'bins': 100, 'order': 1}, title='3ppc_order1', rangex=(0,1)), **plotargs)

        # 10 particles per cell
        plotter.plotField(pa.createField(PA.X, optargsh={'bins': 30, 'order': 0}, title='10ppc_order0', rangex=(0,1)), **plotargs)
        plotter.plotField(pa.createField(PA.X, optargsh={'bins': 30, 'order': 1}, title='10ppc_order1', rangex=(0,1)), **plotargs)




