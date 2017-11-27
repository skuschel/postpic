#!/usr/bin/env python
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

def main():
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

    # we will need a refrence to the MultiSpecies quite often
    from postpic import MultiSpecies as MS

    # create MultiSpecies object for every particle species that exists.
    pas = [MS(dr, s) for s in dr.listSpecies()]

    # --- 1D visualization of particle contributions ---

    def particleshapedemo(shape):
        from postpic.particles import histogramdd
        import matplotlib.pyplot as plt
        ptclpos = np.array([4.5, 9.75, 15.0, 20.25])
        y, edges = histogramdd(ptclpos, bins=25, range=(0,25), shape=shape)
        x = np.convolve(edges, [0.5, 0.5], mode='valid')
        fig = plt.figure()
        fig.suptitle('ParticleShape: {:s}'.format(str(shape)))
        ax = fig.add_subplot(111)
        ax.plot(x,y)
        ax.set_ylim((0,1))
        ax.set_xticks(x, minor=True)
        ax.grid(which='minor')
        for ix in ptclpos:
            ax.axvline(x=ix, color='y')
        fig.savefig(savedir + 'particleshapedemo{:s}.png'.format(str(shape)), dpi=160)
        plt.close(fig)

    if True:
        particleshapedemo(0)
        particleshapedemo(1)
        particleshapedemo(2)

    # --- 1D ---
    if True:
            pa = pas[0]
            plotargs = {'ylim': (0,1600), 'log10plot': False}

            # 1 particle per cell
            plotter.plotField(pa.createField('x', optargsh={'bins': 300, 'shape': 0}, title='1ppc_order0', rangex=(0,1)), **plotargs)
            plotter.plotField(pa.createField('x', optargsh={'bins': 300, 'shape': 1}, title='1ppc_order1', rangex=(0,1)), **plotargs)
            plotter.plotField(pa.createField('x', optargsh={'bins': 300, 'shape': 2}, title='1ppc_order2', rangex=(0,1)), **plotargs)

            # 3 particles per cell
            plotter.plotField(pa.createField('x', optargsh={'bins': 100, 'shape': 0}, title='3ppc_order0', rangex=(0,1)), **plotargs)
            plotter.plotField(pa.createField('x', optargsh={'bins': 100, 'shape': 1}, title='3ppc_order1', rangex=(0,1)), **plotargs)
            plotter.plotField(pa.createField('x', optargsh={'bins': 100, 'shape': 2}, title='3ppc_order2', rangex=(0,1)), **plotargs)

            # 10 particles per cell
            plotter.plotField(pa.createField('x', optargsh={'bins': 30, 'shape': 0}, title='10ppc_order0', rangex=(0,1)), **plotargs)
            plotter.plotField(pa.createField('x', optargsh={'bins': 30, 'shape': 1}, title='10ppc_order1', rangex=(0,1)), **plotargs)
            plotter.plotField(pa.createField('x', optargsh={'bins': 30, 'shape': 2}, title='10ppc_order2', rangex=(0,1)), **plotargs)


    # --- 2D ---
    if True:
            dr = pp.readDump(300*30, seed=None, randfunc=np.random.random)
            pa = MS(dr, dr.listSpecies()[0])
            plotargs = {'clim': (0,3e4), 'log10plot': False}

            # 1 particle per cell
            plotter.plotField(pa.createField('x', 'y', optargsh={'bins': (300,30), 'shape': 0}, title='1ppc_order0', rangex=(0,1), rangey=(0,1)), **plotargs)
            plotter.plotField(pa.createField('x', 'y', optargsh={'bins': (300,30), 'shape': 1}, title='1ppc_order1', rangex=(0,1), rangey=(0,1)), **plotargs)
            plotter.plotField(pa.createField('x', 'y', optargsh={'bins': (300,30), 'shape': 2}, title='1ppc_order2', rangex=(0,1), rangey=(0,1)), **plotargs)

            # 3 particles per cell
            plotter.plotField(pa.createField('x', 'y', optargsh={'bins': (100,10), 'shape': 0}, title='3ppc_order0', rangex=(0,1), rangey=(0,1)), **plotargs)
            plotter.plotField(pa.createField('x', 'y', optargsh={'bins': (100,10), 'shape': 1}, title='3ppc_order1', rangex=(0,1), rangey=(0,1)), **plotargs)
            plotter.plotField(pa.createField('x', 'y', optargsh={'bins': (100,10), 'shape': 2}, title='3ppc_order2', rangex=(0,1), rangey=(0,1)), **plotargs)


    # --- 3D ---
    if True:
        dr = pp.readDump(300*30, seed=None, randfunc=np.random.random, dimensions=3)
        pa = MS(dr, dr.listSpecies()[0])
        # just try to create the field. not plotting routines yet
        f = pa.createField('x', 'y', 'z', optargsh={'bins': (30,30,10), 'shape': 2}, title='1ppc_order2', rangex=(0,1), rangey=(0,1), rangez=(0,1))

if __name__=='__main__':
    main()
