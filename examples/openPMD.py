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
# Copyright Stephan Kuschel 2016
#

def download(url, file):
    import urllib3
    import shutil
    import os
    if os.path.isfile(file):
        return
    urllib3.disable_warnings()
    http = urllib3.PoolManager()
    print('downloading {:} ...'.format(file))
    with http.request('GET', url, preload_content=False) as r, open(file, 'wb') as out_file:
        shutil.copyfileobj(r, out_file)

def main():
    import os
    if not os.path.exists('examples/_openPMDdata'):
        os.mkdir('examples/_openPMDdata')
    download('https://github.com/openPMD/openPMD-example-datasets/'
           + 'raw/776ae3a96c02b20cfae56efafcbda6ca76d4c78d/example-2d.tar.gz',
             'examples/_openPMDdata/example-2d.tar.gz')

    import tarfile
    tar = tarfile.open('examples/_openPMDdata/example-2d.tar.gz')
    tar.extractall('examples/_openPMDdata')


    # now that files are downloaded and extracted, start data evaluation

    import numpy as np
    import postpic as pp

    # postpic will use matplotlib for plotting. Changing matplotlibs backend
    # to "Agg" makes it possible to save plots without a display attached.
    # This is necessary to run this example within the "run-tests" script
    # on travis-ci.
    import matplotlib; matplotlib.use('Agg')
    pp.chooseCode('openpmd')
    dr = pp.readDump('examples/_openPMDdata/example-2d/hdf5/data00000300.h5')
    print('The simulations was running on {} spatial dimensions.'.format(dr.simdimensions()))
    # set and create directory for pictures.
    savedir = '_examplepictures/openPMD/'
    import os
    if not os.path.exists(savedir):
        os.mkdir(savedir)

    # initialze the plotter object.
    # project name will be prepended to all output names
    import postpic.plotting
    plotter = pp.plotting.plottercls(dr, outdir=savedir, autosave=True, project='openPMD')

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

        # plot additional, derived fields if available in `dr.getderived()`
        #plotter.plotField(dr.createfieldfromkey("fields/e_chargeDensity"))

        # Using the MultiSpecies requires an additional step:
        # 1) The MultiSpecies.createField method will be used to create a Field object
        # with choosen particle scalars on every axis
        # 2) Plot the Field object
        optargsh={'bins': [200,50]}
        for pa in pas:
            # Remark on 2D or 3D Simulation:
            # the data fields for the plots in the following section are build by postpic
            # only from the particle data. The following plots would just the same,
            # if we would use the data of a 3D Simulation instead.

            # create a Field object nd holding the number density
            nd = pa.createField('z', 'x', optargsh=optargsh, simextent=False)
            # plot the Field object nd
            plotter.plotField(nd, name='NumberDensity')   # plot 4

            # create a Field object nd holding the charge density
            qd = pa.createField('z', 'x', weights='charge',
                                optargsh=optargsh, simextent=False)
            # plot the Field object qd
            plotter.plotField(qd, name='ChargeDensity')   # plot 5

            # more advanced: create a field holding the total kinetic energy on grid
            ekin = pa.createField('z', 'x', weights='Ekin_MeV', optargsh=optargsh, simextent=False)
            # The Field objectes can be used for calculations. Here we use this to
            # calculate the average kinetic energy on grid and plot
            plotter.plotField(ekin / nd, name='Avg Kin Energy (MeV)')  # plot 6

            # use optargsh to force lower resolution
            # plot number density
            plotter.plotField(pa.createField('z', 'x', optargsh=optargsh), lineoutx=True, lineouty=True)  # plot 7
            # plot phase space
            plotter.plotField(pa.createField('z', 'p', optargsh=optargsh))  # plot 8
            plotter.plotField(pa.createField('z', 'gamma', optargsh=optargsh))  # plot 9
            plotter.plotField(pa.createField('z', 'beta', optargsh=optargsh))  # plot 10


    if True:
        # instead of reading a single dump read a collection of dumps using the simulationreader
        sr = pp.readSim('examples/_openPMDdata/example-2d/hdf5/*.h5')
        print('There are {:} dumps in this simulationreader object:'.format(len(sr)))
        # now you can iterate over the dumps written easily
        for dr in sr:
            print('Simulation time of current dump t = {:.2e} s'.format(dr.time()))
            # there are the frames of a movie:
            plotter.plotField(dr.Ez())

if __name__=='__main__':
    # the openPMD reader needs h5py to be installed.
    # in case its not skip the tests
    try:
        import h5py
    except ImportError as ie:
        print('dependency missing: ' + str(ie))
        print('SKIPPING examples/openPMD.py')
        exit(0)
    # beeing here means, that h5py is available.
    # any other exception should still cause the execution to fail!
    main()
