"""
+-------------------+
|   EPOCHSDFTOOLS   |
+-------------------+

Stephan Kuschel (C) 2013

CLI Interface.
"""

import epochsdftools as ep
import argparse
import numpy as np

print """+------------------------+
|   EPOCHSDFTOOLS  CLI   |
+------------------------+
""" + 'v' + str(ep.__version__)


parser = argparse.ArgumentParser(description='Prints informations about given sdf dump or processes its data.', prog='epochsdftools')
parser.add_argument('inputfile', help='sdf file to process.')
modeparsers = parser.add_subparsers(help='Mode switch')

# Info mode
parser_info = modeparsers.add_parser('info', help='info')
parser_info.set_defaults(mode='info')
parser_info.add_argument('--list-keys', action='store_true', dest='listkeys', help='list keys written to sdf file.')
parser_info.add_argument('-l', action='store_true', dest='listkeys', help='alias for --list-keys')
parser_info.add_argument('keys', help='Show contents of key.', nargs='*')

# Plotting mode
parser_plot = modeparsers.add_parser('plot', help='interactive plotting')
parser_plot.set_defaults(mode='plot')
parser_plot.add_argument('-s', action='store_true', dest='savedata', help='saves data to an appropriately named csv file after plotting.')
modeparsers_plot = parser_plot.add_subparsers(help='Plot Feld or Histogram')
# Plotting mode - Feld
parser_plotfeld = modeparsers_plot.add_parser('feld')
feldchoices = ['Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz', 'energydensityE', 'energydensityM', 'energydensityEM', 'spectrumEx', 'spectrumBz']
parser_plotfeld.add_argument('feld', choices=feldchoices, help='the Feld to plot. Allowed Values are: ' + ', '.join(feldchoices), metavar='Feld')
# Plotting mode - hist
parser_plothist = modeparsers_plot.add_parser('hist')
scalarchoices = ['X', 'X_um', 'Y', 'Y_um', 'Z', 'Z_um', 'Px', 'Py', 'Pz', 'P', 'beta', 'V', 'gamma', 'Ekin', 'Ekin_MeV', 'Ekin_MeV_amu', 'Ekin_MeV_qm', 'Ekin_keV', 'Ekin_keV_amu', 'Ekin_keV_qm', 'angle_xy', 'angle_yz', 'angle_zx', 'angle_offaxis']
parser_plothist.add_argument('species', help='the species used for plotting.')
parser_plothist.add_argument('axes', choices=scalarchoices, help='the axes scalars to use. (order matters!) Allowed Values are: ' + ', '.join(scalarchoices), nargs='+', metavar='scalar')
parser_plothist.add_argument('--weights', choices=scalarchoices, metavar='scalar', help='include additional particle weight factor. Same choices as for axis')

# Parsing
args = parser.parse_args()
print vars(args)


if args.mode == 'info':
    sdfa = ep.SDFAnalyzer(args.inputfile)
    print '--- Header ---'
    print sdfa.header

    print '--- Grid ---'
    fa = sdfa.getfieldanalyzer()
    print 'extent:     ' + str(fa.getextent())
    print 'domainsize: ' + str(fa.getsimdomainsize())
    print 'gridpoints: ' + str(fa.getsimgridpoints())
    print 'resolution: ' + str(fa.getsimdomainsize() / fa.getsimgridpoints())

    print '--- Particles ---'
    info = []
    for s in sdfa.species(ejected='all'):
        pa = sdfa.getparticleanalyzer(s)
        info.append([s, len(pa)])
    print info

    if args.listkeys:
        print('--- List of Keys ---')
        keys = sdfa._data.keys()
        keys.sort()
        print str(keys)

    for key in args.keys:
        try:
            print ('--- Printing key: ' + str(key) + ' ---')
            print str(sdfa._data[key])
            print ('Length of key ' + str(key) + ': ' + str(len(sdfa._data[key])))
        except(KeyError):
            print ('ERROR: Key \'' + str(key) + '\' does not exist!')


def def_onclick(feld=None):
    if feld and feld.dimensions() == 2:
        interpol = feld.interpolater(fill_value=np.nan)
        def onclick(event):
            zdata = interpol(event.xdata, event.ydata)
            print 'button=%d, xdata=%1.4e, ydata=%1.4e, zdata=%1.4e' % (
                event.button, event.xdata, event.ydata, zdata)

    else:
        def onclick(event):
            print 'button=%d, xdata=%1.4e, ydata=%1.4e' % (
                event.button, event.xdata, event.ydata)
    return onclick

if args.mode == 'plot':
    import matplotlib.pyplot as plt
    plt.switch_backend('GTKAgg')
    sdfa = ep.SDFAnalyzer(args.inputfile)
    sdfplots = ep.SDFPlots(sdfa)

    # Feld
    if hasattr(args, 'feld'):
        feld = getattr(sdfa.getfieldanalyzer(), args.feld)()

    # Histogram
    elif hasattr(args, 'axes'):
        pa = sdfa.getparticleanalyzer(args.species)
        kwargs = {}
        if args.weights:
            kwargs = {'weights': getattr(ep.ParticleAnalyzer, args.weights)}
        feld = pa.createFeld(*[getattr(ep.ParticleAnalyzer, x) for x in args.axes], **kwargs)


    # Plot
    fig = sdfplots.plotFeld(feld, format_coord_interactive=True)
    onclick = def_onclick(feld=feld)
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    # show and wait for closing graphics.
    plt.show(block=True)

    if args.savedata:
        feld.exporttocsv(args.inputfile[0:4] + '_' + feld.name + '.csv')
        print ('Data saved to: ' + args.inputfile[0:4] + '_' + feld.name + '.csv')







