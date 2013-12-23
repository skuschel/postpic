"""
+-------------------+
|   EPOCHSDFTOOLS   |
+-------------------+

Stephan Kuschel (C) 2013

CLI Interface.
"""

import epochsdftools.v1p1 as ep
import argparse

print """+------------------------+
|   EPOCHSDFTOOLS  CLI   |
+------------------------+
""" + 'v' + str(ep.__version__)


parser = argparse.ArgumentParser(description='Prints informations about given sdf dump.', prog='epochsdftools')
modeparsers = parser.add_subparsers(help='Mode switch')

#Info mode
parser_info = modeparsers.add_parser('info', help='info')
parser_info.set_defaults(mode='info')
parser_info.add_argument('inputfile', help='sdf file to process.')
parser_info.add_argument('--list-keys', action='store_true', dest='listkeys',help='list keys writtien to sdf file.')

#Plotting mode
parser_plot = modeparsers.add_parser('plot', help='interactive plotting')
parser_plot.set_defaults(mode='plot')
modeparsers_plot = parser_plot.add_subparsers(help='Plot Feld or Histogram')
#Plotting mode - Feld
parser_plotfeld = modeparsers_plot.add_parser('feld')
feldchoices = ['Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz', 'energydensityE', 'energydensityM', 'energydensityEM', 'spectrumEx', 'spectrumBz']
parser_plotfeld.add_argument('feld', choices=feldchoices, help='the Feld to plot. Allowed Values are: '+', '.join(feldchoices), metavar='Feld')
#Plotting mode - hist
parser_plothist = modeparsers_plot.add_parser('hist')
scalarchoices = ['X', 'X_um', 'Y', 'Y_um', 'Z', 'Z_um', 'Px', 'Py', 'Pz', 'beta', 'V', 'gamma', 'Ekin', 'Ekin_MeV', 'Ekin_MeV_amu', 'Ekin_MeV_qm', 'Ekin_keV', 'Ekin_keV_amu', 'Ekin_keV_qm', 'angle_xy', 'angle_yz', 'angle_zx', 'angle_offaxis']
parser_plothist.add_argument('axes', choices=scalarchoices, help='the axes scalars to use. (order matters!) Allowed Values are: '+', '.join(scalarchoices), nargs='+', metavar='scalar')
parser_plothist.add_argument('--weight', choices=scalarchoices, metavar='scalar', help='include additional particle weight factor. Same choices as for axis')

#Parsing
args=parser.parse_args()

print vars(args)


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


