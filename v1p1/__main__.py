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
parser.add_argument('inputfile', help='sdf file to process.')
parser.add_argument('--list-keys', action='store_true', dest='listkeys',help='list keys writtien to sdf file.')

args=parser.parse_args()

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


