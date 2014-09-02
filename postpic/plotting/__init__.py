'''
The plot subpackage should provide an interface to various plot backends.
'''

plottercls = None


def use(plotcls):
    global plottercls
    if isinstance(plotcls, str):
        if plotcls in ['matplotlib', 'plotter_matplotlib']:
            import plotter_matplotlib
            plottercls = plotter_matplotlib.MatplotlibPlotter
        else:
            raise NameError('unknown type {:s}'.format(plotcls))
    else:
        plottercls = plotcls

# Default
use('matplotlib')
