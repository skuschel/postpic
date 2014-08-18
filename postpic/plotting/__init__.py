'''
The plot subpackage should provide an interface to different plot backends.
'''


plotobject = None


def setplotobj(plotobj):
    global plotobject
    plotobject = plotobj
