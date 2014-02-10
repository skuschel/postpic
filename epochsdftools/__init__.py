"""
+-------------------+
|   EPOCHSDFTOOLS   |
+-------------------+

Stephan Kuschel (C) 2013
"""

__version__ = '1.2.0'

# Use Git description for __version__ if present
try:
    import subprocess as sub
    import os.path
    cwd = os.path.dirname(__file__)
    p = sub.Popen(['git', 'describe', '--always'], stdout=sub.PIPE, stderr=sub.PIPE, cwd=cwd)
    out, err = p.communicate()
    if not p.returncode:  # git exited without error
        __version__ += '_' + out
except OSError:
    # 'git' command not found
    pass


# import order matters: feld must be imported before analyzer, since the analyzer module uses the Feld class of the feld module.
from feld import *
from sdfplots import *
from analyzer import *
from sdfdatareader import *


__all__ = ['Feld', 'SDFAnalyzer', 'SDFPlots', 'FieldAnalyzer', 'ParticleAnalyzer', 'OutputAnalyzer']










