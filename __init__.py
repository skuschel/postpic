"""
+-------------------+
|   EPOCHSDFTOOLS   |
+-------------------+

Stephan Kuschel (C) 2013
"""

__version__ = '1.1.0'


# import order matters: feld must be imported before analyzer, since the analyzer module uses the Feld class of the feld module.
from feld import *
from sdfplots import *
from analyzer import *
from sdfdatareader import *


__all__ = ['Feld', 'SDFAnalyzer', 'SDFPlots', 'FieldAnalyzer', 'ParticleAnalyzer', 'OutputAnalyzer']










