"""
+-------------------+
|   EPOCHSDFTOOLS   |
+-------------------+

Stephan Kuschel, 20130525
last update: 130911


UPDATE HISTORY:
---------------------
131206: Aufteilung in mehrere Dateien.
131108: Bumped to v1.1.0: Plot Descriptor hinzugefuegt, inkompatibel mit vorhergehenden Versionen, weil SDFPlots.sdfanalyzer nicht mehr existiert.
131107: OutputAnalyzer hinzugefuegt
131106: lineoutx und lineouty als Option zum 2D Feld Plot hinzugefuegt. Ekennung fuer "ejected_" Particles hinzugefuegt. 
130911: createHistogram2D rangex und rangy hinzugefuegt.
130813: SDFPlots.funcformatter geupdated.
130726: CSV export fuer 1d Felder hinzugefuegt. Kann direkt beim plot mit savecsv=True aktiviert werden. Volumenelement bei angleoffaxis entfernt.
130712: ein haufen Bugfixes.
130702: ParticleAnalyzer benutzt jetzt intern den SingleSpeciesAnalyzer. ParticleAnalyzer Objekte koennen nun addiert werden um unterschiedliche Sets von Teilchen zu erstellen. SingleSpeciesAnalyzer erkennt nun Masse und Ladung der Teilchen, SingleSpeciesAnalyzer.getmass wurde ersetzt durch SingleSpeciesAnalyzer.retrieveparticleinfo. ParticleAnalyzer nimmt nun auch mehrere Spezies direkt entgegen. Elektronen werden erkannt und der ParticleAnalyzer weiss, ob ein Teilchen ein ion ist oder nicht.
130701: Viele tools zum Plotten und erstellen der Histogramme hinzugefuegt. Feld-Klasse hinzugefuegt um Datenformat zu vereinheitlichen.
130628: numpy als np importiert. sdfanalyzer hinzugefuegt.
130619: FieldAnalyzer begonnen
130618: Konversion zu float64 ergaenzt. Verhindert Rechenfehler, falls Daten nur mit single precision geschrieben wurden.
130610: erkennt automatisch die Anzahl der Dimensionen in der Simulation, sowie ionenspezies vom typ ionc22m70
"""

__version__='1.1.0'


import numpy as np



from analyzer import *
from feld import *
from sdfdatareader import *
from sdfplots import *

__all__=['Feld', 'SDFAnalyzer', 'SDFPlots', 'FieldAnalyzer', 'ParticleAnalyzer', 'OutputAnalyzer']










