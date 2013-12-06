"""
Interfaces aus denen Klassen, die besondere Aufgaben erfuellen koennen sollen abgeleitet werden koennen.
"""



__all__=['PlotDescriptor']


class PlotDescriptor():
    """
    Stellt Allgemeine Informationen zum Beschriften der Plots bereit. Wird von Allen Analyzer Klassen Implementiert.
    """

    def __init__(self):
        pass
    
    def getprojektname(self):
        raise NotImplementedError
        
    def getprojektname2(self):
        """
        genauere Beschreibung des Projekts, z.B. dumpname
        """
        raise NotImplementedError
        
    def getprojektname3(self):
        """
        noch genauere Beschreibung des Projekts, z.B. Zeit
        """
        raise NotImplementedError
    
    def getprojektname4(self):
        """
        z.B. Zeitschritt
        """
        raise NotImplementedError

    def getlasnm(self):
        raise NotImplementedError

