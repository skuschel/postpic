
from .functions import replacements
from packaging.version import parse as parse_version
import numpy
from numpy.lib.mixins import NDArrayOperatorsMixin

__all__ = []
for repl in replacements:
    if parse_version(repl.lib.__version__) < parse_version(repl.minver):
        vars()[repl.name] = repl.replacement
    else:
        vars()[repl.name] = getattr(repl.originalmodule, repl.name)

    __all__.append(repl.name)

__all__.append('NDArrayOperatorsMixin')
