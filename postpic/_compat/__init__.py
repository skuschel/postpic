
from .functions import replacements
import pkg_resources as pr
import numpy

__all__ = []
for repl in replacements:
    if pr.parse_version(repl.lib.__version__) < pr.parse_version(repl.minver):
        vars()[repl.name] = repl.replacement
    else:
        vars()[repl.name] = getattr(repl.originalmodule, repl.name)

    __all__.append(repl.name)

if pr.parse_version(numpy.__version__) < pr.parse_version('1.13'):
    from .mixins import NDArrayOperatorsMixin
else:
    from numpy.lib.mixins import NDArrayOperatorsMixin

__all__.append('NDArrayOperatorsMixin')
