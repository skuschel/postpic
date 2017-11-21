
from .functions import replacements
import pkg_resources as pr

__all__ = []
for repl in replacements:
    if pr.parse_version(repl.lib.__version__) < pr.parse_version(repl.minver):
        vars()[repl.name] = repl.replacement
    else:
        vars()[repl.name] = getattr(repl.originalmodule, repl.name)

    __all__.append(repl.name)

