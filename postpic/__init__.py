"""
+--------------+
|   POSTPIC    |
+--------------+

The open source particle-in-cell post processor.
"""

import datareader
import analyzer
import plotting

__version__ = '0.0.0'

# Use Git description for __version__ if present
try:
    import subprocess as sub
    import os.path
    cwd = os.path.dirname(__file__)
    p = sub.Popen(['git', 'describe', '--always'], stdout=sub.PIPE,
                  stderr=sub.PIPE, cwd=cwd)
    out, err = p.communicate()
    if not p.returncode:  # git exited without error
        __version__ += '_g' + out
except OSError:
    # 'git' command not found
    pass


