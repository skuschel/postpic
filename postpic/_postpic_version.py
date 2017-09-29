#!/usr/bin/env python

# Copyright (C) 2016 Stephan Kuschel
#
# This file is part of postpic.
#
# postpic is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# postpic is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with postpic. If not, see <http://www.gnu.org/licenses/>.
#


from __future__ import absolute_import, division, print_function, unicode_literals


__version__ = '0.3'

try:
    FileNotFoundError  # python3
except(NameError):
    FileNotFoundError = IOError  # python2


def _gitversion():
    '''
    returns the git version string if the source in inside a git repo.
    returns None otherwise.
    '''
    ret = None
    try:
        import subprocess as sub
        import os.path
        cwd = os.path.dirname(__file__)
        p = sub.Popen(['git', 'describe', '--always', '--long', '--dirty'], stdout=sub.PIPE,
                      stderr=sub.PIPE, cwd=cwd)
        out, err = p.communicate()
        if not p.returncode:  # git ran without error
            ret = out.decode().strip('\n')
    except OSError:
        # 'git' command not found
        pass
    return ret


def _version(gitversion):
    '''
    returns the version as readable for setup.py.
    For example: v0.2.2, v0.1.7-65
    '''
    ret = gitversion.strip('-dirty')
    ret = ret[:-9]  # remove the SHA
    ret = ret.strip('-0')
    return ret


def getversion_setup():
    # search for the current version with git
    gitversion = _gitversion()
    # if found, write it to _version.txt and postpic/_version.txt
    if gitversion is not None:
        version = _version(gitversion)
        with open('_version.txt', 'w') as fs, open('postpic/_version.txt', 'w') as fp:
            for f in [fs, fp]:
                f.write(gitversion + '\n')
                f.write(version + '\n')
    # if not found, try to read from _version.txt
    else:
        try:
            with open('_version.txt', 'r') as f:
                gitversion = f.readline()
                version = f.readline()
        except(FileNotFoundError):
            # this can happen, if the code is downloaded
            # from github directly as zip or tar
            version = __version__
            gitversion = 'unknown'
    return gitversion, version


def getversion_package():
    # search for the current version with git
    gitversion = _gitversion()
    # if found, write it to _version.txt and postpic/_version.txt
    if gitversion is not None:
        version = _version(gitversion)
    # if not found, try to read from _version.txt
    else:
        try:
            import pkgutil
            data = pkgutil.get_data('postpic', '_version.txt')
            gitversion, version = data.decode().splitlines()
        except(FileNotFoundError):
            # this can happen, if the code is downloaded
            # from github directly as zip or tar
            version = __version__
            gitversion = 'unknown'
    return gitversion, version
