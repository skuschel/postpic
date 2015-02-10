#!/usr/bin/env python2
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
# Copyright Stephan Kuschel, 2014-2015

# run all tests and pep8 verification of this project.
# It is HIGHLY RECOMMENDED to link it as a git pre-commit hook!
# Please see pre-commit for instructions.

# THIS FILE MUST RUN WITHOUT ERROR ON EVERY COMMIT!

import os
import subprocess


def exitonfailure(exitstatus, cmd=None):
    if exitstatus == 0:
        print('OK')
    else:
        print('run-tests.py failed. aborting.')
        if cmd is not None:
            print('The failing command was:')
            print(cmd)
        exit(exitstatus)


def main():
    # run nose tests
    import nose
    ex = nose.run()  # returns True on success
    exitonfailure(not ex, cmd='nosetests')

    cmds = ['pep8 postpic --statistics --count --show-source '
            '--ignore=W391,E123,E226,E24 --max-line-length=99',
            os.path.join('examples', 'simpleexample.py')]
    for cmd in cmds:
        print('=====  running next command =====')
        print(cmd)
        exitonfailure(subprocess.call(cmd, shell=True), cmd=cmd)


if __name__ == '__main__':
    main()
