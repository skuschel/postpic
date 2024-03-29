#!/usr/bin/env python
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

import sys
import os
from packaging.version import parse as parse_version


def runcmd(cmd):
    '''
    run command cmd and exit if it fails.
    '''
    import subprocess
    print('=====  running next command =====')
    print('$ ' + cmd)
    exitstatus = subprocess.call(cmd, shell=True)
    if exitstatus == 0:
        print('OK')
    else:
        print('run-tests.py failed. aborting.')
        print('The failing command was:')
        print(cmd)
        exit(exitstatus)

def run_autopep8(args):
        import autopep8
        if parse_version(autopep8.__version__) < parse_version('1.2'):
            print('upgrade to autopep8 >= 1.2 (installed: {:})'.format(str(autopep8.__version__)))
            exit(1)
        autopep8mode = '--in-place' if args.autopep8 == 'fix' else '--diff'
        argv = ['autopep8', '-r', 'postpic', '--ignore-local-config', autopep8mode, \
                '--ignore=W391,E123,E226,E24' ,'--max-line-length=99']
        print('===== running autopep8 =====')
        print('autopep8 version: ' + autopep8.__version__)
        print('$ ' + ' '.join(argv))
        autopep8.main(argv)

def run_alltests(python='python', fast=False, skip_setup=False):
    '''
    runs all tests on postpic. This function has to exit without error on every commit!
    '''
    cmdrpl = dict(python=python)
    # make sure .pyx sources are up to date and compiled
    if not skip_setup:
        # should be the same as `./setup.py develop --user`
        runcmd('{python} -m pip install -e .'.format(**cmdrpl))

    # find pep8 or pycodestyle (its successor)
    try:
        import pep8
        cmdrpl['pycodestyle'] = 'pep8'
    except(ImportError):
        pass
    try:
        import pycodestyle
        cmdrpl['pycodestyle'] = 'pycodestyle'
    except(ImportError):
        pass
    if 'pycodestyle' not in cmdrpl:
        raise ImportError('Install pep8 or pycodestyle (its successor)')

    cmds = ['{python} -m pycodestyle --version',
            '{python} -m {pycodestyle} postpic --statistics --count --show-source '
            '--ignore=W391,E123,E226,E24,W504 --max-line-length=99',
            '{python} -m nose2 -B']
    cmdo = ['make -C doc/ html',
            '{python} ' + os.path.join('examples', 'simpleexample.py'),
            '{python} ' + os.path.join('examples', 'particleshapedemo.py'),
            '{python} ' + os.path.join('examples', 'time_cythonfunctions.py'),
            '{python} ' + os.path.join('examples', 'openPMD.py'),
            '{python} ' + os.path.join('examples', 'kspace-test-2d.py')]
    if not fast:
        cmds += cmdo
    for cmd in cmds:
        runcmd(cmd.format(**cmdrpl))

def main():
    import argparse
    parser = argparse.ArgumentParser(description='''
        Without arguments this runs all tests
        on the postpic codebase.
        This script MUST EXIT WITHOUT ERROR on EVERY commit!
        ''')
    parser.add_argument('--autopep8', default='', nargs='?', metavar='fix',
                        help='''
        run "autopep8" on the codebase.
        Use "--autopep8" to preview changes. To apply them, use
        "--autopep8 fix"
        ''')
    parser.add_argument('--fast', action='store_true', default=False,
                        help='Only run a subset of tests. Used for commit hook.')
    parser.add_argument('--skip-setup', action='store_true', default=False,
                        help='Skip the "setup.py develop --user" step.')
    pyversiongroup = parser.add_mutually_exclusive_group(required=False)
    pyversiongroup.add_argument('--pycmd', default='python{}'.format(sys.version_info.major),
                                help='use "PYCMD" as python interpreter for all subcommands. '
                                'default: "pythonX" where X is sys.version_info.major. '
                                'This means that by default the same version is used that '
                                'executes this script.'
                               )
    pyversiongroup.add_argument('-2', action='store_const', dest='pycmd', const='python2',
                                help='same as "--pycmd python2"')
    pyversiongroup.add_argument('-3', action='store_const', dest='pycmd', const='python3',
                                help='same as "--pycmd python3"')
    args = parser.parse_args()

    if args.autopep8 != '':
        run_autopep8(args)
        exit()

    run_alltests(args.pycmd, fast=args.fast, skip_setup=args.skip_setup)

if __name__ == '__main__':
    main()
