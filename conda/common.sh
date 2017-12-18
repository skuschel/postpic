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
# Copyright Alexander Blinne 2017

# just get the directory that this file resides within
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# configure these to your needs, defaults should work fine though

# this is the base directory used to store the conda environments, copies of the source-tree and the log files
CONDA_BASE=$DIR/../../conda
CONDA_ENVS=$CONDA_BASE/envs
LOGDIR=$CONDA_BASE/log
TMPDIR=$CONDA_BASE/tmp
mkdir -p "$CONDA_BASE" "$CONDA_ENVS" "$LOGDIR" "$TMPDIR" # just make sure the directories exist

# this is the path to the source that should be tested.
SOURCE="$(dirname "$DIR")" # assume the parent dir of this dir

# this is just the name of the source dir, usually postpic or postpic.git
SOURCEDIRNAME="$(basename "$SOURCE")"

# directory to hold a copy of the source which will be created and maintained by run_tests.sh and then used by do_tests_env.sh
SOURCETMP=$TMPDIR/$SOURCEDIRNAME

# this is the list of all environments specified in create_environment.sh
ENVS=(ubuntu1404 ubuntu1604py2 ubuntu1604py3 default2 default3)
