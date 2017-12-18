#!/bin/bash

# just get the directory that this file resides within and load the common variables
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
. "$DIR/common.sh"

ENV="$1"
shift

#make sure environment is clean
echo "Prepping $ENV..."
rm -rf "${CONDA_ENVS}/${ENV}"
cp -al "${CONDA_ENVS}/${ENV}_clean" "${CONDA_ENVS}/${ENV}"


#make copy of source to tmp dir
SRCDIR="$TMPDIR/$ENV"
rm -rf "$SRCDIR/$SOURCEDIRNAME"
mkdir -p "$SRCDIR"
cp -al "$SOURCETMP" "$SRCDIR/$SOURCEDIRNAME"

#go to source dir and activate environment
pushd "$SRCDIR/$SOURCEDIRNAME" > /dev/null
source activate "$CONDA_ENVS/$ENV"

echo "Starting test for $ENV..."

# simple redirection mixes stdout and stderr in a non-ordered way, very unreadable!
# ./setup.py develop >$LOGDIR/$ENV.log 2>&1
# ./run-tests.py "$@" >$LOGDIR/$ENV.log 2>&1

# script leaves in vt escape codes. a little bad but ok
script -c "./setup.py develop" "$LOGDIR/$ENV.log" >/dev/null
script -ae -c "./run-tests.py --skip-setup $*" "$LOGDIR/$ENV.log" >/dev/null

RESULT=$?

source deactivate
popd > /dev/null
echo "Test for $ENV finished with exit code $RESULT"

exit $RESULT
