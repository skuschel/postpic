#!/bin/bash
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

# just get the directory that this file resides within and load the common variables
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
. "$DIR/common.sh"

rm -rf $SOURCETMP
cp -al $SOURCE $SOURCETMP

pushd $SOURCETMP > /dev/null
    git clean -Xdf
popd > /dev/null

# $SOURCETMP is now the source to be copied for all the envs

PIDS=()

# Start all the tests in the background, remember the PIDs
for ENV in ${ENVS[@]}
do
    "$DIR/do_tests_env.sh" "$ENV" "$@" &
    PIDS+=($!)
done

# Get the return statuses of the PIDs one by one
for pid in ${PIDS[@]}
do
    wait $pid
    STATUS=$?
    if [ $STATUS -ne 0 ]; then
        # One of the tests failed. Wait for all the other running tests
        # and return the failed status
        wait
        exit $STATUS
    fi
done

# Apparently all tests succeded
exit 0
