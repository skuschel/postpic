#!/bin/bash

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
