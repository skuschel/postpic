#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
. "$DIR/common.sh"

for ENV in ${ENVS[@]}; do
    echo Create environment "$CONDA_ENVS/$ENV"
    #script -c "$CONDA_BASE/create_environment.sh $ENV" $LOGDIR/create_$ENV >/dev/null &
    "$DIR/create_environment.sh" $ENV >"$LOGDIR/create_$ENV" 2>&1 &
done

wait
