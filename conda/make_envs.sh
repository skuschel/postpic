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

for ENV in ${ENVS[@]}; do
    echo Create environment "$CONDA_ENVS/$ENV"
    #script -c "$CONDA_BASE/create_environment.sh $ENV" $LOGDIR/create_$ENV >/dev/null &
    "$DIR/create_environment.sh" $ENV >"$LOGDIR/create_$ENV" 2>&1 &
done

wait
