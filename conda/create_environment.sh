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

ENV="$1"

rm -rf "${CONDA_ENVS}/${ENV}" "${CONDA_ENVS}/${ENV}_clean"

case $ENV in
ubuntu1404)
    conda create -y -p "$CONDA_ENVS/$ENV" python=3.4 matplotlib=1.3.1 numpy=1.8.1 scipy=0.13.3 cython=0.20.1 libgfortran=1 pep8 nose sphinx numexpr future
    source activate "$CONDA_ENVS/$ENV"
    pip install recommonmark
    pip install urllib3
    pip install pyvtk
    source deactivate
    ;;

ubuntu1604py2)
    conda create -y -p "$CONDA_ENVS/$ENV" python=2.7 matplotlib=1.5.1 numpy=1.11.0 scipy=0.17.0 cython=0.23.4 pycodestyle nose sphinx numexpr future urllib3 functools32
    source activate "$CONDA_ENVS/$ENV"
    pip install recommonmark
    pip install pyvtk
    source deactivate
    ;;

ubuntu1604py3)
    conda create -y -p "$CONDA_ENVS/$ENV" python=3.5 matplotlib=1.5.1 numpy=1.11.0 scipy=0.17.0 cython=0.23.4 pycodestyle nose sphinx numexpr future urllib3
    source activate "$CONDA_ENVS/$ENV"
    pip install recommonmark
    pip install pyvtk
    source deactivate
    ;;

default2)
    conda create -y -p "$CONDA_ENVS/$ENV" python=2 matplotlib numpy scipy cython pycodestyle nose sphinx numexpr future urllib3 functools32
    source activate "$CONDA_ENVS/$ENV"
    pip install recommonmark
    pip install pyvtk
    source deactivate
    ;;

default3)
    conda create -y -p "$CONDA_ENVS/$ENV" python=3 matplotlib numpy scipy cython pycodestyle nose sphinx numexpr future urllib3
    source activate "$CONDA_ENVS/$ENV"
    pip install recommonmark
    pip install pyvtk
    source deactivate
    ;;

esac

cp -al "${CONDA_ENVS}/${ENV}" "${CONDA_ENVS}/${ENV}_clean"
