#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
. "$DIR/common.sh"

ENV="$1"

rm -rf "${CONDA_ENVS}/${ENV}" "${CONDA_ENVS}/${ENV}_clean"

case $ENV in
ubuntu1404)
    conda create -y -p "$CONDA_ENVS/$ENV" python=3.4 matplotlib=1.3.1 numpy=1.8.1 scipy=0.13.3 cython=0.20.1 libgfortran=1 pep8 nose sphinx numexpr future
    source activate "$CONDA_ENVS/$ENV"
    pip install recommonmark
    pip install versioneer
    pip install urllib3
    pip install pyvtk
    source deactivate
    ;;

ubuntu1604py2)
    conda create -y -p "$CONDA_ENVS/$ENV" python=2.7 matplotlib=1.5.1 numpy=1.11.0 scipy=0.17.0 cython=0.23.4 pycodestyle nose sphinx numexpr future urllib3 functools32
    source activate "$CONDA_ENVS/$ENV"
    pip install recommonmark
    pip install versioneer
    pip install pyvtk
    source deactivate
    ;;

ubuntu1604py3)
    conda create -y -p "$CONDA_ENVS/$ENV" python=3.5 matplotlib=1.5.1 numpy=1.11.0 scipy=0.17.0 cython=0.23.4 pycodestyle nose sphinx numexpr future urllib3
    source activate "$CONDA_ENVS/$ENV"
    pip install recommonmark
    pip install versioneer
    pip install pyvtk
    source deactivate
    ;;

default2)
    conda create -y -p "$CONDA_ENVS/$ENV" python=2 matplotlib numpy scipy cython pycodestyle nose sphinx numexpr future urllib3 functools32
    source activate "$CONDA_ENVS/$ENV"
    pip install recommonmark
    pip install versioneer
    pip install pyvtk
    source deactivate
    ;;

default3)
    conda create -y -p "$CONDA_ENVS/$ENV" python=3 matplotlib numpy scipy cython pycodestyle nose sphinx numexpr future urllib3
    source activate "$CONDA_ENVS/$ENV"
    pip install recommonmark
    pip install versioneer
    pip install pyvtk
    source deactivate
    ;;

esac

cp -al "${CONDA_ENVS}/${ENV}" "${CONDA_ENVS}/${ENV}_clean"
