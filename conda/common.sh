
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# configure these to your needs, defaults should work fine though
CONDA_BASE=$DIR/../../conda
SOURCEDIRNAME="$(basename "$(dirname "$DIR")")"
SOURCE="$(dirname "$DIR")"
CONDA_ENVS=$CONDA_BASE/envs
LOGDIR=$CONDA_BASE/log
TMPDIR=$CONDA_BASE/tmp
ENVS=(ubuntu1404 ubuntu1604py2 ubuntu1604py3 default2 default3)


mkdir -p "$CONDA_BASE" "$CONDA_ENVS" "$LOGDIR" "$TMPDIR"
SOURCETMP=$TMPDIR/$SOURCEDIRNAME
