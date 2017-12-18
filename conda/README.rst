
Scripts for testing
===================

These scripts will help you to test postpic against different python platforms.
Beware that, by default, they will create a directory ``conda``, next to your clone of postpic.git.
So if you have cloned postpic to ``/home/username/postpic``, the scripts will use ``/home/username/conda`` as base directory.
You can override this by changing the ``CONDA_BASE`` in ``conda/common.sh``.
See ``conda/common.sh`` to find out how the other path names mentioned here are created and other variables.

Description of the scripts
--------------------------

The scripts have the following tasks:

* create_environment.sh ENV
    Build the environment ENV in ``CONDA_ENVS/ENV`` and create a hard-linked copy in ``CONDA_ENVS/ENV_clean``.

* make_envs.sh
    Build all the environments listed in ``ENVS``.

* do_tests_env.sh ENV ARGS
    run the tests on a single environment ENV. This will assume that the copy of your sources in ``SOURCETMP`` is already up-to-date.
    Additional ARGS are passed to postpics ``run-tests.py`` script.

* run_tests.sh ARGS
    This runs the tests on all environments after updating the copy of your sources in ``SOURCETMP`` with the contents of ``SOURCE``.
    ARGS are passed on to ``run-tests.py`` (via ``do_tests_env.sh``). Because Environments might be modified during the tests (e.g. by installation of
    third-party packages), a new hard-linked copy of the clean environment is used each time.

tl;dr, what do I need to to?
----------------------------

Run ``conda/make_envs.sh`` once. After that you can use ``conda/run_tests.sh`` as a replacement for ``run-tests.py``. You can even symlink the ``pre-commit`` from this folder to your ``.git/hooks`` like this, if the current working directory is the directory which contains this file::

  ln -s ../../conda/pre-commit ../.git/hooks/pre-commit
