
Scripts for testing
===================

These scripts will help you to test postpic against different python platforms.

Please note that the canonical tests for the postpit projects are the one run via ``.travis.yml`` and ``travis-ci``.
In contrast, the scripts in this directoryare meant to have a method to do local tests against different platforms in parallel and as fast as possible so they are suitable as a pre-commit hook.

Prerequisites
-------------

All you need to install on your system beforehand is the ``conda`` package manager.
On Arch linux it is available as an AUR package ``python-conda``.
The ``conda`` package manager is used to create environments like ``virtualenv``, where it is able to install different python versions like e. g. ``pyenv``.
It can easily install binary python extensions like ``numpy`` into these environments from pre-compiled packages that match the python interpreter, which is quicker and more robust than using ``pip``.
Alternatives to these shell scripts may be tools like ``tox``/``detox`` which will use ``pyenv``, ``virtualenv`` and ``pip`` for all packages, which means a lot of different tools that interoperate, which may or may not lead to difficulties.
The solution here uses only ``bash``, ``conda`` and ``pip``, which may or may not be more robust.

Beware that, by default, they will create a directory ``conda``, next to your clone of postpic.git.
So if you have cloned postpic to ``/home/username/postpic``, the scripts will use ``/home/username/conda`` as base directory.
With the default configuration you will need a few gigabytes of space there.
You can override this by changing the ``CONDA_BASE`` in ``conda/common.sh``.
See ``conda/common.sh`` to find out how the other path names mentioned here are created and other variables.

Quickstart
----------

Run ``conda/make_envs.sh`` once.
After that, you can use ``conda/run_tests.sh`` as a replacement for ``run-tests.py``.
You can even symlink the ``pre-commit`` from this folder to your ``.git/hooks`` like this, if the current working directory is the directory which contains this file::

  ln -s ../../conda/pre-commit ../.git/hooks/pre-commit


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
