postpic
=======

[![run-tests](https://github.com/skuschel/postpic/workflows/run-tests/badge.svg)](https://github.com/skuschel/postpic/actions?query=workflow%3Arun-tests)
[![PyPI version](https://badge.fury.io/py/postpic.png)](http://badge.fury.io/py/postpic)

Postpic is an open-source post-processor for Particle-in-cell (PIC) simulations written in python. If you are doing PIC Simulations (likely for you PhD in physics...) you need tools to provide proper post-processing to create nice graphics from many GB of raw simulation output data -- regardless of what simulation code you are using.

**For working examples, please go to  https://github.com/skuschel/postpic-examples**

The (technical, but complete) documentation is hosted on
https://skuschel.github.io/postpic/


Idea of postpic
---------------

The basic idea of postpic is to calculate the plots you are interested in just from the basic data the simulation provides. This data includes electric and magnetic fields and a tuple (`weight`, `x`, `y`, `z`, `px`, `py`, `pz`, `id`, `mass`, `charge`, `time`) for every macro-particle. Anything else you like to look at (for example a spectrum at your detector) should just be calculated from these values. This is exactly what postpic can do for you, and even more:

postpic has a unified interface for reading the required simulation data. If the simulation code of your choice is not supported by postic, this is the perfect opportunity to add a new datareader.

Additionally postpic can plot and label your plot automatically. This makes it easy to work with and avoids mistakes. Currently matplotlib is used for that but this is also extensible.



Installation
------------

Postpic can be used with `python2` and `python3`. However the usage of `python3` is recommended.

**Users** should install the latest version directly from github using `pip` (python package manager):

`pip install --user git+https://github.com/skuschel/postpic.git`

Please note that, depending on your system python setup, `pip` may default to `python2`.
In that case you will need use `pip3` instead to make sure that postpic is installed for `python3`:

`pip3 install --user git+https://github.com/skuschel/postpic.git`

The latest *release* is also available in the python package index [pypi](https://pypi.python.org/pypi/postpic/), thus it can be installed by using the python package manager pip:

`pip install --user postpic`

In both cases the `--user` option makes sure that the package can be installed without `root` privileges by installing it into your user profile.

**Developers** should clone the git repository (or their fork of it) and install it using

`pip install --user -e .`

This command will link the current folder to global python scope, such that changing the code will immediately update the installed package.

**After installing** you should be able to import it into any python session using `import postpic`.

Postpic's main functions should work but there is still much work to do and lots of documentation missing. If postpic awakened your interest you are welcome to contribute. Even if your programming skills are limited there are various ways how to contribute and adopt postpic for your particular research. Read [CONTRIBUTING.md](../master/CONTRIBUTING.md).


Postpic in Science
------------------

If you use postpic for your research and present or publish results, please show your support for the postpic project and its [contributers](https://github.com/skuschel/postpic/graphs/contributors) by:

  * Add a note in the acknowledgements section of your publication.
  * Drop a line to one of the core developers including a link to your work to make them smile (there might be a public list in the future).


License
-------

Postpic is released under GPLv3+. See [http://www.gnu.org/licenses/gpl-howto.html](http://www.gnu.org/licenses/gpl-howto.html) for further information.
