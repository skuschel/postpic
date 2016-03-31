postpic
=======
[![Build Status](https://travis-ci.org/skuschel/postpic.svg?branch=master)](https://travis-ci.org/skuschel/postpic)
[![PyPI version](https://badge.fury.io/py/postpic.png)](http://badge.fury.io/py/postpic)
[![Documentation Status](https://readthedocs.org/projects/postpic/badge/?version=latest)](https://postpic.readthedocs.org/)

postpic is a open-source post processor for Particle-in-cell (PIC) simulations written in python.  If you are doing PIC Simulations (likely for you PhD in physics...) you need tools to provide proper post-processing to create nice graphics from many GB of raw simulation output data -- regardless of what simulation code you are using.

Idea of postpic
---------------

The basic idea of postpic is to calculate the plots you are interested in just from the basic data the Simulation provides. This data includes electric and magnetic fields and a tuple (weight, x, y, z, px, py, pz) for every macro-particle. Anything else you want to look at (for example a spectrum at your detector) should just be calculated from these values. This is exactly what postpic can do for you, and even more:

postpic has a unified interface for reading the required simulation data. If the simulation code of your choice is not supported by postic, this is the perfect opportunity to add a new datareader.

Additionally postpic can plot and label your plot automatically. This makes it easy to work with and avoids mistakes. Currently matplotlib is used for that but this is also extensible.

Usage
-----
postpic can be used with python2 and python3. However the usage of python3 is recommended.
postpic is available in the python package index [pypi](https://pypi.python.org/pypi/postpic/), thus it can be directly installed by using the python package manager pip:

`pip install postpic`

pip will download and install the latest release from [pypi](https://pypi.python.org/pypi/postpic/) for you.

Alternativeley you can download the latest version from github and run

`python setup.py install --user`

to install postpic in your local home folder. After that you should be able to import it into any python session using `import postpic`.

If you are changing the postpic codebase it is probably best to link the current folder. This can be done by

`python setup.py develop --user`

However, postpic's main functions should work but there is still much work to do and lots of documentation missing. If postpic awakened your interest you are welcome to contribute. Even if your programming skills are limited there are various ways how to contribute and adopt postpic for your particular research. Read [CONTRIBUTING.md](../master/CONTRIBUTING.md).


postpic in Science
------------------

If you use postpic for your research and present or publish results, please show your support for the postpic project and its [contributers](https://github.com/skuschel/postpic/graphs/contributors) by:

  * Add a note in the acknowledgements section of your publication.
  * Drop a line to one of the core developers including a link to your work to make them smile (there might be a public list in the future).

License
-------

Postpic is released under GPLv3+. See [http://www.gnu.org/licenses/gpl-howto.html](http://www.gnu.org/licenses/gpl-howto.html) for further information.
