
Contributing to the PostPic code base
=====================================

Any help contributing to the PostPic project ist greatly appreciated! Feel free to contact any of the developers or ask for help using the [Issues](https://github.com/skuschel/postpic/issues) Page.

Why me?
-------

because you are using it!


How to contribute?
------------------

Reporting bugs or asking questions works with a GitHub account simply on the [Issues](https://github.com/skuschel/postpic/issues) page.

For any coding you need to be familiar with [git](http://git-scm.com/). Its a distributed version control system created by Linus Torvalds (and more importantly: he is also using it for maintaining the linux kernel). There is a nice introduction to git at [try.github.io/](http://try.github.io/), but in general you can follow the bootcamp section at [https://help.github.com/](https://help.github.com/) for your first steps. 

One of the most comprehensive guides is probably [this book](http://git-scm.com/doc). Just start reading from the beginning. It is worth it!

## The Workflow

The typical workflow should be:

  0. [Fork](https://help.github.com/articles/fork-a-repo) the PostPic repo to your own GitHub account.
  0. Clone from your fork to your local computer.
  0. Implement a new feature/bugfix/documentation/whatever commit to your local repository. It is highly recommended that new features will have test cases.
  0. KEEP YOUR FORK UP TO DATE! Your fork is yours, only. So you have to update it to whatever happens in the main repository. To do so add the main repository as a second remote with
  
   `git remote add upstream git@github.com:skuschel/postpic.git`

   and pull from it regularly with

  `git pull --rebase upstream master`
  
  0. Make sure all tests are running smoothly (the `run-tests` script also involves pep8 style verification!)
  0. push to your fork and create a [pull request](https://help.github.com/articles/using-pull-requests/) to merge your changes into the codebase.

## Coding and general remaks

  *  Make sure, that the `run-tests` script exits without error on EVERY commit. To do so, it is HIGHLY RECOMMENDED to add the `pre-commit` script as the git pre-commit hook. For instructions see [pre-commit](../master/pre-commit).
  * The Coding style is according to slightly simplified pep8 rules. This is included in the `run-tests` script. If that script runs without error, you should be good to <del>go</del> commit.
  * If your implemented feature works as expected you can send the pull request to the master branch. Additional branches should be used only if there are unfinished or experimental features.
  * Add the GPLv3+ licence notice on top of every new file. If you add a new file you are free to add your name as a author. This will let other people know that you are in charge if there is any trouble with the code. This is only useful if the file you provide adds functionality like a new datareader. Thats why the `__init__.py` files typically do not have a name written. In doubt, the git revision history will always show who added which line.



## What to contribute?

Here is a list for your inspiration:

  * Add Documentation and usage examples.
  * Report bugs at the [Issues](https://github.com/skuschel/postpic/issues) page.
  * Fix bugs from the [Issues](https://github.com/skuschel/postpic/issues) page.
  * Add python docstrings to the codebase.
  * Add new features.
  * Add new datareader for additional file formats.
  * Add test cases
  * ...
