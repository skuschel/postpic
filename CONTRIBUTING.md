
Contributing to the postpic code base
=====================================

Any help contributing to the postpic project ist greatly appreciated! Feel free to contact any of the developers or ask for help using the [Issues](https://github.com/skuschel/postpic/issues) Page.

Why me?
-------

because you are using it!


How to contribute?
------------------

Reporting bugs or asking questions works with a GitHub account simply on the [Issues](https://github.com/skuschel/postpic/issues) page.

For any coding you need to be familiar with [git](http://git-scm.com/). Its a distributed version control system created by Linus Torvalds (and more importantly: he is also using it for maintaining the linux kernel). There is a nice introduction to git at [try.github.io/](http://try.github.io/), but in general you can follow the bootcamp section at [https://help.github.com/](https://help.github.com/) for your first steps.

One of the most comprehensive guides is probably [this book](http://git-scm.com/doc). Just start reading from the beginning. It is worth it!

## The Workflow

Adding a feature is often triggered by the personal demand for it. Thats why production ready features should propagte to master as fast as possible. Everything on master is considered to be production ready. We follow the  [github-flow](http://scottchacon.com/2011/08/31/github-flow.html) describing this very nicely.

In short:

  0. [Fork](https://help.github.com/articles/fork-a-repo) the PostPic repo to your own GitHub account.
  0. Clone from your fork to your local computer.
  0. Create a branch whose name tells what you do. Something like `codexy-reader` or `fixwhatever`,... is a good choice. Do NOT call it `issue42`. Git history should be clearly readable without external information. If its somehow unspecific in the worst case call it `dev` or even commit onto your `master` branch.
  0. Implement a new feature/bugfix/documentation/whatever commit to your local repository. It is highly recommended that the new features will have test cases.
  0. KEEP YOUR FORK UP TO DATE! Your fork is yours, only. So you have to update it to whatever happens in the main repository. To do so add the main repository as a second remote with

   `git remote add upstream git@github.com:skuschel/postpic.git`

   and pull from it regularly with

  `git pull --rebase upstream master`

  0. Make sure all tests are running smoothly (the `run-tests.py` script also involves pep8 style verification!) Run `run-tests.py` before EVERY commit!
  0. push to your fork and create a [pull request](https://help.github.com/articles/using-pull-requests/) EARLY! Even if your feature or fix is not yet finished, create the pull request and start it with `WIP:` or `[WIP]` (work-in-progress) to show its not yet ready to merge in. But the pull request will
    * trigger travis.ci to run the tests whenever you push
    * show other people what you work on
    * ensure early feedback on your work


## Coding and general remaks

  *  Make sure, that the `run-tests.py` script exits without error on EVERY commit. To do so, it is HIGHLY RECOMMENDED to add the `pre-commit` script as the git pre-commit hook. For instructions see [pre-commit](../master/pre-commit).
  * The Coding style is according to slightly simplified pep8 rules. This is included in the `run-tests.py` script. If that script runs without error, you should be good to <del>go</del> commit.
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
