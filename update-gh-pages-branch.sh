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
# Copyright Stephan Kuschel, 2020

# This script creates the documentation as html and saves it to the
# "gh-pages" branch.

set -eux

# ensure Working Directory is clean
git diff-index --quiet HEAD

if ! git rev-parse --verify --quiet gh-pages; then
  current_branch=$(git branch --show-current)
  echo 'create empty and orphan gh-pages branch'
  git checkout --orphan gh-pages
  git reset
  git commit --allow-empty --no-verify -m "init empty gh-pages branch"
  git checkout -f $current_branch
fi

# assert that Working Directory is still clean
git diff-index --quiet HEAD
SHA=$(git rev-parse --verify HEAD)

# create documentation on gh-pages branch
rm -rf doc/build/html
git worktree add -f doc/build/html gh-pages
cd doc
make html
cd build/html
touch .nojekyll
git add .
git commit -a --no-verify -m "postpic documentation at $SHA"




