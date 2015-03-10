#!/usr/bin/env python2
# -*- coding: utf-8 -*-
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
# Copyright Stephan Kuschel, 2015

import unittest
import postpic.cythonfunctions as cf
import numpy as np

class TestCythonfunctions(unittest.TestCase):

    def setUp(self):
        self.data = np.random.random(1e3)
        self.weights = np.random.random(1e3)

    def test_histogram(self):
        # in this special case, cf.histogram and np.histogram must yield equal results
        # check hist and edges
        cfh, cfe = cf.histogram(self.data,bins=20, range=(0,1))
        nph, npe = np.histogram(self.data, bins=20, range=(0,1))
        self.assertListEqual(list(nph), list(cfh))
        self.assertListEqual(list(npe), list(cfe))

    def test_histogramw(self):
        # in this special case, cf.histogram and np.histogram must yield equal results
        # check hist and edges
        cfh, cfe = cf.histogram(self.data,bins=100, range=(0,1), weights=self.weights)
        nph, npe = np.histogram(self.data, bins=100, range=(0,1), weights=self.weights)
        diff = np.abs(cfh - nph)
        self.assertAlmostEqual(np.sum(diff), 0)
        self.assertListEqual(list(npe), list(cfe))

    def test_histogramo1(self):
        cfh, cfe = cf.histogram(self.data,bins=100, range=(-0.1,1.1), order=1)
        nph, npe = np.histogram(self.data, bins=100, range=(-0.1,1.1))
        # just check that no mass is lost
        diff = np.sum(cfh) - np.sum(nph)
        self.assertAlmostEqual(diff, 0)
        self.assertListEqual(list(npe), list(cfe))

    def test_histogramo1w(self):
        cfh, cfe = cf.histogram(self.data,bins=100, range=(-0.1,1.1), order=1, weights=self.weights)
        nph, npe = np.histogram(self.data, bins=100, range=(-0.1,1.1), weights=self.weights)
        # just check that no mass is lost
        diff = np.sum(cfh) - np.sum(nph)
        self.assertAlmostEqual(diff, 0)
        self.assertListEqual(list(npe), list(cfe))

if __name__ == '__main__':
    unittest.main()
