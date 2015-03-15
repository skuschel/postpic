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

class TestHistogram(unittest.TestCase):

    def setUp(self):
        self.data = np.random.random(1e3)
        self.weights = np.random.random(1e3)

    def test_histogram(self):
        # in this special case, cf.histogram and np.histogram must yield equal results
        # check hist and edges
        cfh, cfe = cf.histogram(self.data, bins=20, range=(0,1), order=0)
        nph, npe = np.histogram(self.data, bins=20, range=(0,1))
        self.assertListEqual(list(nph), list(cfh))
        self.assertListEqual(list(npe), list(cfe))

    def test_histogramw(self):
        # in this special case, cf.histogram and np.histogram must yield equal results
        # check hist and edges
        cfh, cfe = cf.histogram(self.data, bins=100, range=(0,1), order=0, weights=self.weights)
        nph, npe = np.histogram(self.data, bins=100, range=(0,1), weights=self.weights)
        diff = np.abs(cfh - nph)
        self.assertAlmostEqual(np.sum(diff), 0)
        self.assertListEqual(list(npe), list(cfe))

    def test_histogramo1(self):
        cfh, cfe = cf.histogram(self.data, bins=100, range=(-0.1,1.1), order=1)
        nph, npe = np.histogram(self.data, bins=100, range=(-0.1,1.1))
        # just check that no mass is lost
        diff = np.sum(cfh) - np.sum(nph)
        self.assertAlmostEqual(diff, 0)
        self.assertListEqual(list(npe), list(cfe))

    def test_histogramo1w(self):
        cfh, cfe = cf.histogram(self.data, bins=100, range=(-0.1,1.1), order=1, weights=self.weights)
        nph, npe = np.histogram(self.data, bins=100, range=(-0.1,1.1), weights=self.weights)
        # just check that no mass is lost
        diff = np.sum(cfh) - np.sum(nph)
        self.assertAlmostEqual(diff, 0)
        self.assertListEqual(list(npe), list(cfe))


class TestHistogram2d(unittest.TestCase):

    def setUp(self):
        self.datax = np.random.random(1e3)
        self.datay = 2 * np.random.random(1e3)
        self.weights = np.random.random(1e3)

    def test_histogram2d(self):
        # in this special case, cf.histogram2d and np.histogram2d must yield equal results
        # check hist and edges
        cfh, cfex, cfey = cf.histogram2d(self.datax, self.datay,
                                         bins=(20,30), range=((0,1), (0,2)), order=0)
        nph, npex, npey = np.histogram2d(self.datax, self.datay,
                                          bins=(20, 30), range=((0,1), (0,2)))
        self.assertAlmostEqual(0.0, np.sum(np.abs(nph - cfh)))
        self.assertListEqual(list(npex), list(cfex))
        self.assertListEqual(list(npey), list(cfey))

    def test_histogram2dw(self):
        # in this special case, cf.histogram2d and np.histogram2d must yield equal results
        # check hist and edges
        cfh, cfex, cfey = cf.histogram2d(self.datax, self.datay, weights=self.weights,
                                         bins=(20, 30), range=((0,1), (0,2)), order=0)
        nph, npex, npey = np.histogram2d(self.datax, self.datay, weights=self.weights,
                                         bins=(20, 30), range=((0,1), (0,2)))
        self.assertAlmostEqual(0.0, np.sum(np.abs(nph - cfh)))
        self.assertListEqual(list(npex), list(cfex))
        self.assertListEqual(list(npey), list(cfey))



if __name__ == '__main__':
    unittest.main()
