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
        self.data = np.random.random(1e5)
        self.weights = np.random.random(1e5)

    def test_histogram(self):
        # in this special case, cf.histogram and np.histogram must yield equal results
        # check hist and edges
        cfh, cfe = cf.histogram(self.data, bins=20, range=(0,1), shape=0)
        nph, npe = np.histogram(self.data, bins=20, range=(0,1))
        self.assertListEqual(list(nph), list(cfh))
        self.assertListEqual(list(npe), list(cfe))

    def test_histogramw(self):
        # in this special case, cf.histogram and np.histogram must yield equal results
        # check hist and edges
        cfh, cfe = cf.histogram(self.data, bins=100, range=(0,1), shape=0, weights=self.weights)
        nph, npe = np.histogram(self.data, bins=100, range=(0,1), weights=self.weights)
        diff = np.abs(cfh - nph)
        self.assertAlmostEqual(np.sum(diff), 0)
        self.assertListEqual(list(npe), list(cfe))

    def test_histogramo1(self):
        cfh, cfe = cf.histogram(self.data, bins=100, range=(0.0,1.0), shape=1)
        nph, npe = np.histogram(self.data, bins=100, range=(0.0,1.0))
        # just check that no mass is lost (cfh.base includes ghost cells)
        totalmass = len(self.data)
        self.assertAlmostEqual(np.sum(cfh.base) - totalmass, 0)
        self.assertAlmostEqual(np.sum(nph) - totalmass, 0)
        self.assertListEqual(list(npe), list(cfe))

    def test_histogramo1w(self):
        cfh, cfe = cf.histogram(self.data, bins=100, range=(0.0,1.0), shape=1, weights=self.weights)
        nph, npe = np.histogram(self.data, bins=100, range=(0.0,1.0), weights=self.weights)
        # just check that no mass is lost (cfh.base includes ghost cells)
        totalmass = np.sum(self.weights)
        self.assertAlmostEqual(np.sum(cfh.base) - totalmass, 0)
        self.assertAlmostEqual(np.sum(nph) - totalmass, 0)
        self.assertListEqual(list(npe), list(cfe))

    def test_histogramo2(self):
        cfh, cfe = cf.histogram(self.data, bins=100, range=(0.0,1.0), shape=2)
        nph, npe = np.histogram(self.data, bins=100, range=(0.0,1.0))
        # just check that no mass is lost (cfh.base includes ghost cells)
        totalmass = len(self.data)
        self.assertAlmostEqual(np.sum(cfh.base) - totalmass, 0)
        self.assertAlmostEqual(np.sum(nph) - totalmass, 0)
        self.assertListEqual(list(npe), list(cfe))

    def test_histogramo2w(self):
        cfh, cfe = cf.histogram(self.data, bins=100, range=(0.0,1.0), shape=2, weights=self.weights)
        nph, npe = np.histogram(self.data, bins=100, range=(0.0,1.0), weights=self.weights)
        # just check that no mass is lost (cfh.base includes ghost cells)
        totalmass = np.sum(self.weights)
        self.assertAlmostEqual(np.sum(cfh.base) - totalmass, 0)
        self.assertAlmostEqual(np.sum(nph) - totalmass, 0)
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
                                         bins=(20,30), range=((0,1), (0,2)), shape=0)
        nph, npex, npey = np.histogram2d(self.datax, self.datay,
                                          bins=(20, 30), range=((0,1), (0,2)))
        totalmass = len(self.datax)
        self.assertAlmostEqual(np.sum(cfh.base) - totalmass, 0)
        self.assertAlmostEqual(np.sum(nph) - totalmass, 0)
        self.assertListEqual(list(npex), list(cfex))
        self.assertListEqual(list(npey), list(cfey))

    def test_histogram2dw(self):
        # in this special case, cf.histogram2d and np.histogram2d must yield equal results
        # check hist and edges
        cfh, cfex, cfey = cf.histogram2d(self.datax, self.datay, weights=self.weights,
                                         bins=(20, 30), range=((0,1), (0,2)), shape=0)
        nph, npex, npey = np.histogram2d(self.datax, self.datay, weights=self.weights,
                                         bins=(20, 30), range=((0,1), (0,2)))
        totalmass = np.sum(self.weights)
        self.assertAlmostEqual(np.sum(cfh.base) - totalmass, 0)
        self.assertAlmostEqual(np.sum(nph) - totalmass, 0)
        self.assertListEqual(list(npex), list(cfex))
        self.assertListEqual(list(npey), list(cfey))

    def test_histogram2do1(self):
        # in this special case, cf.histogram2d and np.histogram2d must yield equal results
        # check hist and edges
        cfh, cfex, cfey = cf.histogram2d(self.datax, self.datay,
                                         bins=(20,30), range=((0,1), (0,2)), shape=1)
        nph, npex, npey = np.histogram2d(self.datax, self.datay,
                                          bins=(20, 30), range=((0,1), (0,2)))
        totalmass = len(self.datax)
        self.assertAlmostEqual(np.sum(cfh.base) - totalmass, 0)
        self.assertAlmostEqual(np.sum(nph) - totalmass, 0)
        self.assertListEqual(list(npex), list(cfex))
        self.assertListEqual(list(npey), list(cfey))

    def test_histogram2do1w(self):
        # in this special case, cf.histogram2d and np.histogram2d must yield equal results
        # check hist and edges
        cfh, cfex, cfey = cf.histogram2d(self.datax, self.datay, weights=self.weights,
                                         bins=(20, 30), range=((0,1), (0,2)), shape=1)
        nph, npex, npey = np.histogram2d(self.datax, self.datay, weights=self.weights,
                                         bins=(20, 30), range=((0,1), (0,2)))
        totalmass = np.sum(self.weights)
        self.assertAlmostEqual(np.sum(cfh.base) - totalmass, 0)
        self.assertAlmostEqual(np.sum(nph) - totalmass, 0)
        self.assertListEqual(list(npex), list(cfex))
        self.assertListEqual(list(npey), list(cfey))

    def test_histogram2do2(self):
        # in this special case, cf.histogram2d and np.histogram2d must yield equal results
        # check hist and edges
        cfh, cfex, cfey = cf.histogram2d(self.datax, self.datay,
                                         bins=(20,30), range=((0,1), (0,2)), shape=2)
        nph, npex, npey = np.histogram2d(self.datax, self.datay,
                                          bins=(20, 30), range=((0,1), (0,2)))
        totalmass = len(self.datax)
        self.assertAlmostEqual(np.sum(cfh.base) - totalmass, 0)
        self.assertAlmostEqual(np.sum(nph) - totalmass, 0)
        self.assertListEqual(list(npex), list(cfex))
        self.assertListEqual(list(npey), list(cfey))

    def test_histogram2do2w(self):
        # in this special case, cf.histogram2d and np.histogram2d must yield equal results
        # check hist and edges
        cfh, cfex, cfey = cf.histogram2d(self.datax, self.datay, weights=self.weights,
                                         bins=(20, 30), range=((0,1), (0,2)), shape=2)
        nph, npex, npey = np.histogram2d(self.datax, self.datay, weights=self.weights,
                                         bins=(20, 30), range=((0,1), (0,2)))
        totalmass = np.sum(self.weights)
        self.assertAlmostEqual(np.sum(cfh.base) - totalmass, 0)
        self.assertAlmostEqual(np.sum(nph) - totalmass, 0)
        self.assertListEqual(list(npex), list(cfex))
        self.assertListEqual(list(npey), list(cfey))


class TestHistogram3d(unittest.TestCase):

    def setUp(self):
        self.datax = np.random.random(1e3)
        self.datay = 2 * np.random.random(1e3)
        self.dataz = 3 * np.random.random(1e3)
        self.weights = np.random.random(1e3)

    def test_histogram3d(self):
        # in this special case, cf.histogram2d and np.histogram2d must yield equal results
        # check hist and edges
        cfh, cfex, cfey, cfez = cf.histogram3d(self.datax, self.datay, self.dataz,
                                         bins=(20,30,40), range=((0,1), (0,2), (0,3)), shape=0)
        totalmass = len(self.datax)
        self.assertAlmostEqual(np.sum(cfh.base) - totalmass, 0)

    def test_histogram3dw(self):
        # in this special case, cf.histogram2d and np.histogram2d must yield equal results
        # check hist and edges
        cfh, cfex, cfey, cfez = cf.histogram3d(self.datax, self.datay, self.dataz, weights=self.weights,
                                         bins=(20,30,40), range=((0,1), (0,2), (0,3)), shape=0)
        totalmass = np.sum(self.weights)
        self.assertAlmostEqual(np.sum(cfh.base) - totalmass, 0)


    def test_histogram3do1(self):
        # in this special case, cf.histogram2d and np.histogram2d must yield equal results
        # check hist and edges
        cfh, cfex, cfey, cfez = cf.histogram3d(self.datax, self.datay, self.dataz,
                                         bins=(20,30,40), range=((0,1), (0,2), (0,3)), shape=1)
        totalmass = len(self.datax)
        self.assertAlmostEqual(np.sum(cfh.base) - totalmass, 0)

    def test_histogram3do1w(self):
        # in this special case, cf.histogram2d and np.histogram2d must yield equal results
        # check hist and edges
        cfh, cfex, cfey, cfez = cf.histogram3d(self.datax, self.datay, self.dataz, weights=self.weights,
                                         bins=(20,30,40), range=((0,1), (0,2), (0,3)), shape=1)
        totalmass = np.sum(self.weights)
        self.assertAlmostEqual(np.sum(cfh.base) - totalmass, 0)

    def test_histogram3do2(self):
        # in this special case, cf.histogram2d and np.histogram2d must yield equal results
        # check hist and edges
        cfh, cfex, cfey, cfez = cf.histogram3d(self.datax, self.datay, self.dataz,
                                         bins=(20,30,40), range=((0,1), (0,2), (0,3)), shape=2)
        totalmass = len(self.datax)
        self.assertAlmostEqual(np.sum(cfh.base) - totalmass, 0)

    def test_histogram3do2w(self):
        # in this special case, cf.histogram2d and np.histogram2d must yield equal results
        # check hist and edges
        cfh, cfex, cfey, cfez = cf.histogram3d(self.datax, self.datay, self.dataz, weights=self.weights,
                                         bins=(20,30,40), range=((0,1), (0,2), (0,3)), shape=2)
        totalmass = np.sum(self.weights)
        self.assertAlmostEqual(np.sum(cfh.base) - totalmass, 0)

if __name__ == '__main__':
    unittest.main()
