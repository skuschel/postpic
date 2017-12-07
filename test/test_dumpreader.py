#!/usr/bin/env python

import unittest
import postpic.datareader as da
import numpy as np

class TestDumpReader(unittest.TestCase):

    def setUp(self):
        da.chooseCode('dummy')
        self.dr1d = da.readDump(10000, dimensions=1)
        self.dr2d = da.readDump(10000, dimensions=2)
        self.dr3d = da.readDump(10000, dimensions=3)
        self.sr3d = da.readSim(11377, dimensions=3)

    def test_dims(self):
        self.assertEqual(self.dr1d.simdimensions(), 1)
        self.assertEqual(self.dr2d.simdimensions(), 2)
        self.assertEqual(self.dr3d.simdimensions(), 3)
        self.assertEqual(self.sr3d[20].simdimensions(), 3)

    def test_general(self):
        self.assertRaises(KeyError, self.dr1d.getSpecies, 'electron', 1)
        self.assertRaises(KeyError, self.dr1d.getSpecies, 'electron', 2)
        self.assertRaises(KeyError, self.dr2d.getSpecies, 'electron', 2)
        self.assertEqual(len(self.dr3d.getSpecies('electron', 1)), 10000)
        self.assertEqual(len(self.sr3d), 11377)
        pz = self.sr3d[7777].getSpecies('electron', 'pz')
        self.assertEqual(len(pz), 7777)
        pz = self.dr1d.getSpecies('electron', 'pz')
        self.assertAlmostEqual(np.sum(pz), 0)

if __name__ == '__main__':
    unittest.main()
