#!/usr/bin/env python2

import unittest
import postpic as pp
import numpy as np

class TestMultiSpecies(unittest.TestCase):
    
    def setUp(self):
        pp.chooseCode('dummy')
        self.dr = pp.readDump(10000)
        self.p = pp.MultiSpecies(self.dr, 'electron')

    def test_dummyreader(self):
        self.assertEqual(self.dr.timestep(), 10000)

    def test_pa(self):
        self.assertEqual(len(self.p), 10000)

    def test_mean(self):
        self.assertAlmostEqual(self.p.mean(pp.MultiSpecies.X), -0.0184337)
        self.assertAlmostEqual(self.p.mean(pp.MultiSpecies.X, weights=self.p.beta()), -0.01965867)

    def test_var(self):
        self.assertAlmostEqual(self.p.var(pp.MultiSpecies.X),0.97526797)
        self.assertAlmostEqual(self.p.var(pp.MultiSpecies.Y),0.98615759)
        self.assertRaises(KeyError, self.p.var, pp.MultiSpecies.Z)

if __name__ == '__main__':
    unittest.main()
