#!/usr/bin/env python3

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
        self.assertAlmostEqual(self.p.mean('x'), -0.0184337)
        self.assertAlmostEqual(self.p.mean('x', weights='beta'), -0.01965867)

    def test_var(self):
        self.assertAlmostEqual(self.p.var('x'), 0.97526797)
        self.assertAlmostEqual(self.p.var('y'), 0.98615759)
        self.assertRaises(KeyError, self.p.var, 'z')

if __name__ == '__main__':
    unittest.main()
