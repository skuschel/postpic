#!/usr/bin/env python2

import unittest
import postpic.analyzer as pa
import postpic.datareader as da
import numpy as np

class TestParticleAnalyzer(unittest.TestCase):
    
    def setUp(self):
        da.chooseCode('dummy')
        self.dr = da.readDump(10000)
        self.pa = pa.ParticleAnalyzer(self.dr, 'electron')

    def test_dummyreader(self):
        self.assertEqual(self.dr.timestep(), 10000)

    def test_pa(self):
        self.assertEqual(len(self.pa), 10000)

    def test_mean(self):
        self.assertAlmostEqual(self.pa.mean(pa.ParticleAnalyzer.X), -0.0184337)
        self.assertAlmostEqual(self.pa.mean(pa.ParticleAnalyzer.X, weights=self.pa.beta()), -0.01965867)

    def test_var(self):
        self.assertAlmostEqual(self.pa.var(pa.ParticleAnalyzer.X),0.97526797)
        self.assertAlmostEqual(self.pa.var(pa.ParticleAnalyzer.Y),0.98615759)
        self.assertRaises(KeyError, self.pa.var, pa.ParticleAnalyzer.Z)

if __name__ == '__main__':
    unittest.main()
