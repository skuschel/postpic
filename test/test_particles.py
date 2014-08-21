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

if __name__ == '__main__':
    unittest.main()
