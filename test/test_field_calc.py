#!/usr/bin/env python

import unittest
import postpic as pp
import numpy as np

class TestFieldAnalyzer(unittest.TestCase):

    def setUp(self):
        pp.chooseCode('dummy')
        self.dr = pp.readDump(10000)

    def test_fa(self):
        emfield = self.dr.energydensityEM()
        self.assertEqual(emfield.name, 'Energy Density EM-Field')

if __name__ == '__main__':
    unittest.main()
