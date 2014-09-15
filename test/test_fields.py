#!/usr/bin/env python2

import unittest
import postpic.analyzer as pa
import postpic.datareader as da
import numpy as np

class TestFieldAnalyzer(unittest.TestCase):
    
    def setUp(self):
        da.chooseCode('dummy')
        self.dr = da.readDump(10000)
        self.fa = pa.FieldAnalyzer(self.dr)

    def test_fa(self):
        emfield = self.fa.energydensityEM()
        self.assertEqual(emfield.name, 'Energy Density EM-Field')

if __name__ == '__main__':
    unittest.main()
