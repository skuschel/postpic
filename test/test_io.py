#!/usr/bin/env python3

import unittest
import postpic as pp
from postpic import io
import numpy as np



class TestIO(unittest.TestCase):

    def setUp(self):
        pp.chooseCode('DUMMY')
        self.dump = pp.datareader.readDump(100)
        self.testfield = self.dump.Ey()
        io._export_field_npy(self.testfield, 'test.npz')
        self.testfield2 = io._import_field_npy('test.npz')

    def test_data(self):
        assert np.all(np.isclose(self.testfield.matrix, self.testfield2.matrix))

    def test_metadata(self):
        self.assertEqual(self.testfield.name, self.testfield2.name)
        self.assertEqual(self.testfield.unit, self.testfield2.unit)

    def test_axes(self):
        self.assertEqual(len(self.testfield.axes), len(self.testfield2.name))
        for n in range(0, len(self.testfield.axes)):
            self.assertEqual(len(self.testfield.axes[n]), len(self.testfield2.axes[n]))
