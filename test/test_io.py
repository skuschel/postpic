#!/usr/bin/env python

import unittest
import postpic as pp
from postpic import io
import numpy as np
import os


class TestIO(unittest.TestCase):

    def setUp(self):
        pp.chooseCode('DUMMY')
        self.dump = pp.datareader.readDump(100)
        self.testfield = self.dump.Ey()
        import tempfile
        f = tempfile.mkstemp(suffix='.npz')[1]
        print('filename is {}'.format(f))
        self.f = f
        io.export_field(f, self.testfield)
        self.testfield2 = io.load_field(f)

    def tearDown(self):
        os.remove(self.f)

    def test_data(self):
        self.assertTrue(np.all(np.isclose(self.testfield.matrix,
                                          self.testfield2.matrix)))

    def test_metadata(self):
        self.assertEqual(self.testfield.name, self.testfield2.name)
        self.assertEqual(self.testfield.unit, self.testfield2.unit)
        self.assertTrue(np.all(self.testfield.axes_transform_state == self.testfield2.axes_transform_state))
        self.assertTrue(np.all(self.testfield.transformed_axes_origins == self.testfield2.transformed_axes_origins))


    def test_axes(self):
        for n in range(0, len(self.testfield.axes)):
            self.assertEqual(len(self.testfield.axes[n]), len(self.testfield2.axes[n]))
            self.assertTrue(np.all(np.isclose(self.testfield.axes[n].grid_node,
                                              self.testfield2.axes[n].grid_node)))

    def test_vectors_help(self):
        fieldX = np.arange(0,24).reshape(2,3,4)
        fieldY = np.arange(1,25).reshape(2,3,4)
        fieldZ = np.arange(2,26).reshape(2,3,4)
        lengths = fieldX.shape

        vectors_help = io._make_vectors_help(fieldX, fieldY, fieldZ, lengths)

        reference = [[0, 1, 2], [12, 13, 14], [4, 5, 6], [16, 17, 18], [8, 9, 10], [20, 21, 22],
                     [1, 2, 3], [13, 14, 15], [5, 6, 7], [17, 18, 19], [9, 10, 11], [21, 22, 23],
                     [2, 3, 4], [14, 15, 16], [6, 7, 8], [18, 19, 20], [10, 11, 12], [22, 23, 24],
                     [3, 4, 5], [15, 16, 17], [7, 8, 9], [19, 20, 21], [11, 12, 13], [23, 24, 25]]

        self.assertEqual(vectors_help, reference)




if __name__ == '__main__':
    unittest.main()
