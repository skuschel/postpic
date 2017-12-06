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

if __name__ == '__main__':
    unittest.main()
