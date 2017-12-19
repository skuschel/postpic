#!/usr/bin/env python

import unittest
import postpic as pp
from postpic import io
#import postpic.io.common
import numpy as np
import os


class TestIO(unittest.TestCase):

    def gettempfile(self, suffix=''):
        import tempfile
        h, f = tempfile.mkstemp(suffix=suffix)
        os.close(h)
        print('filename is {}'.format(f))
        self._tempfiles.append(f)
        return f

    def setUp(self):
        self._tempfiles = []
        pp.chooseCode('DUMMY')
        self.dump = pp.datareader.readDump(100)
        self.testfield = self.dump.Ey()

    def tearDown(self):
        for f in self._tempfiles:
            os.remove(f)

    def test_importexport_npz(self):
        filename = self.gettempfile(suffix='.npz')
        self.testfield.export(filename)
        testfield2 = pp.load_field(filename)

        # now check if fields are equal
        self.assertTrue(np.all(np.isclose(np.asarray(self.testfield),
                                          np.asarray(testfield2))))
        # metadata
        self.assertEqual(self.testfield.name, testfield2.name)
        self.assertEqual(self.testfield.unit, testfield2.unit)
        self.assertTrue(np.all(self.testfield.axes_transform_state == testfield2.axes_transform_state))
        self.assertTrue(np.all(self.testfield.transformed_axes_origins == testfield2.transformed_axes_origins))
        # axes
        for n in range(0, len(self.testfield.axes)):
            self.assertEqual(len(self.testfield.axes[n]), len(testfield2.axes[n]))
            self.assertTrue(np.all(np.isclose(self.testfield.axes[n].grid_node,
                                              testfield2.axes[n].grid_node)))

    def test_export_csv(self):
        filename = self.gettempfile(suffix='.csv')
        self.testfield.export(filename)

    def test_export_vtk(self):
        filename = self.gettempfile(suffix='.vtk')
        self.testfield.export(filename)

    def test_arraydata(self):
        fieldX = np.arange(0,24).reshape(2,3,4)
        fieldY = np.arange(1,25).reshape(2,3,4)
        fieldZ = np.arange(2,26).reshape(2,3,4)

        vectors_help = io.vtk.ArrayData(fieldX, fieldY, fieldZ).transform_data(np.dtype('I'))

        reference = [0, 1, 2, 12, 13, 14, 4, 5, 6, 16, 17, 18, 8, 9, 10, 20, 21, 22,
                     1, 2, 3, 13, 14, 15, 5, 6, 7, 17, 18, 19, 9, 10, 11, 21, 22, 23,
                     2, 3, 4, 14, 15, 16, 6, 7, 8, 18, 19, 20, 10, 11, 12, 22, 23, 24,
                     3, 4, 5, 15, 16, 17, 7, 8, 9, 19, 20, 21, 11, 12, 13, 23, 24, 25]

        self.assertTrue(np.all(vectors_help == reference))


if __name__ == '__main__':
    unittest.main()
