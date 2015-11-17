#!/usr/bin/env python2

import unittest
import postpic.datahandling as dh
import numpy as np


class TestAxis(unittest.TestCase):

    def setUp(self):
        self.ax = dh.Axis(name='name', unit='unit')

    def test_simple(self):
        self.assertEqual(self.ax.name, 'name')
        self.assertEqual(self.ax.unit, 'unit')

    def test_grid_general(self):
        self.ax.setextent([-1, 1], 101)
        self.assertEqual(self.ax.extent, [-1, 1])
        self.assertEqual(len(self.ax), 101)
        self.assertEqual(len(self.ax.grid), 101)
        self.assertEqual(len(self.ax.grid_node), 102)
        self.assertEqual(self.ax.grid_node[0], -1)
        self.assertEqual(self.ax.grid_node[-1], 1)
        self.assertTrue(self.ax.islinear())
        self.assertTrue(self.ax.islinear(force=True))

    def test_initiate(self):
        ax = dh.Axis()
        self.assertEqual(ax.name, '')
        self.assertEqual(ax.unit, '')

    def test_cutout(self):
        # even number of grid points
        self.ax.setextent((-1, 1), 100)
        self.ax.cutout((0, 1))
        self.assertEqual(len(self.ax), 50)
        self.assertEqual(self.ax.grid_node[0], 0)
        # odd number of grid points
        self.ax.setextent((-1, 1), 101)
        self.ax.cutout((-0.01, 1))
        self.assertEqual(len(self.ax), 51)
        self.assertEqual(self.ax.grid[0], 0)

    def test_half_resolution(self):
        # even number of grid points
        self.ax.setextent((10, 20), 100)
        self.ax.half_resolution()
        self.assertEqual(len(self.ax), 50)
        # odd number of grid points
        self.ax.setextent((10, 20), 101)
        self.ax.half_resolution()
        self.assertEqual(len(self.ax), 50)

    def test_extent(self):
        self.assertTrue(self.ax.extent is None)
        self.ax.grid_node = [1]
        self.assertTrue(self.ax.extent is None)
        self.ax.grid_node = [1, 2.7]
        self.assertEqual(self.ax.extent, [1, 2.7])

    def test_grid(self):
        self.ax.grid = [5, 6]
        self.assertEqual(self.ax.grid[0], 5)
        self.assertEqual(self.ax.grid[1], 6)

    def test_grid_node(self):
        self.ax.grid_node = [5, 6]
        self.assertEqual(self.ax.grid[0], 5.5)
        self.assertTrue(all(self.ax.grid_node == [5, 6]))


class TestField(unittest.TestCase):

    def setUp(self):
        self.fempty = dh.Field([])
        self.f0d = dh.Field([42])
        m = np.reshape(np.arange(10), 10)
        self.f1d = dh.Field(m)
        m = np.reshape(np.arange(20), (4, 5))
        self.f2d = dh.Field(m)
        m = np.reshape(np.arange(60), (4, 5, 3))
        self.f3d = dh.Field(m)

    def checkFieldConsistancy(self, field):
        '''
        general consistancy check. must never fail.
        '''
        self.assertEqual(field.dimensions, len(field.axes))
        for i in range(len(field.axes)):
            self.assertEqual(len(field.axes[i]), field.matrix.shape[i])

    def test_extent(self):
        self.assertListEqual(list(self.f0d.extent), [])
        self.assertListEqual(list(self.f1d.extent), [0, 1])
        self.f1d.extent = [3.3, 5.5]
        self.assertListEqual(list(self.f1d.extent), [3.3, 5.5])
        self.assertListEqual(list(self.f2d.extent), [0, 1, 0, 1])
        self.f2d.extent = [3.3, 5.5, 7.7, 9.9]
        self.assertListEqual(list(self.f2d.extent), [3.3, 5.5, 7.7, 9.9])
        self.assertListEqual(list(self.f3d.extent), [0, 1, 0, 1, 0, 1])
        self.f3d.extent = [3.3, 5.5, 7.7, 9.9, 11.1, 13.3]
        self.assertListEqual(list(self.f3d.extent), [3.3, 5.5, 7.7, 9.9, 11.1,13.3])

    def test_dimensions(self):
        self.assertEqual(self.fempty.dimensions, -1)
        self.assertEqual(self.f0d.dimensions, 0)
        self.assertEqual(self.f1d.dimensions, 1)
        self.assertEqual(self.f2d.dimensions, 2)
        self.assertEqual(self.f3d.dimensions, 3)

    def test_half_resolution(self):
        self.f1d.half_resolution('x')
        self.checkFieldConsistancy(self.f1d)
        self.f2d.half_resolution('x')
        self.checkFieldConsistancy(self.f2d)
        self.f2d.half_resolution('y')
        self.checkFieldConsistancy(self.f2d)
        self.f3d.half_resolution('x')
        self.checkFieldConsistancy(self.f3d)
        self.f3d.half_resolution('y')
        self.checkFieldConsistancy(self.f3d)
        self.f3d.half_resolution('z')
        self.checkFieldConsistancy(self.f3d)

    def test_autoreduce(self):
        self.f3d.autoreduce(maxlen=2)
        self.assertEqual(self.f3d.shape, (2, 2, 1))
        self.assertEqual(self.f3d.extent[0], 0)
        self.assertEqual(self.f3d.extent[1], 1)
        self.checkFieldConsistancy(self.f3d)


if __name__ == '__main__':
    unittest.main()
