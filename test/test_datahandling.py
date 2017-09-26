#!/usr/bin/env python2

import unittest
import postpic.datahandling as dh
import numpy as np
import copy


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

    def test_getitem(self):
        # even number of grid points
        self.ax.setextent((-1, 1), 100)
        ax = self.ax[0.0:1.0]
        self.assertEqual(len(ax), 50)
        self.assertEqual(ax.grid_node[0], 0)
        # odd number of grid points
        self.ax.setextent((-1, 1), 101)
        ax = self.ax[-0.01: 1]
        self.assertEqual(len(ax), 51)
        self.assertEqual(ax.grid[0], 0)

    def test_half_resolution(self):
        # even number of grid points
        self.ax.setextent((10, 20), 100)
        ax = self.ax.half_resolution()
        self.assertEqual(len(ax), 50)
        # odd number of grid points
        self.ax.setextent((10, 20), 101)
        ax = self.ax.half_resolution()
        self.assertEqual(len(ax), 50)

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
        f = self.f1d.half_resolution('x')
        self.checkFieldConsistancy(f)
        f2 = self.f2d.half_resolution('x')
        self.checkFieldConsistancy(f2)
        f2 = f2.half_resolution('y')
        self.checkFieldConsistancy(f2)
        f3 = self.f3d.half_resolution('x')
        self.checkFieldConsistancy(f3)
        f3 = f3.half_resolution('y')
        self.checkFieldConsistancy(f3)
        f3 = f3.half_resolution('z')
        self.checkFieldConsistancy(f3)

    def test_autoreduce(self):
        f = self.f3d.autoreduce(maxlen=2)
        self.assertEqual(f.shape, (2, 2, 1))
        self.assertEqual(f.extent[0], 0)
        self.assertEqual(f.extent[1], 1)
        self.checkFieldConsistancy(f)

    def test_slicing(self):
        self.assertEqual(self.f1d[0.15:0.75].matrix.shape, (6,))
        self.assertEqual(self.f1d[5].matrix.shape, (1,))

        self.assertEqual(self.f2d[0.5:, :].matrix.shape, (2, 5))

        self.assertEqual(self.f3d[0.5:, :, 0.5].matrix.shape, (2, 5, 1))
        self.assertEqual(self.f3d[0.5:, :, 0.5].squeeze().matrix.shape, (2, 5))

    def test_fourier_inverse(self):
        f1d_orig = copy.deepcopy(self.f1d)
        self.f1d.fft()
        self.f1d.fft()
        self.assertTrue(np.all(np.isclose(f1d_orig.matrix, self.f1d.matrix)))
        self.assertTrue(np.all(np.isclose(f1d_orig.grid, self.f1d.grid)))

        f2d_orig = copy.deepcopy(self.f2d)
        self.f2d.fft()
        self.f2d.fft()
        self.assertTrue(np.all(np.isclose(f2d_orig.matrix, self.f2d.matrix)))
        self.assertTrue(
            all(
                np.all(np.isclose(f2d_orig.grid[i], self.f2d.grid[i]))
                for i in (0, 1)
                )
            )

        f3d_orig = copy.deepcopy(self.f3d)
        self.f3d.fft()
        self.f3d.fft()
        self.assertTrue(np.all(np.isclose(f3d_orig.matrix, self.f3d.matrix)))
        self.assertTrue(
            all(
                np.all(np.isclose(f3d_orig.grid[i], self.f3d.grid[i]))
                for i in (0, 1, 2)
                )
            )

    def test_fourier_shift_spatial_domain(self):
        f1d_orig = copy.deepcopy(self.f1d)
        dx = [ax.grid[1]-ax.grid[0] for ax in self.f1d.axes]
        f = self.f1d.shift_grid_by(dx)
        self.assertTrue(np.all(np.isclose(np.roll(f1d_orig.matrix, -1), f.matrix.real)))

        f2d_orig = copy.deepcopy(self.f2d)
        dx = [ax.grid[1]-ax.grid[0] for ax in self.f2d.axes]
        f = self.f2d.shift_grid_by([dx[0], 0])
        self.assertTrue(np.all(np.isclose(np.roll(f2d_orig.matrix, -1, axis=0), f.matrix.real)))

        self.f2d = copy.deepcopy(f2d_orig)
        f = self.f2d.shift_grid_by(dx)
        self.assertTrue(np.all(np.isclose(np.roll(
            np.roll(f2d_orig.matrix, -1, axis=0), -1, axis=1
            ), f.matrix.real)))

        f3d_orig = copy.deepcopy(self.f3d)
        f = self.f3d.shift_grid_by([0.25, 0, 0])
        self.assertTrue(np.all(np.isclose(np.roll(f3d_orig.matrix, -1, axis=0), f.matrix.real)))

    def test_fourier_shift_frequency_domain(self):
        f = self.f1d.fft()
        dk = f.grid[1] - f.grid[0]
        f2 = f.shift_grid_by([dk])
        self.assertTrue(np.all(np.isclose(np.roll(f.matrix, -1), f2.matrix)))
        self.assertTrue(f.matrix is not f2.matrix)

        f = self.f2d.fft()
        dk = [ax.grid[1]-ax.grid[0] for ax in f.axes]
        f2 = f.shift_grid_by(dk)
        self.assertTrue(np.all(np.isclose(np.roll(
            np.roll(f.matrix, -1, axis=0), -1, axis=1),
        f2.matrix)))
        self.assertTrue(f.matrix is not f2.matrix)

        f = self.f3d.fft()
        dk = [ax.grid[1]-ax.grid[0] for ax in f.axes]
        f3d_orig = copy.deepcopy(self.f3d)
        f2 = f.shift_grid_by(dk)
        self.assertTrue(np.all(np.isclose(
            np.roll(
                np.roll(
                    np.roll(f.matrix, -1, axis=0),
                -1, axis=1),
            -1, axis=2),
        f2.matrix)))
        self.assertTrue(f.matrix is not f2.matrix)

    def test_fourier_norm(self):
        fft = self.f1d.fft()
        I1 = np.sum(abs(fft.matrix)**2) * (fft.grid[1]-fft.grid[0])
        I2 = np.sum(abs(self.f1d.matrix)**2) * (self.f1d.grid[1]-self.f1d.grid[0])
        print(I1, I2)
        self.assertTrue(np.isclose(I1, I2))

        fft = self.f2d.fft()
        I1 = np.sum(abs(fft.matrix)**2) * (fft.grid[0][1]-fft.grid[0][0]) \
            * (fft.grid[1][1]-fft.grid[1][0])
        I2 = np.sum(abs(self.f2d.matrix)**2) * (self.f2d.grid[0][1]-self.f2d.grid[0][0]) \
            * (self.f2d.grid[1][1]-self.f2d.grid[1][0])
        print(I1, I2)
        self.assertTrue(np.isclose(I1, I2))

        fft = self.f3d.fft()
        I1 = np.sum(abs(fft.matrix)**2) * (fft.grid[0][1]-fft.grid[0][0]) \
            * (fft.grid[1][1]-fft.grid[1][0]) * (fft.grid[2][1]-fft.grid[2][0])
        I2 = np.sum(abs(self.f3d.matrix)**2) * (self.f3d.grid[0][1]-self.f3d.grid[0][0]) \
            * (self.f3d.grid[1][1]-self.f3d.grid[1][0]) * (self.f3d.grid[2][1]-self.f3d.grid[2][0])
        print(I1, I2)
        self.assertTrue(np.isclose(I1, I2))

    def test_pad(self):
        padded_1d = self.f1d.pad(2)
        self.assertEqual(padded_1d.shape, (14,))

        padded_1d = self.f1d.pad([[0.2, 2]])
        self.assertEqual(padded_1d.shape, (14,))

        padded_2d = self.f2d.pad([[0.2, 2], [0.1, 1]])
        self.assertEqual(padded_2d.shape, (7, 7))

        padded_2d = self.f2d.pad([[2], [3]])
        self.assertEqual(padded_2d.shape, (8, 11))
        self.assertTrue(np.all(np.isclose(padded_2d.grid[0][2:-2], self.f2d.grid[0])))

        backcut = padded_2d[padded_2d._extent_to_slices(self.f2d.extent)]
        self.assertTrue(np.all(np.isclose(backcut, self.f2d)))
        self.assertTrue(np.all(np.isclose(backcut.grid[0], self.f2d.grid[0])))
        self.assertTrue(np.all(np.isclose(backcut.grid[1], self.f2d.grid[1])))

    def test_topolar(self):
        polar = self.f2d.topolar()

    def test_integrate(self):
        print('start f1d.integrate')
        a = self.f1d.integrate()

        print('start f1d.mean * length')
        b = self.f1d.mean() * self.f1d.axes[0].physical_length

        print('type(a.matrix)', type(a.matrix))
        print('type(b.matrix)', type(b.matrix))

        self.assertTrue(np.isclose(a, b))

    def test_arithmetic(self):
        c1d = self.f1d + 3j*self.f1d
        i1d = c1d.imag
        a1d = c1d.angle
        cc1d = c1d.conj()


if __name__ == '__main__':
    unittest.main()
