#!/usr/bin/env python2

import unittest
import postpic.datahandling as dh
import postpic.helper as helper
import numpy as np
import copy
import scipy.integrate
import pkg_resources as pr


class TestAxis(unittest.TestCase):

    def setUp(self):
        self.ax = dh.Axis(name='name', unit='unit', extent=[-1,1], n=101)

    def test_simple(self):
        self.assertEqual(self.ax.name, 'name')
        self.assertEqual(self.ax.unit, 'unit')

    def test_grid_general(self):
        self.assertEqual(self.ax.extent, [-1, 1])
        self.assertEqual(len(self.ax), 101)
        self.assertEqual(len(self.ax.grid), 101)
        self.assertEqual(len(self.ax.grid_node), 102)
        self.assertEqual(self.ax.grid_node[0], -1)
        self.assertEqual(self.ax.grid_node[-1], 1)
        self.assertTrue(self.ax.islinear())
        self.assertTrue(self.ax.islinear(force=True))

    def test_initiate(self):
        ax = dh.Axis(extent=(-1,1), n=101)
        self.assertEqual(ax.name, '')
        self.assertEqual(ax.unit, '')

    def test_getitem(self):
        # even number of grid points
        ax = dh.Axis(extent=(-1,1), n=100)
        ax = ax[0.0:1.0]
        self.assertEqual(len(ax), 50)
        self.assertEqual(ax.grid_node[0], 0)
        # odd number of grid points
        ax = dh.Axis(extent=(-1,1), n=101)
        ax = ax[-0.01: 1]
        self.assertEqual(len(ax), 51)
        self.assertEqual(ax.grid[0], 0)

    def test_half_resolution(self):
        # even number of grid points
        ax = dh.Axis(extent=(10, 20), n=100)
        ax = ax.half_resolution()
        self.assertEqual(len(ax), 50)
        # odd number of grid points
        ax = dh.Axis(extent=(10, 20), n=101)
        ax = ax.half_resolution()
        self.assertEqual(len(ax), 50)

    def test_extent(self):
        ax = dh.Axis(grid_node = [1, 2.7])
        self.assertEqual(ax.extent, [1, 2.7])

    def test_grid(self):
        ax = dh.Axis(grid = [5, 6])
        self.assertEqual(ax.grid[0], 5)
        self.assertEqual(ax.grid[1], 6)

    def test_grid_node(self):
        ax = dh.Axis(grid_node = [5, 6])
        self.assertEqual(ax.grid[0], 5.5)
        self.assertTrue(ax.grid_node == [5, 6])


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

        x, y = helper.meshgrid(np.linspace(0,2*np.pi,100), np.linspace(0,2*np.pi,100), indexing='ij', sparse=True)
        self.f2d_fine = dh.Field(np.sin(x)*np.cos(y))

    def assertClose(self, a, b):
        self.assertTrue(np.all(np.isclose(a, b)))

    def assertAllEqual(self, a, b):
        self.assertTrue(np.all( a == b ))

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

    def test_transpose(self):
        f3d_T = self.f3d.T
        c, b, a = self.f3d.shape
        self.assertEqual((a,b,c), f3d_T.shape)
        self.assertEqual(len(self.f3d.axes[0]), len(f3d_T.axes[2]))
        self.assertEqual(len(self.f3d.axes[1]), len(f3d_T.axes[1]))
        self.assertEqual(len(self.f3d.axes[2]), len(f3d_T.axes[0]))

        f3d_T = self.f3d.transpose(0,2,1)
        a, c, b = self.f3d.shape
        self.assertEqual((a,b,c), f3d_T.shape)
        self.assertEqual(len(self.f3d.axes[0]), len(f3d_T.axes[0]))
        self.assertEqual(len(self.f3d.axes[1]), len(f3d_T.axes[2]))
        self.assertEqual(len(self.f3d.axes[2]), len(f3d_T.axes[1]))

    def test_swapaxes(self):
        f2d_swapped = self.f2d.swapaxes(0,1)
        b, a = self.f2d.shape
        self.assertEqual((a,b), f2d_swapped.shape)
        self.assertEqual(self.f2d.axes[0].extent, f2d_swapped.axes[1].extent)
        self.assertEqual(self.f2d.axes[1].extent, f2d_swapped.axes[0].extent)
        self.assertEqual(len(self.f2d.axes[0]), len(f2d_swapped.axes[1]))
        self.assertEqual(len(self.f2d.axes[1]), len(f2d_swapped.axes[0]))

    def test_autocutout(self):
        f2d_f_c = self.f2d_fine.autocutout(fractions=(0.01, 0.02))
        self.assertEqual(f2d_f_c.shape, (98,100))

    def test_fft_autopad(self):
        s = self.f2d_fine[1:,5:].fft_autopad(fft_padsize=helper.FFTW_Pad(fftsize_max=10000, factors=(2, 3, 5, 7, 11, 13)))
        self.assertEqual(s.shape, (99,96))
        s = self.f2d_fine[1:,5:].fft_autopad(fft_padsize=helper.fft_padsize_power2)
        self.assertEqual(s.shape, (128,128))

    def test_conjugate_grid(self):
        f1d_grid = self.f1d.grid
        f1d_grid2 = self.f1d.ensure_frequency_domain()._conjugate_grid()
        self.assertTrue(np.all(np.isclose(f1d_grid, f1d_grid2[0])))

        f2d_grid = self.f2d.grid
        f2d_grid2 = self.f2d.fft()._conjugate_grid()
        for k, v in f2d_grid2.items():
            print(f2d_grid[k], v)
            self.assertTrue(np.all(np.isclose(f2d_grid[k], v)))

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

        print('self.f2d.axes', self.f2d.axes)
        f = self.f2d.fft()
        print('f.axes', f.axes)
        dk = [ax.grid[1]-ax.grid[0] for ax in f.axes]
        f2 = f.shift_grid_by(dk)
        print('f2.axes', f2.axes)
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

    def test_map_coordinates(self):
        a = self.f2d.integrate().matrix

        th_axis = dh.Axis(grid = np.linspace(0, 2*np.pi, 100))

        r_axis = dh.Axis(grid = np.linspace(0, 1.5, 100))

        # this calculates numerical approximation of jacobi determinant and thus also tests
        # helper.jac_det and
        # helper.approx_jacobian
        polar = self.f2d.map_coordinates([th_axis, r_axis], helper.polar2linear)
        b = polar.integrate().matrix

        print(a, b)
        self.assertTrue(np.isclose(a, b, rtol=0.01))

    def test_map_coordinates_2(self):
        orig = self.f2d_fine

        # This uses analytical Jacobian determinant
        polar = orig.topolar(extent=(0, 2*np.pi, 0, 1.5), shape=(500,500))

        # This numerically calculates the jacobian determinant of the inverse transform
        backtransformed = polar.map_coordinates(orig.axes, transform=helper.linear2polar)

        # Some border pixels may be wrong due to third order interpolation in combination with
        # clipping, so compare only the inner part
        a = orig[5:-5, 5:-5].matrix
        b = backtransformed[5:-5, 5:-5].matrix
        relerr = 2*abs(a-b)/(a+b)

        i = np.argmax(relerr)
        i = np.unravel_index(i, relerr.shape)
        print(i, a[i], b[i], relerr[i])
        maxrelerr = relerr[i]

        testrelerr = 0.001

        abserr = abs(a-b) - testrelerr*(a+b)/2

        i = np.argmax(abserr)
        i = np.unravel_index(i, abserr.shape)
        print(i, a[i], b[i], abserr[i])

        self.assertTrue(np.all(np.isclose(a, b, rtol=0.001, atol=0.001)))


    def test_topolar(self):
        a = self.f2d.integrate().matrix

        #test if topolar runs with default args
        polar = self.f2d.topolar()

        # this uses the analytically known jacobi determinant of the transform in question
        polar = self.f2d.topolar(extent=(0, 2*np.pi, 0, 1.5), shape=(100,100))
        b = polar.integrate().matrix

        print(a, b)
        self.assertTrue(np.isclose(a, b, rtol=0.01))

    def test_map_axis_grid(self):
        a = self.f2d.integrate().matrix

        # this calculates numerical approx of derivative and tests
        # helper.approx_1d_jacobian_det
        b = self.f2d.map_axis_grid(1, lambda x: 12*x).integrate().matrix
        print(a, b)
        self.assertTrue(np.isclose(a, b))

        a = self.f2d.integrate().matrix

        # this calculates numerical approx of derivative and tests
        # helper.approx_1d_jacobian_det
        b = self.f2d.map_axis_grid(1, lambda x: 20*x**2).integrate().matrix
        print(a, b)
        self.assertTrue(np.isclose(a, b, rtol=0.04))

    def test_integrate(self):
        print('start f1d.integrate')
        a = self.f1d.integrate(method='constant')

        print('start f1d.mean * length')
        b = self.f1d.mean() * self.f1d.axes[0].physical_length

        print('type(a.matrix)', type(a.matrix))
        print('type(b.matrix)', type(b.matrix))

        self.assertTrue(np.isclose(a, b))

        b = self.f2d_fine.integrate(method=scipy.integrate.simps)
        c = self.f2d_fine.integrate(method=scipy.integrate.trapz)

        self.assertTrue(np.isclose(b, 0))
        self.assertTrue(np.isclose(c, 0))

    def test_arithmetic(self):
        c1d = self.f1d + 3j*self.f1d
        i1d = c1d.imag
        a1d = c1d.angle
        cc1d = c1d.conj()
        self.assertClose(cc1d.matrix, (c1d-2j*i1d).matrix)

    def test_operators(self):
        # test unary operators
        a = self.f2d

        self.assertAllEqual((-a).matrix, -(a.matrix))

        # f1d_neg contains negative numbers
        b = a - np.mean(a.matrix)

        # avoid runtime errors
        b.matrix[b==0] = 1

        self.assertAllEqual(abs(b).matrix, abs(b.matrix))

        #test binary operators
        self.assertAllEqual((a+b).matrix, a.matrix + b.matrix)
        self.assertAllEqual((a*b).matrix, a.matrix * b.matrix)
        self.assertAllEqual((a/b).matrix, a.matrix / b.matrix)
        self.assertAllEqual((b**a).matrix, b.matrix ** a.matrix)
        self.assertAllEqual((b<=a).matrix, b.matrix <= a.matrix)

        #test other ufuncs
        self.assertAllEqual(np.sin(a).matrix, np.sin(a.matrix))
        self.assertAllEqual(np.exp(a).matrix, np.exp(a.matrix))

    def test_numpy_methods_1(self):
        a = np.ptp(self.f2d)
        self.assertEqual(a.matrix, 19)

        b = np.std(self.f2d)
        self.assertTrue(np.isclose(b.matrix, 5.766281297335398))

    @unittest.skipIf(pr.parse_version(np.__version__) < pr.parse_version("1.13"),
                     "This behaviour is not supported for numpy older than 1.13")
    def test_numpy_methods_2(self):
        a = np.mean(self.f2d, keepdims=True)
        self.assertEqual(a.matrix[0,0], 9.5)

        np.std(self.f2d, out=a, keepdims=True)
        self.assertEqual(a.matrix[0,0], 5.766281297335398)

    @unittest.skipIf(pr.parse_version(np.__version__) < pr.parse_version("1.13"),
                     "This behaviour is not supported for numpy older than 1.13")
    def test_numpy_ufuncs(self):
        a = np.add.reduce(self.f2d, axis=1)
        self.assertTrue(a.axes[0] is self.f2d.axes[0])
        self.assertAllEqual(a.matrix, np.add.reduce(self.f2d.matrix, axis=1))

        b = np.multiply.outer(self.f1d, self.f2d)
        self.assertEqual(b.axes, self.f1d.axes + self.f2d.axes)
        self.assertAllEqual(b.matrix, np.multiply.outer(self.f1d.matrix, self.f2d.matrix))


if __name__ == '__main__':
    unittest.main()
