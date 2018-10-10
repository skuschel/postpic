#!/usr/bin/env python

import unittest
import postpic as pp
import numpy as np

class TestMultiSpecies(unittest.TestCase):

    def setUp(self):
        pp.chooseCode('dummy')
        self.dr = pp.readDump(10000)
        self.p = pp.MultiSpecies(self.dr, 'electron')

    def test_dummyreader(self):
        self.assertEqual(self.dr.timestep(), 10000)

    def test_pa(self):
        self.assertEqual(len(self.p), 10000)

    def test_mean(self):
        self.assertAlmostEqual(self.p.mean('x'), -0.0184337)
        self.assertAlmostEqual(self.p.mean('x', weights='beta'), -0.01708324558)

    def test_var(self):
        self.assertAlmostEqual(self.p.var('x'), 0.97526797)
        self.assertAlmostEqual(self.p.var('y'), 0.98615759)
        self.assertRaises(KeyError, self.p.var, 'z')

    def test_quantile(self):
        self.assertAlmostEqual(self.p.quantile('x', 0.4), -0.26393734)
        self.assertAlmostEqual(self.p.quantile('x', 0.6, weights='gamma'), 0.2220709288)
        self.assertTrue(np.allclose(self.p.quantile('x', [0.4]), -0.26393734))
        print(self.p.quantile('x', [0.4, 0.5, 0.6, 0.4]))
        self.assertTrue(np.allclose(self.p.quantile('x', [0.4, 0.5, 0.6, 0.4]),
                                    [-0.26393735, -0.02714493, 0.21839707, -0.26393735]
))
    def test_operatorOverloading(self):
        p = self.p
        lenp = len(p)
        p2 = p + p
        self.assertEqual(lenp, len(p2)/2)
        f = p.createField('x', 'y')
        f2 = p2.createField('x', 'y')
        self.assertTrue(np.allclose(f, f2/2))
        # ensure p and p2 are still independent
        pf = p.filter('x>0')
        self.assertEqual(lenp, len(p2)/2)
        p2f = p2.filter('x<0')
        self.assertEqual(lenp, len(p))

    def test_compress(self):
        def cf(ms):
            return ms('x>0')
        p2 = self.p.compressfn(cf)
        lencf = len(p2)
        p3 = self.p.filter('x>0')
        lenf = len(p3)
        self.assertEqual(lencf, lenf)

    def test_compress_ids(self):
        ids = [1,5,10]
        p2 = self.p.compress(ids)
        lenc = len(p2)
        self.assertEqual(lenc, 3)

    def test_repr(self):
        print(self.p)
        print(self.p._ssas[0])

    def test_multifilter(self):
        l0 = len(self.p)
        s1 = self.p.filter('y>0')
        l1 = len(s1)
        s2 = s1.filter('x>0')
        l2 = len(s2)
        s3 = s2.filter('px>0')
        l3 = len(s3)
        self.assertEqual(len(self.p), l0)
        self.assertEqual(len(s1), l1)
        self.assertEqual(len(s2), l2)
        self.assertEqual(len(s3), l3)

    def test_invert(self):
        l0 = len(self.p)
        sgtr = self.p.filter('y>0')
        lgtr = len(sgtr)
        sless = self.p.filter('y<=0')
        lless = len(sless)
        self.assertEqual(lgtr + lless, l0)
        self.assertEqual(lgtr, len(~sless))
        self.assertEqual(lless, len(~sgtr))
        self.assertEqual(lgtr, len(~~sgtr))

    def test_invert2(self):
        l0 = len(self.p)
        self.assertEqual(len(~self.p), 0)
        self.assertEqual(len(~~self.p), l0)

if __name__ == '__main__':
    unittest.main()
