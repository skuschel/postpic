#!/usr/bin/env python2

import unittest
import postpic as pp

class TestSpeciesIdentifier(unittest.TestCase):
    
    def setUp(self):
        pass

    def checke(self, data, m, c, eject, tracer, ision):
        pc = pp.PhysicalConstants
        self.assertEqual(data['mass'], m * pc.me)
        self.assertEqual(data['charge'], c * pc.qe)
        self.assertEqual(data['ejected'], eject)
        self.assertEqual(data['tracer'], tracer)
        self.assertEqual(data['ision'], ision)

    def checkion(self, data, m, c, eject, tracer, ision):
        self.checke(data, 1836.2 * m, c, eject, tracer, ision)

    def test_identifyspecies_ion(self):
        idfy = pp.identifyspecies
        self.checkion(idfy('proton'), 1, 1, False, False, True)
        self.checkion(idfy('H1'), 1, 1, False, False, True)
        self.checkion(idfy('tracer_O3'), 16, 3, False, True, True)
        self.checkion(idfy('ejected_tracer_C4'), 12, 4, True, True, True)
        self.checkion(idfy('ionm3c7'), 3, 7, False, False, True)
        self.checkion(idfy('ionm30c70xx5'), 30, 70, False, False, True)
        self.checkion(idfy('tracer_ejected_Au27a'), 197, 27, True, True, True)
        self.checkion(idfy('ejected_tracer_Au27'), 197, 27, True, True, True)
        self.checkion(idfy('tracer_blahh_Au27x'), 197, 27, False, True, True)

    def test_identifyspecies_electron(self):
        idfy = pp.identifyspecies
        self.checke(idfy('Elektron'), 1, -1, False, False, False)
        self.checke(idfy('Elektronx'), 1, -1, False, False, False)
        self.checke(idfy('Elektron2'), 1, -1, False, False, False)
        self.checke(idfy('ElektronAu2'), 1, -1, False, False, False)
        self.checke(idfy('ejected_ElektronAu2'), 1, -1, True, False, False)
        self.checke(idfy('tracer_blahh_electronHe2b'), 1, -1, False, True, False)
        self.checke(idfy('tracer_blahh_elecHe2b'), 1, -1, False, True, False)
        self.checke(idfy('tracer_blahh_HeElec2b'), 1, -1, False, True, False)
        self.checke(idfy('Elec'), 1, -1, False, False, False)
        self.checke(idfy('Elec2x'), 1, -1, False, False, False)
        self.checke(idfy('Electrons'), 1, -1, False, False, False)

    def test_identifyspecies_praefix(self):
        x = pp.identifyspecies('a_b_c_xxx_tracer_h_w_33_He5_O3x2')
        self.assertEqual(x['a'], True)
        self.assertEqual(x['b'], True)
        self.assertEqual(x['c'], True)
        self.assertEqual(x['xxx'], True)
        self.assertEqual(x['h'], True)
        self.assertEqual(x['w'], True)
        self.assertEqual(x['33'], True)
        self.assertEqual(x['He5'], True)
        self.checkion(x, 16,3, False, True, True)

    def test_identifyspecies_extendedsyntax(self):
        idfy = pp.identifyspecies
        self.checkion(idfy('H'), 1, 0, False, False, True)
        self.checkion(idfy('HPlus'), 1, 1, False, False, True)
        self.checkion(idfy('H1Plus'), 1, 1, False, False, True)
        #self.checkion(idfy('Hplus'), 1, 1, False, False, True)
        self.checkion(idfy('HePlusPlus'), 4, 2, False, False, True)
        self.checkion(idfy('He2Plus'), 4, 2, False, False, True)
        self.checkion(idfy('HPlus'), 1, 1, False, False, True)
        self.checke(idfy('HElectron'), 1, -1, False, False, False)
        self.checke(idfy('HElectrons'), 1, -1, False, False, False)
        self.checke(idfy('HElec'), 1, -1, False, False, False)
        self.checke(idfy('HPlusElec'), 1, -1, False, False, False)
        self.checke(idfy('HePlusPluselectrons'), 1, -1, False, False, False)

    def test_falsefriends(self):
        idfy = pp.identifyspecies
        # careful: the last one is an uncharged ion
        self.checke(idfy('HElectron'), 1, -1, False, False, False)
        self.checke(idfy('HeElectron'), 1, -1, False, False, False)
        self.checke(idfy('Heelectron'), 1, -1, False, False, False)
        self.checkion(idfy('Helectron'), 4, 0, False, False, True)
        # Tiny differences may decide about Ne or electrons
        self.checke(idfy('NElectron'), 1, -1, False, False, False)
        self.checkion(idfy('Nelectron'), 20.2, 0, False, False, True)
        self.checke(idfy('Neelectron'), 1, -1, False, False, False)

if __name__ == '__main__':
    unittest.main()
