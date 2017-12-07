#!/usr/bin/env python

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



class TestKspace(unittest.TestCase):

    def setUp(self):
        pass

    def test_kspace(self):
        pp.chooseCode('dummy')
        for d in (1,2,3):
            dr = pp.readDump(10000, dimensions=d)
            kspace = pp.helper.kspace("Ex", fields=dict(Ex=dr.Ex(), By=dr.By(), Bz=dr.Bz()))

        omega = pp.helper.omega_yee_factory([ax.grid[1] - ax.grid[0] for ax in dr.Ey().axes], 1e-10)
        kspace = pp.helper.kspace("Ex", fields=dict(Ex=dr.Ex(), By=dr.By(), Bz=dr.Bz()), omega_func=omega)

        kspace = pp.helper.kspace("Ex", fields=dict(Ex=dr.Ex(), By=dr.By(), Bz=dr.Bz()), extent=[0, 0.5, 0, 0.5, 0, 0.5])

    def test_kspace_epoch_like(self):
        pp.chooseCode('dummy')
        for d in (1,2,3):
            dr = pp.readDump(10000, dimensions=d)
            fields = dict(Ex=dr.Ex(), By=dr.By(), Bz=dr.Bz())
            for k, v in fields.items():
                print(k, v.extent)
            kspace = pp.helper.kspace_epoch_like("Ex", fields, 0.1)

    def test_kspace_propagate(self):
        pp.chooseCode('dummy')
        dr = pp.readDump(10000, dimensions=2)
        kspace = pp.helper.kspace("Ex", fields=dict(Ex=dr.Ex(), By=dr.By(), Bz=dr.Bz()))
        pp.helper.kspace_propagate(kspace, 0.1, moving_window_vect=(1,0))

    def test_time_profile_at_plane(self):
        dr = pp.readDump(10000, dimensions=3)
        kspace = pp.helper.kspace("Ex", fields=dict(Ex=dr.Ex(), By=dr.By(), Bz=dr.Bz()))
        complex_ex = kspace.fft()
        pp.helper.time_profile_at_plane(complex_ex, axis='z', value=1.0, dir=-1)


class TestHelper(unittest.TestCase):
    def setUp(self):
        pass

    def test_fftpadsize(self):
        fft_padsize = pp.helper.FFTW_Pad(fftsize_max=10000, factors=(2, 3, 5, 7, 11, 13))
        self.assertEqual(fft_padsize(223), 224)
        self.assertEqual(fft_padsize(224), 224)
        self.assertEqual(fft_padsize(250), 250)
        self.assertEqual(fft_padsize(251), 252)


if __name__ == '__main__':
    unittest.main()
