#!/usr/bin/env python2

import unittest
import epochsdftools as ep

class TestSingleSpeciesAnalyzer(unittest.TestCase):
    
    def setUp(self):
        pass
        
    def test_particleinfo_ion(self):
        self.assertEqual(ep.ParticleAnalyzer.retrieveparticleinfo('proton'), {'charge':1*ep.ParticleAnalyzer._qe, 'mass':1*1836.2*ep.ParticleAnalyzer._me, 'tracer':False, 'ejected':False, 'ision':True})
        self.assertEqual(ep.ParticleAnalyzer.retrieveparticleinfo('H1'), {'charge':1*ep.ParticleAnalyzer._qe, 'mass':1*1836.2*ep.ParticleAnalyzer._me, 'tracer':False, 'ejected':False, 'ision':True})
        self.assertEqual(ep.ParticleAnalyzer.retrieveparticleinfo('tracer_O3'), {'charge':3*ep.ParticleAnalyzer._qe, 'mass':16*1836.2*ep.ParticleAnalyzer._me, 'tracer':True, 'ejected':False, 'ision':True})
        self.assertEqual(ep.ParticleAnalyzer.retrieveparticleinfo('ejected_tracer_C4'), {'charge':4*ep.ParticleAnalyzer._qe, 'mass':12*1836.2*ep.ParticleAnalyzer._me, 'tracer':True, 'ejected':True, 'ision':True})
        self.assertEqual(ep.ParticleAnalyzer.retrieveparticleinfo('H1'), {'charge':1*ep.ParticleAnalyzer._qe, 'mass':1*1836.2*ep.ParticleAnalyzer._me, 'tracer':False, 'ejected':False, 'ision':True})
        self.assertEqual(ep.ParticleAnalyzer.retrieveparticleinfo('ionm3c7'), {'charge':7*ep.ParticleAnalyzer._qe, 'mass':3*1836.2*ep.ParticleAnalyzer._me, 'tracer':False, 'ejected':False, 'ision':True})
        self.assertEqual(ep.ParticleAnalyzer.retrieveparticleinfo('ionc3m7'), {'charge':3*ep.ParticleAnalyzer._qe, 'mass':7*1836.2*ep.ParticleAnalyzer._me, 'tracer':False, 'ejected':False, 'ision':True})
        self.assertEqual(ep.ParticleAnalyzer.retrieveparticleinfo('tracer_ionc9m5'), {'charge':9*ep.ParticleAnalyzer._qe, 'mass':5*1836.2*ep.ParticleAnalyzer._me, 'tracer':True, 'ejected':False, 'ision':True})
        self.assertEqual(ep.ParticleAnalyzer.retrieveparticleinfo('tracer_ejected_Au27a'), {'charge':27*ep.ParticleAnalyzer._qe, 'mass':197*1836.2*ep.ParticleAnalyzer._me, 'tracer':True, 'ejected':True, 'ision':True})
        self.assertEqual(ep.ParticleAnalyzer.retrieveparticleinfo('ejected_tracer_Au27'), {'charge':27*ep.ParticleAnalyzer._qe, 'mass':197*1836.2*ep.ParticleAnalyzer._me, 'tracer':True, 'ejected':True, 'ision':True})


    def test_particleinfo_electron(self):
        self.assertEqual(ep.ParticleAnalyzer.retrieveparticleinfo('Elektron'), {'charge':-1*ep.ParticleAnalyzer._qe, 'mass':1*ep.ParticleAnalyzer._me, 'tracer':False, 'ejected':False, 'ision':False})
        self.assertEqual(ep.ParticleAnalyzer.retrieveparticleinfo('Electronx'), {'charge':-1*ep.ParticleAnalyzer._qe, 'mass':1*ep.ParticleAnalyzer._me, 'tracer':False, 'ejected':False, 'ision':False})
        self.assertEqual(ep.ParticleAnalyzer.retrieveparticleinfo('Electron2'), {'charge':-1*ep.ParticleAnalyzer._qe, 'mass':1*ep.ParticleAnalyzer._me, 'tracer':False, 'ejected':False, 'ision':False})
        self.assertEqual(ep.ParticleAnalyzer.retrieveparticleinfo('ElectronAu2'), {'charge':-1*ep.ParticleAnalyzer._qe, 'mass':1*ep.ParticleAnalyzer._me, 'tracer':False, 'ejected':False, 'ision':False})
        self.assertEqual(ep.ParticleAnalyzer.retrieveparticleinfo('ejected_ElectronAu2'), {'charge':-1*ep.ParticleAnalyzer._qe, 'mass':1*ep.ParticleAnalyzer._me, 'tracer':False, 'ejected':True, 'ision':False})
        
        
    def test_particleinfo_extra(self):
        self.assertEqual(ep.ParticleAnalyzer.retrieveparticleinfo('tracer_blahh_Au27x'), {'charge':27*ep.ParticleAnalyzer._qe, 'mass':197*1836.2*ep.ParticleAnalyzer._me, 'tracer':True, 'ejected':False, 'ision':True, 'blahh': True})
        self.assertEqual(ep.ParticleAnalyzer.retrieveparticleinfo('tracer_blahh_electronHe2b'), {'charge':-1*ep.ParticleAnalyzer._qe, 'mass':1*ep.ParticleAnalyzer._me, 'tracer':True, 'ejected':False, 'ision':False, 'blahh': True})



if __name__ == '__main__':
    unittest.main()
