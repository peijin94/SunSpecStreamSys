"""Physical secants and strict request handling for spectrum measurements."""
import importlib
import importlib.util
import math
import unittest

class MeasurementTests(unittest.TestCase):
    def measure(self,payload):
        self.assertIsNotNone(importlib.util.find_spec('spectrum_measure'),'Measurement API core must exist')
        return importlib.import_module('spectrum_measure').measure_spectrum(payload)

    def test_saito_known_radius_track_recovers_drift_and_radial_speed(self):
        frequency=lambda r:8.93e-3*math.sqrt(1.36e6*r**-2.14+1.68e8*r**-6.13)
        f1,f2=frequency(1.5),frequency(1.6)
        result=self.measure({'points':[[1000,f1],[1020,f2]]})
        self.assertAlmostEqual(result['points'][0]['r_sun'],1.5,places=10)
        self.assertAlmostEqual(result['points'][1]['r_sun'],1.6,places=10)
        segment=result['segments'][0]
        self.assertAlmostEqual(segment['drift_mhz_s'],(f2-f1)/20,places=10)
        self.assertAlmostEqual(segment['r_sun'],1.55,places=10)
        self.assertEqual(segment['time'],1010)
        self.assertAlmostEqual(segment['radial_velocity_km_s'],.1*696000/20,places=6)
        self.assertAlmostEqual(segment['speed_km_s'],.1*696000/20,places=6)
        self.assertEqual(result['harmonic'],1)

    def test_harmonic_and_density_scale_recover_the_same_synthetic_track(self):
        frequency=lambda r:2*8.93e-3*math.sqrt(3*(1.36e6*r**-2.14+1.68e8*r**-6.13))
        result=self.measure({'points':[[1000,frequency(1.7)],[1010,frequency(1.8)]],
                             'harmonic':2,'density_scale':3})
        self.assertAlmostEqual(result['points'][0]['r_sun'],1.7,places=10)
        self.assertAlmostEqual(result['points'][1]['r_sun'],1.8,places=10)
        self.assertEqual(result['harmonic'],2)
        self.assertEqual(result['density_scale'],3)

    def test_invalid_or_ambiguous_points_are_rejected_not_clipped(self):
        invalid=[None,[],{}, {'points':[]},{'points':[[1,30]]},
            {'points':[[1,30],[1,20]]},{'points':[[1,30],[2,0]]},
            {'points':[[True,30],[2,20]]},{'points':[['1',30],[2,20]]},
            {'points':[[1,float('nan')],[2,20]]},{'points':[[1,30],[float('inf'),20]]},
            {'points':[[1,30,4],[2,20]]},{'points':[[1,30],[2,20]],'harmonic':True},
            {'points':[[1,30],[2,20]],'harmonic':3},
            {'points':[[1,30],[2,20]],'density_scale':0},
            {'points':[[1,30],[2,20]],'model':'invalid'},
            {'points':[[1,300],[2,200]]},{'points':[[1,.0001],[2,.0002]]},
            {'points':[[n,30] for n in range(129)]},
            {'points':[[1,30],[2,20]],'extra':1},
            {'points':[[10**1000,30],[2,20]]},
            {'points':[[1,30],[2,20]],'density_scale':10**1000},
            {'points':[[0,30],[5e-324,20]]}]
        for payload in invalid:
            with self.subTest(payload=payload), self.assertRaises(ValueError):self.measure(payload)

    def test_supported_models_match_the_reference_density_at_known_radii(self):
        densities={'leblanc98':lambda r:3.3e5*r**-2+4.1e6*r**-4+8e7*r**-6,
                   'newkirk':lambda r:4.2e4*10**(4.32/r)}
        for model,density in densities.items():
            with self.subTest(model=model):
                result=self.measure({'points':[[20,8.93e-3*math.sqrt(density(1.5))],
                                               [10,8.93e-3*math.sqrt(density(1.6))]],'model':model})
                self.assertEqual(result['model'],model)
                self.assertAlmostEqual(result['points'][0]['r_sun'],1.6,places=10)
                self.assertAlmostEqual(result['points'][1]['r_sun'],1.5,places=10)
                self.assertLess(result['segments'][0]['radial_velocity_km_s'],0)
                self.assertGreater(result['segments'][0]['speed_km_s'],0)

    def test_superluminal_result_is_warned_not_silently_clipped(self):
        result=self.measure({'points':[[1000,80],[1000.001,20]]})
        self.assertGreater(result['segments'][0]['speed_km_s'],299800)
        self.assertTrue(any('light' in warning for warning in result['warnings']))
        self.assertTrue(result['assumptions'])

if __name__=='__main__':unittest.main()
