"""Solar context uses one UTC instant and the existing OVRO location."""
import unittest
from astropy.time import Time
from astropy.coordinates import AltAz, get_sun
import check_sun_elevation

class SolarPositionTests(unittest.TestCase):
    def test_position_returns_elevation_and_north_through_east_azimuth(self):
        instant=Time('2026-10-16T19:00:00',scale='utc')
        position=check_sun_elevation.get_sun_position(time=instant)
        expected=get_sun(instant).transform_to(AltAz(obstime=instant,location=check_sun_elevation.OVRO))
        self.assertAlmostEqual(position['elevation_deg'],float(expected.alt.deg),places=8)
        self.assertAlmostEqual(position['azimuth_deg'],float(expected.az.deg),places=8)
        self.assertAlmostEqual(position['elevation_deg'],check_sun_elevation.get_sun_elevation(time=instant),places=8)
        self.assertGreater(position['elevation_deg'],0)
        self.assertGreaterEqual(position['azimuth_deg'],0)
        self.assertLess(position['azimuth_deg'],360)

if __name__=='__main__':unittest.main()
