"""Cache consistency, polling isolation, spatial geometry, and cached HTTP replies."""
from datetime import datetime, timezone
import io
import json
import logging
import threading
import time
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

import numpy as np
from astropy.io import fits
from fastapi.testclient import TestClient
from radio_images import RadioImageCache, FREQUENCIES, ImagePlane, read_plane, solar_view, render_planes
from stream_receiver import StreamReceiver


class Reply(io.BytesIO):
    def __init__(self, body, stamp=None):
        super().__init__(body)
        self.headers = {'X-Observation-Timestamp': stamp}


class RadioImageTests(unittest.TestCase):
    def setUp(self):
        self.now = time.time()
        self.stamp = datetime.fromtimestamp(self.now, timezone.utc).strftime('%Y%m%dT%H%M%S')
        self.calls = []
        self.fail = None
        self.mismatched = None
        self.rendered = []
        self.cache = RadioImageCache('http://source:8080', opener=self.open,
                                    renderer=self.render, clock=lambda: self.now)

    def metadata(self):
        return {'timestamp': self.stamp, 'images': [
            {'target_mhz': target, 'freq_mhz': target + .1, 'date_obs': self.stamp}
            for target in FREQUENCIES]}

    def npz(self, target, **changes):
        header = fits.Header({'CTYPE1': 'HPLN-TAN', 'CTYPE2': 'HPLT-TAN',
                              'CUNIT1': 'arcsec', 'CUNIT2': 'arcsec',
                              'CRPIX1': 64.5, 'CRPIX2': 64.5, 'CRVAL1': 0., 'CRVAL2': 0.,
                              'CDELT1': 50., 'CDELT2': 50., 'RSUN_OBS': 960.})
        y, x = np.indices((128, 128))
        image = np.exp(-((x - 80) ** 2 + (y - 64) ** 2) / 30) * 1e6
        arrays = dict(image=image, target_mhz=target, freq_mhz=target + .1,
                      date_obs=self.stamp, bunit='K', header=header.tostring())
        arrays.update(changes)
        output = io.BytesIO()
        np.savez_compressed(output, **arrays)
        return output.getvalue()

    def open(self, url, timeout):
        self.calls.append(url)
        if self.fail and self.fail in url:
            raise OSError('source unavailable')
        if '/images?' in url:
            return Reply(json.dumps(self.metadata()).encode())
        target = int(url.split('/images/')[1].split('.')[0])
        return Reply(self.npz(target), '20200101T000000' if self.mismatched == target else self.stamp)

    def render(self, planes):
        self.rendered.append(planes)
        return tuple(b'png-' + str(plane.target).encode() for plane in planes)

    def test_complete_observation_is_published_atomically_and_served_without_io(self):
        self.assertEqual(self.cache.metadata()['status'], 'loading')
        self.cache.refresh_once()
        metadata = self.cache.metadata()
        self.assertEqual(metadata['status'], 'ready')
        self.assertEqual([row['target_mhz'] for row in metadata['images']], list(FREQUENCIES))
        self.assertEqual(metadata['poll_interval_seconds'], 60)
        self.assertEqual(len(self.calls), 5)
        for url in self.calls[1:]: self.assertIn('timestamp=' + self.stamp, url)
        count = len(self.calls)
        for _ in range(20):
            self.cache.metadata()
            for target in FREQUENCIES: self.assertEqual(self.cache.image(target)[0], b'png-' + str(target).encode())
        self.assertEqual(count, len(self.calls))
        self.assertAlmostEqual(float(self.rendered[0][0].image.max()), 1., places=5)

    def test_unchanged_observation_only_requests_metadata(self):
        self.cache.refresh_once()
        version = self.cache.metadata()['version']
        self.calls.clear()
        self.now += 60
        self.cache.refresh_once()
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(len(self.rendered), 1)
        self.assertEqual(self.cache.metadata()['version'], version)
        self.assertEqual(self.cache.metadata()['fetched_at'], self.now)

    def test_empty_upstream_window_reports_waiting_without_fabricated_images(self):
        def empty(*args, **kwargs):
            raise HTTPError('http://source/images', 404, 'No matching data', {}, None)
        self.cache.opener = empty
        self.cache.refresh_once()
        result = self.cache.metadata()
        self.assertEqual(result['status'], 'unavailable')
        self.assertEqual(result['images'], [])
        self.assertIn('Waiting for the next observation', result['message'])

    def test_partial_failure_keeps_all_four_previous_images(self):
        self.cache.refresh_once()
        old = self.cache.metadata()
        self.stamp = datetime.fromtimestamp(self.now + 60, timezone.utc).strftime('%Y%m%dT%H%M%S')
        self.fail = '/60.npz'
        self.cache.refresh_once()
        self.assertEqual(self.cache.metadata()['version'], old['version'])
        self.assertEqual(self.cache.metadata()['status'], 'stale')
        self.assertEqual(len(self.cache.metadata()['images']), 4)
        self.assertEqual(len(self.rendered), 1)

    def test_nearest_observation_substitution_is_rejected(self):
        self.mismatched = 75
        self.cache.refresh_once()
        self.assertEqual(self.cache.metadata()['status'], 'unavailable')
        self.assertEqual(self.cache.metadata()['images'], [])
        self.assertEqual(self.rendered, [])

    def test_previous_version_survives_atomic_replacement_then_expires(self):
        self.cache.refresh_once()
        old = self.cache.metadata()['version']
        for minutes in (1, 2):
            self.stamp = datetime.fromtimestamp(self.now + minutes * 60, timezone.utc).strftime('%Y%m%dT%H%M%S')
            self.cache.refresh_once()
            if minutes == 1: self.assertIsNotNone(self.cache.image(30, old))
        self.assertIsNone(self.cache.image(30, old))
        self.assertEqual(len(self.cache.snapshots), 2)

    def test_slow_fetch_does_not_hold_cache_lock_or_overlap_another_poll(self):
        self.cache.refresh_once()
        opened, release = threading.Event(), threading.Event()
        original = self.cache.opener
        def blocked(*args, **kwargs):
            opened.set(); release.wait(2)
            return original(*args, **kwargs)
        self.cache.opener = blocked
        worker = threading.Thread(target=self.cache.refresh_once)
        worker.start()
        self.assertTrue(opened.wait(1))
        start = time.monotonic()
        self.assertEqual(self.cache.metadata()['status'], 'ready')
        self.cache.refresh_once()
        self.assertLess(time.monotonic() - start, .1)
        release.set(); worker.join(2)
        self.assertFalse(worker.is_alive())

    def test_spatial_wcs_and_kelvin_conversion_are_preserved(self):
        plane = read_plane(self.npz(30), self.metadata()['images'][0])
        view = solar_view(plane, size=128)
        _, x = np.unravel_index(np.nanargmax(view), view.shape)
        self.assertGreater(x, 64)
        plane.header['CDELT1'] = -50.
        opposite = solar_view(plane, size=128)
        _, x = np.unravel_index(np.nanargmax(opposite), opposite.shape)
        self.assertLess(x, 64)
        for changes in ({'bunit': 'Jy'}, {'target_mhz': 45}, {'image': np.zeros(5)}, {'freq_mhz': 29.}):
            with self.assertRaises(ValueError): read_plane(self.npz(30, **changes), self.metadata()['images'][0])

    def test_rendered_images_are_real_pngs_with_identical_size(self):
        from PIL import Image
        planes = tuple(read_plane(self.npz(target), row) for target, row in zip(FREQUENCIES, self.metadata()['images']))
        pngs = render_planes(planes)
        self.assertEqual(len(pngs), 4)
        for png in pngs:
            with Image.open(io.BytesIO(png)) as image:
                self.assertEqual(image.format, 'PNG')
                self.assertEqual(image.size, (600, 500))

    def test_wider_field_preserves_emission_outside_previous_crop(self):
        header = fits.Header({'CTYPE1': 'HPLN-TAN', 'CTYPE2': 'HPLT-TAN',
                              'CUNIT1': 'arcsec', 'CUNIT2': 'arcsec',
                              'CRPIX1': 128.5, 'CRPIX2': 128.5, 'CRVAL1': 0., 'CRVAL2': 0.,
                              'CDELT1': 60., 'CDELT2': 60.})
        y, x = np.indices((256, 256))
        image = np.exp(-((x - 202.5) ** 2 + (y - 127.5) ** 2) / 8)
        plane = ImagePlane(30, 30.1, self.stamp, image, header)
        self.assertLess(np.nanmax(solar_view(plane, extent=3000)), .01)
        view = solar_view(plane)
        row, column = np.unravel_index(np.nanargmax(view), view.shape)
        self.assertGreater(view[row, column], .9)
        self.assertAlmostEqual(-6000 + (column + .5) * 12000 / 512, 4500, delta=30)
        self.assertAlmostEqual(-6000 + (row + .5) * 12000 / 512, 0, delta=30)

    def test_changed_render_profile_changes_immutable_image_version(self):
        self.cache.refresh_once()
        previous = self.cache.metadata()['version']
        other = RadioImageCache('http://source:8080', opener=self.open,
                                renderer=self.render, clock=lambda: self.now)
        with patch('radio_images.RENDER_PROFILE', b'previous-narrow-field'):
            other.refresh_once()
        self.assertNotEqual(previous, other.metadata()['version'])

    def test_http_replies_are_cached_and_versioned_with_etags(self):
        self.cache.refresh_once()
        receiver = StreamReceiver.__new__(StreamReceiver)
        receiver.radio_images = self.cache
        receiver.log = logging.getLogger('test-radio-images')
        with patch('threading.Thread.start'), patch('time.sleep'):
            receiver.start_webserver()
        count = len(self.calls)
        with TestClient(receiver.app) as client:
            metadata = client.get('/radio-images').json()
            path = metadata['images'][0]['image_url']
            image = client.get(path)
            self.assertEqual(image.content, b'png-30')
            self.assertIn('immutable', image.headers['cache-control'])
            self.assertEqual(client.get(path, headers={'If-None-Match': image.headers['etag']}).status_code, 304)
            self.assertEqual(client.get('/radio-images/31.png').status_code, 404)
            self.assertEqual(client.get('/radio-images/30.png?version=expired').status_code, 404)
        self.assertEqual(len(self.calls), count)


if __name__ == '__main__':
    unittest.main()
