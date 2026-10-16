"""One-hour history/range regression tests; never use the production archive."""
import asyncio
from datetime import datetime, timezone
import json
import logging
import struct
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from stream_receiver import StreamReceiver
import test_polarization_api as polarization


class HistoryTests(unittest.TestCase):
    def fixture(self, count=8000):
        case = polarization.PolarizationTests(); case.setUp()
        r = case.receiver
        r.buffer_length = count
        r.ring_buffer = np.zeros((count, 768), np.float32)
        r.ring_buffer_v = np.zeros_like(r.ring_buffer)
        r.ring_arr_mjd = np.zeros((count, 1))
        r.streaming_interval = .5
        return case, r

    def fill(self, r, times):
        times = np.asarray(times, float)
        n = len(times)
        r.frames_received = n
        r.buffer_index = n % r.buffer_length
        keep = min(n, r.buffer_length)
        idx = np.arange(n - keep, n) % r.buffer_length
        r.ring_arr_mjd[idx, 0] = times[-keep:] / 86400 + 40587
        r.ring_buffer[idx] = np.arange(n - keep, n)[:, None] + 100
        r.ring_buffer_v[idx] = -.25 * r.ring_buffer[idx]

    def test_default_ring_capacity_and_empty_metadata(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(StreamReceiver, '_init_visitor_db'):
            receiver = StreamReceiver(plot_dir=directory, save_dir=directory, save_all_to_file=False)
            self.assertEqual(receiver.buffer_length, 7201)
            self.assertEqual(receiver.archive_frames, 0)
            empty = receiver.get_spectrum_range(seconds=180)
            self.assertEqual(empty['I'], [])
            self.assertIsNone(empty['coverage_start'])
            self.assertIsNone(empty['coverage_end'])
            receiver.socket.close(); receiver.context.term()

    def test_nan_aware_means_are_separate_before_fraction(self):
        _, r = self.fixture()
        width = 600/599  # UTC-aligned grid reserves a cell for a partial edge.
        center = (np.floor(1792108800/width)+.5)*width
        self.fill(r, [center-.1, center+.1])
        r.ring_buffer[:2, 0] = [10, 30]
        r.ring_buffer_v[:2, 0] = [-3, 9]
        r.ring_buffer[:2, 1] = [np.nan, 30]
        r.ring_buffer_v[:2, 1] = [np.nan, 9]
        result = r.get_spectrum_range(start=center-2, end=center+598, max_rows=600)
        valid = [n for n, row in enumerate(result['I']) if row[0] is not None]
        self.assertEqual(len(valid), 1)
        row = valid[0]
        self.assertEqual(result['I'][row][0], 20)
        self.assertEqual(result['V'][row][0], 3)
        self.assertEqual(result['I'][row][1], 30)
        self.assertEqual(result['V'][row][1], 9)

    def test_native_integration_coverage_does_not_fill_repeated_dropped_frames(self):
        _, r = self.fixture()
        self.fill(r, 1792108800 + np.arange(1000)*1.5)
        result = r.get_spectrum_range(seconds=180)
        self.assertIsNotNone(result.get('integration_seconds'))
        self.assertLessEqual(result['integration_seconds'], .525)

    def test_UTC_bin_membership_is_stable_across_fractional_view_shifts(self):
        _, r = self.fixture()
        self.fill(r, 1792108800 + np.arange(8000)*.5)
        a = r.get_spectrum_range(start=1792110000.1, end=1792110600.1, max_rows=600)
        b = r.get_spectrum_range(start=1792110000.35, end=1792110600.35, max_rows=600)
        common, ai, bi = np.intersect1d(a['timestamps'], b['timestamps'], return_indices=True)
        interior = (common > 1792110003) & (common < 1792110597)
        self.assertGreater(np.count_nonzero(interior), 500)
        np.testing.assert_equal(np.asarray(a['I'])[ai[interior]], np.asarray(b['I'])[bi[interior]])
        self.assertLessEqual(len(a['timestamps']), 600)
        self.assertLessEqual(len(b['timestamps']), 600)

    def test_one_hour_range_retains_real_times_after_wrap(self):
        _, r = self.fixture()
        self.fill(r, 1792120000 + np.arange(9500) * .5)
        method = getattr(r, 'get_spectrum_range', None)
        self.assertTrue(callable(method), 'One-hour selected spectrum range API is missing')
        data = method(seconds=180, max_rows=1200)
        self.assertTrue(data['range'])
        self.assertEqual(data['bin_seconds'], 0)
        self.assertIn(len(data['timestamps']), (360, 361))
        self.assertAlmostEqual(data['display_end'] - data['display_start'], 180)
        self.assertGreaterEqual(data['coverage_start'], data['coverage_end'] - 3600 - 1e-5)
        np.testing.assert_allclose(np.asarray(data['V']), -.25 * np.asarray(data['I']))
        self.assertTrue(np.all(np.diff(data['timestamps']) > 0))

    def test_coarse_bins_preserve_gaps_and_average_signed_polarizations(self):
        _, r = self.fixture()
        times = 1792108800 + np.concatenate([np.arange(0, 600, .5), np.arange(900, 3600, .5)])
        self.fill(r, times)
        data = r.get_spectrum_range(seconds=3600, max_rows=600)
        self.assertEqual(len(data['timestamps']), 600)
        self.assertAlmostEqual(data['bin_seconds'], 3600/599)
        rows = np.asarray(data['I'], dtype=float)
        stamps = np.asarray(data['timestamps'])
        missing = (stamps > 1792108800 + 605) & (stamps < 1792108800 + 895)
        self.assertTrue(np.isnan(rows[missing]).all())
        np.testing.assert_allclose(np.asarray(data['V'], float), -.25 * rows, equal_nan=True)
        self.assertLessEqual(len(rows), 600)

    def test_native_delta_and_boot_reset(self):
        _, r = self.fixture()
        self.fill(r, 1792120000 + np.arange(800) * .5)
        data = r.get_spectrum_range(seconds=180, since=797, boot_id=r.boot_id)
        self.assertFalse(data['reset'])
        self.assertEqual(len(data['timestamps']), 3)
        self.assertEqual(r.get_spectrum_range(seconds=180, since=800, boot_id=r.boot_id)['I'], [])
        self.assertTrue(r.get_spectrum_range(seconds=180, since=797, boot_id='other')['reset'])
        self.assertTrue(r.get_spectrum_range(seconds=180, since=1, boot_id=r.boot_id)['reset'])

    def test_invalid_and_expired_ranges(self):
        _, r = self.fixture()
        self.fill(r, 1792120000 + np.arange(8000) * .5)
        for kw in ({'seconds':3601}, {'start':0,'end':4000}, {'start':1},
                   {'start':5,'end':4}, {'seconds':float('nan')}, {'max_rows':999999}):
            with self.assertRaises(ValueError): r.get_spectrum_range(**kw)
        data = r.get_spectrum_range(start=100, end=200)
        self.assertEqual(data['timestamps'], [])
        self.assertEqual((data['display_start'], data['display_end']), (100, 200))

    def test_analysis_window_does_not_expand_with_ring_capacity(self):
        _, r = self.fixture()
        self.fill(r, 1792120000 + np.arange(8000) * .5)
        _, i, v, t = r._get_spectrum_arrays()
        self.assertEqual(len(i), 600)
        self.assertEqual(v.shape, i.shape)
        self.assertLessEqual(t[-1] - t[0], 300)

    def test_archive_chunks_stay_600_new_frames_with_paired_timestamps(self):
        case, r = self.fixture()
        calls = []
        r.save_all_to_file = True
        r.archive_frames = 0
        with tempfile.TemporaryDirectory() as directory:
            r.save_dir = directory
            r.save_latest_data_to_npz = lambda **kw: calls.append(kw)
            for n in range(1201): case.frame(n, 1792120000 + n*.5)
        self.assertEqual(len(calls), 2)
        self.assertEqual(r.frames_received, 1201)

    def test_npz_does_not_use_ring_order_or_unwritten_rows(self):
        _, r = self.fixture(700)
        self.fill(r, 1792120000 + np.arange(810) * .5)
        with tempfile.TemporaryDirectory() as directory:
            r.save_latest_data_to_npz(save_dir=directory, filename='test.npz', save_when_sun_up=False)
            with np.load(Path(directory) / 'test.npz') as data:
                self.assertEqual(data['data'].shape, (600, 768, 2))
                self.assertEqual(data['mjd'].shape, (600, 1))
                self.assertTrue(np.all(np.diff(data['mjd'][:,0]) > 0))
                np.testing.assert_allclose(data['data'][:,:,1], -.25 * data['data'][:,:,0])

    def test_http_range_is_bounded_binary_and_legacy_stays_compatible(self):
        case, r = self.fixture()
        self.fill(r, 1792120000 + np.arange(8000)*.5)
        with patch('threading.Thread.start'), patch('time.sleep'): r.start_webserver()
        case.receiver = r
        start, body = case.request('/spectrum-range?seconds=180&format=binary')
        self.assertEqual(start['status'],200)
        off = 4+struct.unpack_from('<I',body)[0]
        meta = json.loads(body[4:off])
        self.assertTrue(meta['range'])
        self.assertLessEqual(meta['rows'],1200)
        self.assertEqual(len(body), off + meta['rows']*(8+768*8))
        self.assertEqual(case.request('/spectrum-range?seconds=3601')[0]['status'],422)
        self.assertEqual(case.request('/spectrum-range?start=1&end=2&seconds=nan')[0]['status'],422)
        self.assertEqual(case.request('/spectrum-range?start=2&end=1')[0]['status'],400)
        start, body = case.request('/spectrum?format=json')
        self.assertLessEqual(len(json.loads(body)['I']),600)
        start, body = case.request('/spectrum?since=1&boot_id=test-boot')
        self.assertLessEqual(len(json.loads(body)['I']),600)


if __name__ == '__main__': unittest.main()
