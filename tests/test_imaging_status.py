"""Matched QA observations, absent measurements, retention, and cached HTTP."""
from datetime import datetime, timezone
import io
import json
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from fastapi.testclient import TestClient
from imaging_status import ImagingStatusCache
from stream_receiver import StreamReceiver


class ImagingStatusTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 10, 8, 12, tzinfo=timezone.utc).timestamp()
        self.calls = []
        self.failure = None
        self.flags = {'timestamp': '20261008T120000', 'rows': [
            {'freq_mhz': 59, 'flagged_frac': .25, 'ant_list': [], 'n_bad_ant': 0}]}
        self.flux = {'timestamp': '20261008T120000', 'rows': [
            {'freq_mhz': 59, 'source': 'CasA', 'measured_jy': 200., 'expected_jy': 100., 'ratio': 999.},
            {'freq_mhz': 59, 'source': 'CygA', 'measured_jy': 0., 'expected_jy': 100., 'ratio': 0.}]}
        self.cache = ImagingStatusCache('http://source', opener=self.open, clock=lambda: self.now)

    def open(self, url, timeout):
        self.calls.append(url)
        self.assertEqual(timeout, 10)
        if self.failure and '/flux?' in url:
            if self.failure == 404:
                raise HTTPError(url, 404, 'missing', {}, None)
            raise OSError('unavailable')
        return io.BytesIO(json.dumps(self.flags if '/flagging?' in url else self.flux).encode())

    def test_flux_is_pinned_to_flagging_timestamp_and_ratios_use_actual_values(self):
        self.cache.refresh_once()
        result = self.cache.metadata()
        self.assertEqual(result['status'], 'ready')
        self.assertEqual(self.calls[1], 'http://source/flux?timestamp=20261008T120000')
        self.assertEqual(result['bands'][0]['flagged_frac'], .25)
        self.assertEqual(result['flux_rows'][0]['ratio'], 2.)
        self.assertEqual(result['flux_rows'][1]['ratio'], 0.)

    def test_nearest_older_flux_never_mixes_with_new_flagging(self):
        self.flux['timestamp'] = '20261008T115900'
        self.cache.refresh_once()
        result = self.cache.metadata()
        self.assertEqual(result['bands'][0]['flagged_frac'], .25)
        self.assertEqual(result['flux_status'], 'unavailable')
        self.assertEqual(result['flux_rows'], [])
        self.assertIn('this observation', result['flux_message'])

    def test_missing_flux_or_zero_expected_is_unknown_not_a_passing_check(self):
        self.failure = 404
        self.cache.refresh_once()
        self.assertEqual(self.cache.metadata()['flux_rows'], [])
        self.assertEqual(self.cache.metadata()['status'], 'ready')
        self.failure = None
        self.flux['rows'][0]['expected_jy'] = 0.
        self.flux['rows'][1]['measured_jy'] = None
        self.cache.refresh_once()
        self.assertTrue(all(row['ratio'] is None for row in self.cache.metadata()['flux_rows']))

    def test_failures_keep_the_previous_complete_observation_and_mark_it_stale(self):
        self.cache.refresh_once()
        previous = self.cache.metadata()
        self.flags['timestamp'] = '20261008T120100'
        self.failure = 503
        self.cache.refresh_once()
        after = self.cache.metadata()
        self.assertEqual(after['status'], 'stale')
        self.assertEqual(after['observation_time'], previous['observation_time'])
        self.assertEqual(after['flux_rows'], previous['flux_rows'])
        self.failure = None
        self.flags['timestamp'] = '20261008T120000'
        self.now += 601
        self.cache.refresh_once()
        self.assertIn('10 minutes', self.cache.metadata()['message'])

    def test_nonfinite_or_duplicate_measurements_do_not_replace_valid_cache(self):
        self.cache.refresh_once()
        previous = self.cache.metadata()['flux_rows']
        self.flux['rows'][0]['measured_jy'] = float('nan')
        self.cache.refresh_once()
        self.assertEqual(self.cache.metadata()['flux_rows'], previous)
        self.flux['rows'][0]['measured_jy'] = 200.
        self.flux['rows'].append(dict(self.flux['rows'][0]))
        self.cache.refresh_once()
        self.assertEqual(self.cache.metadata()['flux_rows'], previous)

    def test_http_requests_never_poll_upstream(self):
        self.cache.refresh_once()
        with tempfile.TemporaryDirectory() as directory:
            receiver = StreamReceiver(plot_dir=directory, save_dir=directory, save_all_to_file=False)
            receiver.imaging_status = self.cache
            try:
                with patch('threading.Thread.start'), patch('time.sleep'):
                    receiver.start_webserver()
                with TestClient(receiver.app) as client:
                    for _ in range(8):
                        reply = client.get('/imaging-status')
                        self.assertEqual(reply.status_code, 200)
                        self.assertEqual(reply.headers['cache-control'], 'no-store')
                        self.assertEqual(reply.json()['flux_rows'][0]['ratio'], 2.)
                self.assertEqual(len(self.calls), 2)
            finally:
                receiver.socket.close()
                receiver.context.term()
