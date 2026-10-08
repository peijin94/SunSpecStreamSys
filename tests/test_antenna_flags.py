"""Correct indices, missing lists, atomic failures, and memory-only HTTP reads."""
from datetime import datetime, timezone
import io
import json
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from antenna_flags import AntennaFlagCache
from fastapi.testclient import TestClient
from stream_receiver import StreamReceiver


class AntennaFlagTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 10, 8, 12, tzinfo=timezone.utc).timestamp()
        self.calls = 0
        self.fail = False
        self.payload = {'timestamp': '20261008T120000', 'rows': [
            {'freq_mhz': 60, 'ant_list': [351, 0], 'n_bad_ant': 2, 'flagged_frac': .4},
            {'freq_mhz': 30, 'ant_list': [], 'n_bad_ant': 0, 'flagged_frac': .1}]}
        self.cache = AntennaFlagCache('http://source', opener=self.open, clock=lambda: self.now)

    def open(self, url, timeout):
        self.calls += 1
        self.assertEqual(url, 'http://source/flagging?timestamp=newest')
        self.assertEqual(timeout, 10)
        if self.fail:
            raise OSError('unavailable')
        return io.BytesIO(json.dumps(self.payload).encode())

    def test_indices_zero_and_351_are_preserved_and_bands_sorted(self):
        self.cache.refresh_once()
        result = self.cache.metadata()
        self.assertEqual(result['status'], 'ready')
        self.assertEqual(result['antenna_count'], 352)
        self.assertEqual(result['id_system'], 'ms_antenna_index')
        self.assertEqual(result['bands'][1]['ant_list'], [0, 351])
        self.assertEqual(result['bands'][0]['ant_list'], [])

    def test_legacy_missing_lists_remain_distinct_from_no_bad_antennas(self):
        self.payload['rows'][0]['ant_list'] = None
        self.cache.refresh_once()
        bands = self.cache.metadata()['bands']
        self.assertEqual(bands[0]['ant_list'], [])
        self.assertIsNone(bands[1]['ant_list'])
        self.assertEqual(bands[1]['n_bad_ant'], 2)

    def test_invalid_ids_or_counts_never_replace_previous_observation(self):
        self.cache.refresh_once()
        previous = self.cache.metadata()
        for ids, count in [([352], 1), ([-1], 1), ([True], 1), ([1, 1], 2), ([1], 3), ([1.5], 1)]:
            with self.subTest(ids=ids):
                self.payload['rows'][0].update(ant_list=ids, n_bad_ant=count)
                self.cache.refresh_once()
                after = self.cache.metadata()
                self.assertEqual(after['status'], 'stale')
                self.assertEqual(after['bands'], previous['bands'])
                self.assertEqual(after['observation_time'], previous['observation_time'])

    def test_failures_keep_last_observation_and_old_observations_are_labeled(self):
        self.cache.refresh_once()
        self.fail = True
        self.cache.refresh_once()
        self.assertEqual(self.cache.metadata()['status'], 'stale')
        self.assertEqual(self.cache.metadata()['bands'][1]['ant_list'], [0, 351])
        self.fail = False
        self.now += 601
        self.cache.refresh_once()
        self.assertEqual(self.cache.metadata()['status'], 'stale')
        self.assertIn('10 minutes', self.cache.metadata()['message'])

    def test_initial_source_failure_has_no_green_list(self):
        def missing(*args, **kwargs):
            raise HTTPError('http://source', 404, 'missing', {}, None)
        self.cache.opener = missing
        self.cache.refresh_once()
        result = self.cache.metadata()
        self.assertEqual(result['status'], 'unavailable')
        self.assertEqual(result['bands'], [])
        self.assertIsNone(result['observation_time'])

    def test_http_replies_only_read_the_cache(self):
        self.cache.refresh_once()
        with tempfile.TemporaryDirectory() as directory:
            receiver = StreamReceiver(plot_dir=directory, save_dir=directory, save_all_to_file=False)
            receiver.antenna_flags = self.cache
            try:
                with patch('threading.Thread.start'), patch('time.sleep'):
                    receiver.start_webserver()
                with TestClient(receiver.app) as client:
                    for _ in range(8):
                        response = client.get('/antenna-flags')
                        self.assertEqual(response.status_code, 200)
                        self.assertEqual(response.headers['cache-control'], 'no-store')
                        self.assertEqual(response.json()['bands'][1]['ant_list'], [0, 351])
                self.assertEqual(self.calls, 1)
            finally:
                receiver.socket.close()
                receiver.context.term()
