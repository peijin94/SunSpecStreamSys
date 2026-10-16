"""Public/dev deployment variants; no production archive or network use."""
import asyncio
import json
import tempfile
import threading
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlsplit

import numpy as np

import deployment_mode
import stream_receiver as receiver_module
from stream_receiver import StreamReceiver, build_parser
import test_polarization_api as fixtures


def make_receiver(mode='dev', capacity=8000, web_port=9527, no_archive=False,
                  streaming_interval=0.5):
    case = fixtures.PolarizationTests()
    case.setUp()
    r = case.receiver
    r.app_mode = mode
    r.web_port = web_port
    r.no_archive = no_archive
    r.buffer_length = capacity
    r.ring_buffer = np.zeros((capacity, 768), np.float32)
    r.ring_buffer_v = np.zeros_like(r.ring_buffer)
    r.ring_arr_mjd = np.zeros((capacity, 1))
    r.streaming_interval = streaming_interval
    r.archive_frames = 0
    return case, r


def fill(r, times):
    times = np.asarray(times, float)
    n = len(times)
    r.frames_received = n
    r.buffer_index = n % r.buffer_length
    keep = min(n, r.buffer_length)
    idx = np.arange(n - keep, n) % r.buffer_length
    r.ring_arr_mjd[idx, 0] = times[-keep:] / 86400 + 40587
    r.ring_buffer[idx] = np.arange(n - keep, n)[:, None] + 100
    r.ring_buffer_v[idx] = -.25 * r.ring_buffer[idx]


def asgi_request(app, method, url, headers=None, body=b''):
    parsed = urlsplit(url)

    async def run():
        messages = []

        async def receive():
            return {'type': 'http.request', 'body': body, 'more_body': False}

        async def send(message):
            messages.append(message)

        await app({
            'type': 'http', 'asgi': {'version': '3.0'}, 'http_version': '1.1',
            'method': method, 'scheme': 'http', 'path': parsed.path,
            'raw_path': parsed.path.encode(),
            'query_string': parsed.query.encode(),
            'root_path': '', 'headers': headers or [],
            'client': ('127.0.0.1', 1), 'server': ('127.0.0.1', 9527),
        }, receive, send)
        start = next(m for m in messages if m['type'] == 'http.response.start')
        payload = b''.join(m.get('body', b'')
                            for m in messages if m['type'] == 'http.response.body')
        return start, payload

    return asyncio.run(run())


class DeploymentPolicyTests(unittest.TestCase):
    def test_mode_normalization_defaults_to_dev(self):
        self.assertEqual(deployment_mode.normalize_mode(None), 'dev')
        self.assertEqual(deployment_mode.normalize_mode('dev'), 'dev')
        self.assertEqual(deployment_mode.normalize_mode('PUBLIC'), 'public')
        with self.assertRaises(ValueError):
            deployment_mode.normalize_mode('staging')

    def test_history_seconds_per_mode(self):
        self.assertEqual(deployment_mode.history_seconds('dev'), 3600)
        self.assertEqual(deployment_mode.history_seconds('public'), 300)

    def test_ring_capacity_preserves_archive_and_analysis(self):
        self.assertEqual(deployment_mode.ring_capacity(0.5, 'dev'), 7201)
        self.assertEqual(deployment_mode.ring_capacity(0.5, 'public'), 601)
        # A slow cadence must still hold the 600-frame archive/analysis window.
        self.assertGreaterEqual(deployment_mode.ring_capacity(2.0, 'public'), 601)
        with self.assertRaises(ValueError):
            deployment_mode.ring_capacity(0, 'dev')

    def test_constructor_defaults_match_existing_dev_behavior(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(StreamReceiver, '_init_visitor_db'):
            receiver = StreamReceiver(plot_dir=directory, save_dir=directory,
                                      save_all_to_file=False)
            try:
                self.assertEqual(receiver.app_mode, 'dev')
                self.assertEqual(receiver.web_port, 9527)
                self.assertFalse(receiver.no_archive)
                self.assertEqual(receiver.buffer_length, 7201)
                self.assertEqual(receiver.get_history_seconds(), 3600)
                self.assertEqual(receiver.get_buffer_status()['history_seconds'], 3600)
            finally:
                receiver.socket.close()
                receiver.context.term()

    def test_constructor_public_capacity_keeps_600_frame_minimum(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(StreamReceiver, '_init_visitor_db'):
            receiver = StreamReceiver(plot_dir=directory, save_dir=directory,
                                      save_all_to_file=False, app_mode='public',
                                      streaming_interval=2.0)
            try:
                self.assertEqual(receiver.app_mode, 'public')
                self.assertGreaterEqual(receiver.buffer_length, 601)
                self.assertEqual(receiver.get_history_seconds(), 300)
            finally:
                receiver.socket.close()
                receiver.context.term()

    def test_cli_defaults_and_overrides(self):
        defaults = build_parser().parse_args([])
        self.assertEqual(defaults.app_mode, 'dev')
        self.assertEqual(defaults.web_port, 9527)
        self.assertFalse(defaults.no_archive)
        # --port stays the ZMQ source port.
        self.assertTrue(hasattr(defaults, 'port'))
        custom = build_parser().parse_args(['--app-mode', 'public', '--web-port',
                                            '9529', '--no-archive'])
        self.assertEqual(custom.app_mode, 'public')
        self.assertEqual(custom.web_port, 9529)
        self.assertTrue(custom.no_archive)

    def test_range_policy_dev_keeps_full_hour(self):
        _, r = make_receiver('dev')
        fill(r, 1792120000 + np.arange(8000) * .5)
        self.assertIn(len(r.get_spectrum_range(seconds=180)['timestamps']), (360, 361))
        explicit = r.get_spectrum_range(start=1792120000, end=1792120600, max_rows=600)
        self.assertEqual(len(explicit['timestamps']), 600)
        self.assertEqual(r.get_spectrum_range(seconds=180)['history_seconds'], 3600)

    def test_range_policy_public_is_fixed_rolling_300s(self):
        _, r = make_receiver('public')
        fill(r, 1792120000 + np.arange(8000) * .5)
        rolling = r.get_spectrum_range(seconds=300)
        self.assertEqual(rolling['history_seconds'], 300)
        self.assertLessEqual(rolling['display_end'] - rolling['display_start'], 300 + 1e-6)
        with self.assertRaises(ValueError):
            r.get_spectrum_range(seconds=180)
        with self.assertRaises(ValueError):
            r.get_spectrum_range(seconds=3600)
        with self.assertRaises(ValueError):
            r.get_spectrum_range(start=1792120000, end=1792120300)

    def test_spectrum_paged_history_cutoff(self):
        base = 1792120000.0
        _, dev = make_receiver('dev')
        fill(dev, base + np.arange(2400) * 1.5)
        _, public = make_receiver('public')
        fill(public, base + np.arange(2400) * 1.5)
        dev_tail = dev.get_spectrum_snapshot(limit=600)
        public_tail = public.get_spectrum_snapshot(limit=600)
        self.assertGreater(dev_tail['timestamps'][-1] - dev_tail['timestamps'][0], 300)
        self.assertLessEqual(
            public_tail['timestamps'][-1] - public_tail['timestamps'][0], 300 + 1e-6)
        self.assertAlmostEqual(public_tail['timestamps'][-1],
                               dev_tail['timestamps'][-1], places=5)
        # Dev pages further back; the public tail already reaches the 300 s
        # horizon, and older pages are rejected outright (live deltas use since).
        dev_older = dev.get_spectrum_snapshot(boot_id=dev.boot_id, limit=600,
                                              before=dev_tail['next_before'])
        self.assertEqual(len(dev_older['I']), 600)
        self.assertFalse(public_tail['has_more'])
        self.assertIsNone(public_tail['next_before'])
        with self.assertRaises(ValueError):
            public.get_spectrum_snapshot(boot_id=public.boot_id, limit=600,
                                         before=public_tail['first_sequence'] - 1)
        # An in-horizon cursor delta still works on public.
        delta = public.get_spectrum_snapshot(public_tail['sequence'] - 1, public.boot_id)
        self.assertFalse(delta['reset'])
        self.assertEqual(len(delta['I']), 1)

    def test_legacy_data_and_refresh_cutoff(self):
        base = 1792120000.0
        _, dev = make_receiver('dev')
        fill(dev, base + np.arange(2400) * 1.5)
        _, public = make_receiver('public')
        fill(public, base + np.arange(2400) * 1.5)
        dev_values, dev_mjd = dev.get_latest_data(n_frames=600)
        public_values, public_mjd = public.get_latest_data(n_frames=600)
        self.assertEqual(dev_values.shape[0], 600)
        dev_span = (dev_mjd[-1, 0] - dev_mjd[0, 0]) * 86400
        public_span = (public_mjd[-1, 0] - public_mjd[0, 0]) * 86400
        self.assertGreater(dev_span, 300)
        self.assertLessEqual(public_span, 300 + 1e-6)
        self.assertAlmostEqual((public_mjd[-1, 0] - dev_mjd[-1, 0]) * 86400, 0, places=5)

    def test_measure_visibility(self):
        _, dev = make_receiver('dev')
        with patch('threading.Thread.start'), patch('time.sleep'):
            dev.start_webserver()
        _, public = make_receiver('public')
        with patch('threading.Thread.start'), patch('time.sleep'):
            public.start_webserver()
        body = json.dumps({'points': [[1000, 60], [1010, 40]]}).encode()
        dev_status, _ = asgi_request(dev.app, 'POST', '/spectrum-measure', body=body,
                                     headers=[(b'content-type', b'application/json')])
        self.assertEqual(dev_status['status'], 200)
        public_status, _ = asgi_request(
            public.app, 'POST', '/spectrum-measure', body=body,
            headers=[(b'content-type', b'application/json')])
        self.assertEqual(public_status['status'], 404)
        # Public range violations surface as client errors, not data.
        bad_status, _ = asgi_request(public.app, 'GET', '/spectrum-range?seconds=180')
        self.assertIn(bad_status['status'], (400, 422))
        frozen_status, _ = asgi_request(
            public.app, 'GET', '/spectrum-range?start=1792120000&end=1792120300')
        self.assertEqual(frozen_status['status'], 400)
        # Public rejects before pagination over HTTP; dev still pages.
        before_status, _ = asgi_request(
            public.app, 'GET', '/spectrum?before=10&boot_id=x&limit=2')
        self.assertEqual(before_status['status'], 400)
        case, dev_paged = make_receiver('dev')
        case.frame(1)
        with patch('threading.Thread.start'), patch('time.sleep'):
            dev_paged.start_webserver()
        dev_before, dev_before_body = asgi_request(
            dev_paged.app, 'GET', '/spectrum?before=1&boot_id=test-boot&limit=2')
        self.assertEqual(dev_before['status'], 200)
        self.assertIn('I', json.loads(dev_before_body))

    def test_deployment_capability_metadata(self):
        _, dev = make_receiver('dev', web_port=9529)
        with patch('threading.Thread.start'), patch('time.sleep'):
            dev.start_webserver()
        _, public = make_receiver('public', web_port=9527)
        with patch('threading.Thread.start'), patch('time.sleep'):
            public.start_webserver()
        dev_status, dev_body = asgi_request(dev.app, 'GET', '/deployment')
        public_status, public_body = asgi_request(public.app, 'GET', '/deployment')
        self.assertEqual(dev_status['status'], 200)
        self.assertEqual(public_status['status'], 200)
        dev_meta, public_meta = json.loads(dev_body), json.loads(public_body)
        self.assertEqual(dev_meta, {'mode': 'dev', 'history_seconds': 3600,
                                    'interactive': True, 'measurement_enabled': True,
                                    'web_port': 9529})
        self.assertEqual(public_meta, {'mode': 'public', 'history_seconds': 300,
                                       'interactive': False,
                                       'measurement_enabled': False, 'web_port': 9527})

    def test_debug_request_never_reflects_credentials(self):
        _, r = make_receiver('dev')
        with patch('threading.Thread.start'), patch('time.sleep'):
            r.start_webserver()
        status, body = asgi_request(
            r.app, 'GET', '/debug/request',
            headers=[(b'authorization', b'Basic c2VjcmV0'),
                     (b'proxy-authorization', b'Basic cHJveHk='),
                     (b'cookie', b'session=abc123'),
                     (b'user-agent', b'test-agent'),
                     (b'x-forwarded-for', b'203.0.113.7')])
        self.assertEqual(status['status'], 200)
        payload = json.loads(body)
        echoed = {k.lower(): v for k, v in payload['headers'].items()}
        self.assertNotIn('authorization', echoed)
        self.assertNotIn('proxy-authorization', echoed)
        self.assertNotIn('cookie', echoed)
        for value in payload['headers'].values():
            self.assertNotIn('c2VjcmV0', value)
        self.assertEqual(echoed.get('user-agent'), 'test-agent')

    def test_no_archive_suppresses_writes_but_keeps_prefill(self):
        with tempfile.TemporaryDirectory() as root:
            _, r = make_receiver('dev', capacity=64, no_archive=True)
            r.save_all_to_file = True
            r.save_dir = root
            r.archive_frames = 599
            self.assertIsNone(r.save_latest_data_to_npz(save_dir=root,
                                                         filename='manual.npz',
                                                         save_when_sun_up=False))
            self.assertEqual(list(Path(root).glob('*.npz')), [])
            self.assertEqual(r.archive_frames, 599)
            # Archive prefill reading still works under --no-archive.
            now = time.time()
            save_root = Path(root) / 'archives'
            save_root.mkdir()

            stamp = datetime.fromtimestamp(now - 1, timezone.utc)
            folder = save_root / stamp.strftime('%Y-%m-%d')
            folder.mkdir(parents=True, exist_ok=True)
            values = np.stack(
                (np.full((2, 768), 240000.0, np.float32),
                 np.full((2, 768), -60000.0, np.float32)), axis=2) / (1e4 * 24)
            mjd = np.array([now - 2, now - 1]) / 86400 + 40587
            np.savez(folder / stamp.strftime('%Y-%m-%d_%H_%M_%S.npz'), data=values,
                     mjd=mjd[:, None],
                     freq=np.linspace(14.35546875, 87.83154296875, 768))
            r.save_dir = str(save_root)
            r.frames_received = 0
            r.buffer_index = 0
            r.restore_archive_buffer()
            self.assertEqual(r.frames_received, 2)
            self.assertEqual(r.archive_frames, 599)

    def test_handoff_accepts_hour_and_trims_to_capacity(self):
        with tempfile.TemporaryDirectory() as directory:
            latest = time.time() - 0.5
            count = 2000
            times = latest - np.arange(count - 1, -1, -1) * 0.5
            intensity = np.full((count, 768), 60, np.float32)
            circular = np.full((count, 768), -20, np.float32)
            np.savez(Path(directory) / 'restart-snapshot.npz', I=intensity, V=circular,
                     timestamps=times)
            _, public = make_receiver('public', capacity=601)
            public.plot_dir = directory
            public.restore_restart_snapshot()
            self.assertEqual(public.frames_received, 601)
            self.assertEqual(public.archive_frames, 0)
            data = public.get_spectrum_snapshot()
            self.assertAlmostEqual(data['timestamps'][-1], latest, places=4)
            self.assertLessEqual(data['timestamps'][-1] - data['timestamps'][0],
                                 300 + 1e-6)
            self.assertTrue((Path(directory) / 'restart-snapshot.used.npz').exists())

    def test_handoff_preserves_slow_cadence_raw_for_public(self):
        # 600 frames at 2 s cadence span 1198 s: the public API horizon is
        # 300 s, but the raw restart handoff must survive up to ring
        # capacity so detection keeps the latest 600.
        with tempfile.TemporaryDirectory() as directory:
            latest = time.time() - 0.5
            count = 600
            times = latest - np.arange(count - 1, -1, -1) * 2.0
            np.savez(Path(directory) / 'restart-snapshot.npz',
                     I=np.full((count, 768), 60, np.float32),
                     V=np.full((count, 768), -20, np.float32), timestamps=times)
            _, public = make_receiver('public', capacity=601,
                                      streaming_interval=2.0)
            public.plot_dir = directory
            public.restore_restart_snapshot()
            self.assertEqual(public.frames_received, count)
            self.assertEqual(public.archive_frames, 0)
            _, _, _, internal = public._get_spectrum_arrays(_public_cutoff=False)
            self.assertEqual(len(internal), count)
            self.assertAlmostEqual(internal[-1] - internal[0], 1198.0, delta=0.05)
            data = public.get_spectrum_snapshot()
            self.assertAlmostEqual(data['timestamps'][-1], latest, places=4)
            self.assertLessEqual(data['timestamps'][-1] - data['timestamps'][0],
                                 300 + 1e-6)
            window = public.get_spectrum_range(seconds=300)
            self.assertLessEqual(window['display_end'] - window['display_start'],
                                 300 + 1e-6)
            self.assertTrue((Path(directory) / 'restart-snapshot.used.npz').exists())

    def test_handoff_keeps_full_hour_for_dev(self):
        with tempfile.TemporaryDirectory() as directory:
            latest = time.time() - 0.5
            count = 2000
            times = latest - np.arange(count - 1, -1, -1) * 0.5
            np.savez(Path(directory) / 'restart-snapshot.npz',
                     I=np.full((count, 768), 60, np.float32),
                     V=np.full((count, 768), -20, np.float32), timestamps=times)
            _, dev = make_receiver('dev', capacity=7201)
            dev.plot_dir = directory
            dev.restore_restart_snapshot()
            self.assertEqual(dev.frames_received, count)
            self.assertEqual(dev.archive_frames, 0)

    def test_cors_policy_per_mode(self):
        self.assertEqual(deployment_mode.cors_options('dev')['allow_origins'],
                         ['https://ovsa.njit.edu'])
        self.assertTrue(deployment_mode.cors_options('dev')['allow_credentials'])
        public_options = deployment_mode.cors_options('public')
        self.assertEqual(public_options['allow_origins'], ['*'])
        self.assertFalse(public_options['allow_credentials'])
        self.assertNotIn('POST', public_options['allow_methods'])

    def test_attacker_origin_is_never_credentialed(self):
        _, dev = make_receiver('dev')
        with patch('threading.Thread.start'), patch('time.sleep'):
            dev.start_webserver()
        _, public = make_receiver('public')
        with patch('threading.Thread.start'), patch('time.sleep'):
            public.start_webserver()

        def response_headers(app, method, url, headers):
            start, _ = asgi_request(app, method, url, headers=headers)
            return {k.decode(): v.decode() for k, v in start['headers']}

        # Dev echoes only the observatory origin, never an attacker origin.
        dev_good = response_headers(dev.app, 'GET', '/deployment',
                                    [(b'origin', b'https://ovsa.njit.edu')])
        self.assertEqual(dev_good.get('access-control-allow-origin'),
                         'https://ovsa.njit.edu')
        self.assertEqual(dev_good.get('access-control-allow-credentials'), 'true')
        dev_evil = response_headers(dev.app, 'GET', '/deployment',
                                    [(b'origin', b'https://evil.example')])
        # No origin echo means the browser grants nothing; a lone
        # allow-credentials without allow-origin is inert.
        self.assertNotIn('access-control-allow-origin', dev_evil)
        # Public stays a credentialless wildcard even for an attacker origin,
        # and its preflight never grants POST.
        public_evil = response_headers(public.app, 'GET', '/deployment',
                                       [(b'origin', b'https://evil.example')])
        self.assertEqual(public_evil.get('access-control-allow-origin'), '*')
        self.assertNotIn('access-control-allow-credentials', public_evil)
        preflight = response_headers(
            public.app, 'OPTIONS', '/spectrum',
            [(b'origin', b'https://evil.example'),
             (b'access-control-request-method', b'POST')])
        self.assertNotIn('POST', preflight.get('access-control-allow-methods', ''))

    def test_web_port_reaches_configured_loopback_binding(self):
        _, r = make_receiver('dev', web_port=9531)
        targets = []

        class FakeThread:
            def __init__(self, target=None, **kwargs):
                targets.append(target)

            def start(self):
                pass

        with patch('threading.Thread', FakeThread), patch('time.sleep'):
            r.start_webserver()
        self.assertTrue(targets)
        with patch.object(receiver_module.uvicorn, 'run') as run:
            targets[-1]()
        _, kwargs = run.call_args
        self.assertEqual(kwargs['port'], 9531)
        self.assertEqual(kwargs['host'], '127.0.0.1')
        r.start_webshow = True
        with patch.object(StreamReceiver, 'is_webserver_running', return_value=True):
            self.assertEqual(r.get_webserver_status()['url'], 'http://localhost:9531')


if __name__ == '__main__':
    unittest.main()
