"""Regression tests for synchronized Stokes I/V ingestion and browser history."""
import asyncio
from datetime import datetime, timezone
import json
import logging
import struct
import threading
import unittest
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import urlsplit

import numpy as np
from stream_receiver import StreamReceiver
from spectrum_time import time_raster, source_box


class PolarizationTests(unittest.TestCase):
    def setUp(self):
        self.receiver = receiver = StreamReceiver.__new__(StreamReceiver)
        receiver.buffer_length = 4
        receiver.ring_buffer = np.zeros((4, 768), dtype=np.float32)
        receiver.ring_buffer_v = np.zeros((4, 768), dtype=np.float32)
        receiver.ring_arr_mjd = np.zeros((4, 1), dtype=np.float64)
        receiver.buffer_lock = threading.RLock()
        receiver.buffer_index = 0
        receiver.frames_received = 0
        receiver.boot_id = 'test-boot'
        receiver.N_freq = 3072
        receiver.N_pol = 4
        receiver.verbose = False
        receiver.log = logging.getLogger('test-iv')
        receiver.delay_method = 'manual'
        receiver.save_all_to_file = False
        receiver.run_type3_detection = lambda: None

    def frame(self, value, timestamp=1791406800.0):
        data = np.zeros((1, 3072, 4), dtype=np.float32)
        data[:, :, 0] = 2 * value
        data[:, :, 1] = 4 * value
        data[:, :, 2] = 1000  # U must not be mistaken for V.
        data[:, :, 3] = -value
        self.receiver.process_frame(data.tobytes(), {
            'last_block_time': datetime.fromtimestamp(timestamp, timezone.utc)
                .replace(tzinfo=None).isoformat(sep=' ', timespec='milliseconds'),
        })

    def test_ingest_preserves_matching_i_v_and_source_time(self):
        self.frame(10, 1791406800.5)
        data = self.receiver.get_spectrum_snapshot()
        np.testing.assert_equal(data['I'], np.full((1, 768), 30))
        np.testing.assert_equal(data['V'], np.full((1, 768), -10))
        self.assertAlmostEqual(data['timestamps'][0], 1791406800.5, places=5)
        self.assertEqual(data['sequence'], 1)
        self.assertTrue(data['reset'])

    def test_incremental_catchup_and_unchanged_cursor(self):
        for n in range(1, 4): self.frame(n, 1791406800 + n * 0.5)
        data = self.receiver.get_spectrum_snapshot(1, 'test-boot')
        self.assertFalse(data['reset'])
        self.assertEqual(data['first_sequence'], 2)
        self.assertEqual([row[0] for row in data['I']], [6, 9])
        self.assertEqual([row[0] for row in data['V']], [-2, -3])
        self.assertEqual(self.receiver.get_spectrum_snapshot(3, 'test-boot')['I'], [])

    def test_wrap_keeps_chronology_and_does_not_hide_data(self):
        for n in range(1, 9): self.frame(n, 1791406800 + n * 0.5)
        self.assertEqual(self.receiver.buffer_index, 0)
        data = self.receiver.get_spectrum_snapshot(7, 'test-boot')
        self.assertEqual(data['I'][0][0], 24)
        old = self.receiver.get_spectrum_snapshot(1, 'test-boot')
        self.assertTrue(old['reset'])
        self.assertEqual([row[0] for row in old['I']], [15, 18, 21, 24])
        self.assertEqual(old['timestamps'], sorted(old['timestamps']))
        latest, mjd = self.receiver.get_latest_data(1)
        self.assertEqual(latest[0, 0], 24)
        self.assertAlmostEqual((mjd[0, 0] - 40587) * 86400, old['timestamps'][-1])

    def test_restart_and_future_cursors_reset(self):
        self.frame(5)
        for since, boot in [(0, 'old-boot'), (100, 'test-boot'), (-1, 'test-boot')]:
            self.assertTrue(self.receiver.get_spectrum_snapshot(since, boot)['reset'])

    def test_recent_tail_and_older_pages_are_contiguous_after_wrap(self):
        for n in range(1, 9): self.frame(n, 1791406800 + n * .5)
        tail = self.receiver.get_spectrum_snapshot(limit=2)
        self.assertEqual([row[0] for row in tail['I']], [21, 24])
        self.assertEqual(tail['first_sequence'], 7)
        self.assertEqual(tail['next_before'], 6)
        self.assertAlmostEqual(tail['previous_timestamp'], 1791406803, places=5)
        older = self.receiver.get_spectrum_snapshot(boot_id='test-boot', limit=2, before=6)
        self.assertTrue(older['history'])
        self.assertFalse(older['reset'])
        self.assertEqual([row[0] for row in older['I']], [15, 18])
        self.assertFalse(older['has_more'])
        self.assertEqual(older['sequence'], 8)
        empty = self.receiver.get_spectrum_snapshot(boot_id='test-boot', limit=2, before=4)
        self.assertEqual(empty['I'], [])
        restarted = self.receiver.get_spectrum_snapshot(boot_id='old-boot', limit=2, before=6)
        self.assertTrue(restarted['reset'])
        self.assertFalse(restarted['history'])

    def test_model_work_does_not_run_during_ingest(self):
        self.receiver.run_type3_detection = lambda: self.fail('Inference ran in ingestion')
        self.receiver.run_ai_summary = lambda: self.fail('AI ran in ingestion')
        for n in range(8): self.frame(n)
        self.assertEqual(self.receiver.frames_received, 8)

    def test_restart_handoff_preserves_both_polarizations_and_source_times(self):
        with tempfile.TemporaryDirectory() as directory:
            self.receiver.plot_dir = directory
            timestamps = np.array([time.time() - 1, time.time() - .5])
            np.savez(Path(directory) / 'restart-snapshot.npz', I=np.full((2, 768), 60),
                     V=np.full((2, 768), -20), timestamps=timestamps)
            self.receiver.restore_restart_snapshot()
            data = self.receiver.get_spectrum_snapshot()
            self.assertEqual(data['I'][1][0], 60)
            self.assertEqual(data['V'][1][0], -20)
            np.testing.assert_allclose(data['timestamps'], timestamps, atol=1e-5, rtol=0)
            self.assertTrue((Path(directory) / 'restart-snapshot.used.npz').exists())

    def test_binary_has_aligned_float_arrays_and_matches_json(self):
        self.frame(7)
        payload = self.receiver.get_spectrum_binary()
        header_size = struct.unpack_from('<I', payload)[0]
        offset = 4 + header_size
        self.assertEqual(offset % 8, 0)
        metadata = json.loads(payload[4:offset])
        self.assertEqual(metadata['schema'], 'lwa-spectrum-iv-v1')
        self.assertEqual(metadata['order'], ['I', 'V'])
        self.assertEqual(metadata['rows'], 1)
        times = np.frombuffer(payload, dtype='<f8', count=1, offset=offset)
        intensity = np.frombuffer(payload, dtype='<f4', count=768, offset=offset + 8)
        circular = np.frombuffer(payload, dtype='<f4', count=768, offset=offset + 8 + 768 * 4)
        data = self.receiver.get_spectrum_snapshot()
        np.testing.assert_equal(intensity, data['I'][0])
        np.testing.assert_equal(circular, data['V'][0])
        np.testing.assert_equal(times, data['timestamps'])
        self.assertEqual(len(payload), offset + 8 + 768 * 4 * 2)

    def test_missing_values_are_json_null_and_binary_nan(self):
        self.frame(7)
        self.receiver.ring_buffer_v[0, 10] = np.nan
        self.assertIsNone(self.receiver.get_spectrum_snapshot()['V'][0][10])
        payload = self.receiver.get_spectrum_binary()
        offset = 4 + struct.unpack_from('<I', payload)[0] + 8 + 768 * 4
        self.assertTrue(np.isnan(np.frombuffer(payload, '<f4', count=768, offset=offset)[10]))

    def test_snapshot_cannot_mix_polarizations_during_ingestion(self):
        def write():
            for n in range(1, 41): self.frame(n, 1791406800 + n * 0.5)
        writer = threading.Thread(target=write)
        writer.start()
        for _ in range(60):
            data = self.receiver.get_spectrum_snapshot()
            np.testing.assert_equal(np.asarray(data['I']), -3 * np.asarray(data['V']))
        writer.join(timeout=10)
        self.assertFalse(writer.is_alive())
        self.assertEqual(self.receiver.frames_received, 40)

    def request(self, url):
        async def run():
            parsed = urlsplit(url)
            messages = []
            async def receive(): return {'type': 'http.request', 'body': b'', 'more_body': False}
            async def send(message): messages.append(message)
            await self.receiver.app({
                'type': 'http', 'asgi': {'version': '3.0'}, 'http_version': '1.1',
                'method': 'GET', 'scheme': 'http', 'path': parsed.path,
                'raw_path': parsed.path.encode(), 'query_string': parsed.query.encode(),
                'root_path': '', 'headers': [], 'client': ('127.0.0.1', 1),
                'server': ('127.0.0.1', 9527),
            }, receive, send)
            start = next(m for m in messages if m['type'] == 'http.response.start')
            body = b''.join(m.get('body', b'') for m in messages if m['type'] == 'http.response.body')
            return start, body
        return asyncio.run(run())

    def test_http_formats_legacy_compatibility_and_cursor_validation(self):
        with patch('threading.Thread.start'), patch('time.sleep'):
            self.receiver.start_webserver()
        for n in range(1, 5): self.frame(n)
        start, body = self.request('/spectrum?format=binary')
        self.assertEqual(start['status'], 200)
        self.assertIn((b'content-type', b'application/octet-stream'), start['headers'])
        self.assertIn((b'cache-control', b'no-store'), start['headers'])
        self.assertGreater(len(body), 4)
        start, body = self.request('/spectrum')
        self.assertEqual(json.loads(body)['V'][-1][0], -4)
        start, body = self.request('/data')
        self.assertEqual(start['status'], 200)
        self.assertEqual(json.loads(body)[0], 12)
        start, body = self.request('/refresh?n_frames=2')
        self.assertEqual(len(json.loads(body)['data']), 2)
        self.assertEqual(self.request('/spectrum?since=bad')[0]['status'], 422)
        self.assertEqual(self.request('/spectrum?format=bad')[0]['status'], 400)
        self.assertEqual(self.request('/spectrum?limit=0')[0]['status'], 422)
        self.assertEqual(self.request('/spectrum?since=1&before=2')[0]['status'], 400)
        start, body = self.request('/spectrum?limit=2')
        self.assertEqual(len(json.loads(body)['I']), 2)


class SourceGeometryTests(unittest.TestCase):
    def test_detection_raster_uses_the_same_half_second_slots_and_real_gaps(self):
        times = np.array([100.49, 101.0036, 102.0308])
        raster, start, end = time_raster(np.array([[1], [2], [3]], dtype=np.float32), times)
        self.assertEqual(end, 102.5)
        self.assertEqual(end - start, 300)
        self.assertEqual(raster[-5:, 0].tolist(), [1, 2, 2, 0, 3])

    def test_image_box_has_absolute_source_times_and_positive_height(self):
        geometry = source_box([64, 128, 128, 256], (640, 640), 1000, 1300)
        np.testing.assert_allclose(geometry['bbox'], [.1, .6, .1, .2])
        self.assertEqual(geometry['time_start'], 1030)
        self.assertEqual(geometry['time_end'], 1060)

    def test_inference_keeps_its_snapshot_when_new_frames_arrive(self):
        fixture = PolarizationTests(); fixture.setUp()
        fixture.frame(1, 1791406800); fixture.frame(2, 1791406800.5)
        receiver = fixture.receiver
        receiver.last_detection_attempt = 0
        receiver.detection_interval = 5
        receiver.detection_confidence = .55
        receiver.detection_lock = threading.Lock()
        class Tensor:
            def __init__(self, value): self.value = np.asarray(value)
            def cpu(self): return self
            def numpy(self): return self.value
        class Boxes:
            xyxy = [Tensor([64, 128, 128, 256])]
            cls = [Tensor(0)]
            conf = [Tensor(.57)]
            def __len__(self): return 1
        class Model:
            def predict(self, path, **options):
                self.options = options
                fixture.frame(3, 1791406801)
                return [SimpleNamespace(boxes=Boxes(), orig_shape=(640, 640))]
        receiver.detection_model = model = Model()
        receiver.save_latest_data_to_jpg = lambda **kwargs: 'snapshot.jpg'
        StreamReceiver.run_type3_detection(receiver)
        detection = receiver.latest_detections[0]
        self.assertEqual(model.options['conf'], .55)
        self.assertEqual(detection['frame_sequence'], 2)
        self.assertAlmostEqual(detection['timestamp'], 1791406800.5, places=5)
        self.assertGreater(detection['bbox'][3], 0)
        self.assertEqual(receiver.frames_received, 3)

    def test_blocked_analysis_does_not_block_frame_publication(self):
        fixture = PolarizationTests(); fixture.setUp()
        receiver = fixture.receiver
        receiver.shutdown_event = threading.Event()
        entered = threading.Event(); release = threading.Event()
        def work(): entered.set(); release.wait(3)
        worker = threading.Thread(target=receiver.analysis_loop, args=(work,))
        worker.start()
        try:
            self.assertTrue(entered.wait(2))
            fixture.frame(5)
            self.assertEqual(receiver.frames_received, 1)
            self.assertFalse(release.is_set())
        finally:
            receiver.shutdown_event.set(); release.set(); worker.join(3)
        self.assertFalse(worker.is_alive())


if __name__ == '__main__': unittest.main()
