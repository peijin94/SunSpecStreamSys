"""Cached, observation-matched visibility flagging and A-team flux QA."""
from copy import deepcopy
from datetime import datetime
import json
import logging
import math
import threading
import time
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen

from antenna_flags import parse_flagging

POLL_SECONDS = 60
ATEAM_SOURCES = ('CasA', 'CygA', 'TauA', 'VirA')


def parse_flux(data):
    if not isinstance(data['rows'], list):
        raise ValueError('Invalid flux rows')
    rows = []
    for row in data['rows']:
        frequency, source = row['freq_mhz'], row['source']
        if (isinstance(frequency, bool) or not isinstance(frequency, (int, float))
                or not math.isfinite(frequency) or frequency <= 0 or source not in ATEAM_SOURCES):
            raise ValueError('Invalid flux band/source')
        measured, expected = row.get('measured_jy'), row.get('expected_jy')
        for value in (measured, expected):
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
                raise ValueError('Invalid flux measurement')
        if expected is not None and expected < 0:
            raise ValueError('Invalid expected flux')
        ratio = measured / expected if measured is not None and expected is not None and expected > 0 else None
        if ratio is not None and not math.isfinite(ratio):
            raise ValueError('Invalid flux ratio')
        rows.append({'freq_mhz': frequency, 'source': source, 'measured_jy': measured,
                     'expected_jy': expected, 'ratio': ratio})
    if len({(r['freq_mhz'], r['source']) for r in rows}) != len(rows):
        raise ValueError('Duplicate flux measurement')
    return sorted(rows, key=lambda row: (row['freq_mhz'], row['source']))


class ImagingStatusCache:
    def __init__(self, base_url, *, opener=urlopen, clock=time.time):
        self.base_url, self.opener, self.clock = base_url.rstrip('/'), opener, clock
        self.lock, self.refresh_lock = threading.Lock(), threading.Lock()
        self.current = None
        self.last_attempt = self.last_success = self.error = None
        self.log = logging.getLogger(__name__)

    def request(self, path):
        with self.opener(self.base_url + path, timeout=10) as response:
            payload = response.read(1024 * 1024 + 1)
        if len(payload) > 1024 * 1024:
            raise ValueError('QA response too large')
        return json.loads(payload)

    def refresh_once(self):
        if not self.refresh_lock.acquire(blocking=False):
            return
        with self.lock:
            self.last_attempt = self.clock()
        try:
            flags = self.request('/flagging?timestamp=newest')
            observed, bands = parse_flagging(flags)
            flux_rows, flux_message = [], None
            try:
                # The historical API selects nearest; require an exact match.
                flux = self.request('/flux?' + urlencode({'timestamp': flags['timestamp']}))
                if flux['timestamp'] == flags['timestamp']:
                    flux_rows = parse_flux(flux)
                if not flux_rows:
                    flux_message = 'No A-team measurements for this observation.'
            except HTTPError as exc:
                if exc.code != 404:
                    raise
                flux_message = 'No A-team measurements for this observation.'
            current = {'observation_time': observed,
                       'bands': [{'freq_mhz': b['freq_mhz'], 'flagged_frac': b['flagged_frac']} for b in bands],
                       'flux_rows': flux_rows, 'flux_status': 'ready' if flux_rows else 'unavailable',
                       'flux_message': flux_message}
            with self.lock:
                self.current = current
                self.last_success, self.error = self.clock(), None
        except Exception as exc:
            with self.lock:
                self.error = 'Imaging QA source temporarily unavailable; retrying every minute.'
            self.log.warning('Imaging QA refresh failed (%s)', type(exc).__name__)
        finally:
            self.refresh_lock.release()

    def run(self, shutdown_event):
        while not shutdown_event.is_set():
            started = time.monotonic()
            self.refresh_once()
            shutdown_event.wait(max(0, POLL_SECONDS - (time.monotonic() - started)))

    def metadata(self):
        with self.lock:
            current = deepcopy(self.current)
            old = bool(current and self.clock() - datetime.fromisoformat(
                current['observation_time'].replace('Z', '+00:00')).timestamp() > 600)
            stale = bool(current and (self.error or old or self.last_success is None
                         or self.clock() - self.last_success > 2 * POLL_SECONDS))
            return {'schema': 'lwa-imaging-status-v1',
                    'status': 'stale' if stale else 'ready' if current else 'unavailable' if self.error else 'loading',
                    'poll_interval_seconds': POLL_SECONDS,
                    'checked_at': self.last_attempt, 'fetched_at': self.last_success,
                    'message': self.error or ('Last observation is more than 10 minutes old.' if old else None),
                    **(current or {'observation_time': None, 'bands': [], 'flux_rows': [],
                                   'flux_status': 'unavailable', 'flux_message': None})}
