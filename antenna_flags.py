"""Minute-polled QA antenna flags, served from memory without upstream I/O."""
from copy import deepcopy
from datetime import datetime, timezone
import json
import logging
import math
import threading
import time
from urllib.error import HTTPError
from urllib.request import urlopen

ANTENNA_COUNT = 352
POLL_SECONDS = 60


def parse_flagging(data):
    stamp = data['timestamp']
    observed = datetime.strptime(stamp, '%Y%m%dT%H%M%S').replace(tzinfo=timezone.utc)
    if observed.strftime('%Y%m%dT%H%M%S') != stamp:
        raise ValueError('Invalid observation timestamp')
    rows = data['rows']
    if not isinstance(rows, list) or not rows:
        raise ValueError('No flagging bands')
    bands = []
    for row in rows:
        frequency = row['freq_mhz']
        if isinstance(frequency, bool) or not isinstance(frequency, (int, float)) or not math.isfinite(frequency) or frequency <= 0:
            raise ValueError('Invalid band frequency')
        ids = row.get('ant_list')
        count = row.get('n_bad_ant', row.get('bad_ant'))
        if count is not None and (type(count) is not int or not 0 <= count <= ANTENNA_COUNT):
            raise ValueError('Invalid antenna count')
        if ids is not None:
            if not isinstance(ids, list) or any(type(i) is not int or not 0 <= i < ANTENNA_COUNT for i in ids):
                raise ValueError('Invalid antenna index')
            if len(set(ids)) != len(ids) or count is not None and count != len(ids):
                raise ValueError('Inconsistent antenna list')
            ids = sorted(ids)
            count = len(ids)
        fraction = row.get('flagged_frac')
        if fraction is not None and (isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not math.isfinite(fraction) or not 0 <= fraction <= 1):
            raise ValueError('Invalid flagging fraction')
        bands.append({'freq_mhz': frequency, 'ant_list': ids,
                      'n_bad_ant': count, 'flagged_frac': fraction})
    if len({row['freq_mhz'] for row in bands}) != len(bands):
        raise ValueError('Duplicate flagging band')
    return observed.isoformat().replace('+00:00', 'Z'), sorted(bands, key=lambda row: row['freq_mhz'])


class AntennaFlagCache:
    def __init__(self, base_url, *, opener=urlopen, clock=time.time):
        self.base_url = base_url.rstrip('/')
        self.opener, self.clock = opener, clock
        self.lock, self.refresh_lock = threading.Lock(), threading.Lock()
        self.current = None
        self.last_attempt = self.last_success = self.error = None
        self.log = logging.getLogger(__name__)

    def refresh_once(self):
        if not self.refresh_lock.acquire(blocking=False):
            return
        with self.lock:
            self.last_attempt = self.clock()
        try:
            with self.opener(self.base_url + '/flagging?timestamp=newest', timeout=10) as response:
                payload = response.read(1024 * 1024 + 1)
            if len(payload) > 1024 * 1024:
                raise ValueError('Flagging response too large')
            observation, bands = parse_flagging(json.loads(payload))
            with self.lock:
                self.current = {'observation_time': observation, 'bands': bands}
                self.last_success = self.clock()
                self.error = None
        except Exception as exc:
            with self.lock:
                self.error = ('No antenna flagging observation is available.'
                              if isinstance(exc, HTTPError) and exc.code == 404 else
                              'Antenna flagging source temporarily unavailable; retrying every minute.')
            self.log.warning('Antenna flag refresh failed (%s)', type(exc).__name__)
        finally:
            self.refresh_lock.release()

    def run(self, shutdown_event):
        while not shutdown_event.is_set():
            started = time.monotonic()
            self.refresh_once()
            shutdown_event.wait(max(0, POLL_SECONDS - (time.monotonic() - started)))

    def metadata(self):
        with self.lock:
            current = self.current
            old = bool(current and self.clock() - datetime.fromisoformat(
                current['observation_time'].replace('Z', '+00:00')).timestamp() > 600)
            stale = bool(current and (self.error or old or self.last_success is None
                         or self.clock() - self.last_success > 2 * POLL_SECONDS))
            return {'status': 'stale' if stale else 'ready' if current else 'unavailable' if self.error else 'loading',
                    'antenna_count': ANTENNA_COUNT, 'id_system': 'ms_antenna_index',
                    'poll_interval_seconds': POLL_SECONDS,
                    'checked_at': self.last_attempt, 'fetched_at': self.last_success,
                    'message': self.error or ('Last observation is more than 10 minutes old.' if old else None),
                    'observation_time': current['observation_time'] if current else None,
                    'bands': deepcopy(current['bands']) if current else []}
