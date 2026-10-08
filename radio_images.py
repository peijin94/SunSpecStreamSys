"""Minute-polled, in-memory fine-channel image snapshots for the live page."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import logging
import threading
import time
from urllib.parse import urlencode
from urllib.error import HTTPError
from urllib.request import urlopen

import numpy as np

FREQUENCIES = (30, 45, 60, 75)
POLL_SECONDS = 60
SOLAR_EXTENT_ARCSEC = 6000
RENDER_PROFILE = f'solar-hpc:{SOLAR_EXTENT_ARCSEC}:512:v2'.encode()


@dataclass(frozen=True)
class ImagePlane:
    target: int
    frequency: float
    date_obs: str
    image: np.ndarray
    header: object


@dataclass(frozen=True)
class ImageSnapshot:
    timestamp: str
    observation_time: str
    version: str
    planes: tuple
    pngs: tuple


def read_plane(payload, expected):
    """Require an actual 2D brightness-temperature plane and matching channel."""
    from astropy import units as u
    from astropy.io import fits

    with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
        image = np.array(arrays['image'], dtype=np.float32, copy=True)
        target = int(arrays['target_mhz'])
        frequency = float(arrays['freq_mhz'])
        date_obs = str(arrays['date_obs'])
        unit = u.Unit(str(arrays['bunit']))
        header = fits.Header.fromstring(str(arrays['header']))
    if (target != expected['target_mhz'] or image.ndim != 2 or not image.size
            or not np.isfinite(frequency) or abs(frequency - float(expected['freq_mhz'])) > .001
            or not np.isfinite(image).any() or not unit.is_equivalent(u.K)):
        raise ValueError('Inconsistent image plane')
    image *= unit.to(u.MK)
    return ImagePlane(target, frequency, date_obs, image, header)


def solar_view(plane, size=512, extent=SOLAR_EXTENT_ARCSEC):
    """Resample helioprojective WCS onto a common solar arcsecond grid."""
    from astropy import units as u
    from astropy.wcs import WCS
    from scipy.ndimage import map_coordinates

    if not (str(plane.header.get('CTYPE1', '')).startswith('HPLN')
            and str(plane.header.get('CTYPE2', '')).startswith('HPLT')):
        raise ValueError('Image lacks helioprojective spatial coordinates')
    wcs = WCS(plane.header, naxis=2)
    # imshow's extent describes pixel edges, so sample the pixel centers.
    coordinates = np.linspace(-extent, extent, size, endpoint=False) + extent / size
    x, y = np.meshgrid(coordinates, coordinates)
    world_x = (x * u.arcsec).to_value(u.Unit(wcs.wcs.cunit[0]))
    world_y = (y * u.arcsec).to_value(u.Unit(wcs.wcs.cunit[1]))
    pixel_x, pixel_y = wcs.all_world2pix(world_x, world_y, 0)
    view = map_coordinates(plane.image, [pixel_y, pixel_x], order=1,
                           mode='constant', cval=np.nan, prefilter=False)
    if not np.isfinite(view).any():
        raise ValueError('Solar field lies outside the image')
    return view


def render_planes(planes):
    """Render all four channels once, with a shared MK scale and solar field."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import Circle, Ellipse
    from matplotlib import pyplot as plt

    views = [solar_view(plane) for plane in planes]
    maximum = max(1., *(float(np.nanpercentile(view, 99.8)) for view in views))
    pngs = []
    for plane, view in zip(planes, views):
        figure = Figure(figsize=(6, 5), dpi=100, facecolor='#0b1120')
        FigureCanvasAgg(figure)
        axes = figure.add_axes([.15, .13, .67, .80], facecolor='#020617')
        cmap = plt.get_cmap('inferno').copy()
        cmap.set_bad('#020617')
        extent = SOLAR_EXTENT_ARCSEC
        artist = axes.imshow(view, origin='lower', extent=(-extent, extent, -extent, extent),
                             cmap=cmap, vmin=0, vmax=maximum, interpolation='nearest')
        radius = float(plane.header.get('RSUN_OBS', 959.63))
        axes.add_patch(Circle((0, 0), radius, fill=False, edgecolor='white',
                              linestyle='--', linewidth=.8, alpha=.7))
        if plane.header.get('BMAJ') and plane.header.get('BMIN'):
            axes.add_patch(Ellipse((-.77 * extent, -.77 * extent), float(plane.header['BMIN']) * 3600,
                                   float(plane.header['BMAJ']) * 3600,
                                   angle=-float(plane.header.get('BPA', 0)),
                                   facecolor='none', edgecolor='white', linewidth=.8))
        axes.set(xlabel='Solar X (arcsec)', ylabel='Solar Y (arcsec)',
                 xlim=(-extent, extent), ylim=(-extent, extent))
        axes.set_xticks([-4000, -2000, 0, 2000, 4000]); axes.set_yticks([-4000, -2000, 0, 2000, 4000])
        axes.tick_params(colors='#cbd5e1', labelsize=10)
        for axis in (axes.xaxis, axes.yaxis):
            axis.label.set_color('#cbd5e1')
        for spine in axes.spines.values():
            spine.set_edgecolor('#475569')
        colorbar = figure.colorbar(artist, cax=figure.add_axes([.85, .20, .025, .66]))
        colorbar.set_label('Brightness temperature (MK)', color='#cbd5e1', fontsize=10)
        colorbar.ax.tick_params(colors='#cbd5e1', labelsize=9)
        colorbar.outline.set_edgecolor('#475569')
        output = io.BytesIO()
        figure.savefig(output, format='png', facecolor=figure.get_facecolor())
        pngs.append(output.getvalue())
        figure.clear()
    return tuple(pngs)


class RadioImageCache:
    def __init__(self, base_url, *, opener=urlopen, renderer=render_planes, clock=time.time):
        self.base_url = base_url.rstrip('/')
        self.opener = opener
        self.renderer = renderer
        self.clock = clock
        self.lock = threading.Lock()
        self.refresh_lock = threading.Lock()
        self.snapshots = ()
        self.last_attempt = None
        self.last_success = None
        self.error = None
        self.log = logging.getLogger(__name__)

    def request(self, path):
        with self.opener(self.base_url + path, timeout=10) as response:
            payload = response.read(32 * 1024 * 1024 + 1)
            if len(payload) > 32 * 1024 * 1024:
                raise ValueError('Image response too large')
            return payload, response.headers.get('X-Observation-Timestamp')

    def refresh_once(self):
        if not self.refresh_lock.acquire(blocking=False):
            return
        with self.lock:
            self.last_attempt = self.clock()
        try:
            metadata = json.loads(self.request('/images?timestamp=newest')[0])
            stamp = metadata['timestamp']
            observed = datetime.strptime(stamp, '%Y%m%dT%H%M%S').replace(tzinfo=timezone.utc)
            rows = metadata['images']
            if len(rows) != 4 or sorted(row['target_mhz'] for row in rows) != list(FREQUENCIES):
                raise ValueError('Expected four fine-channel images')
            with self.lock:
                current = self.snapshots[0] if self.snapshots else None
            if not current or current.timestamp != stamp:
                query = urlencode({'timestamp': stamp})
                def download(target):
                    data, selected = self.request(f'/images/{target}.npz?{query}')
                    # Historical selection is nearest; expiry must never mix observations.
                    if selected != stamp:
                        raise ValueError('Observation changed during image download')
                    return data
                with ThreadPoolExecutor(max_workers=4) as pool:
                    payloads = list(pool.map(download, FREQUENCIES))
                by_target = {row['target_mhz']: row for row in rows}
                planes = tuple(read_plane(data, by_target[target]) for target, data in zip(FREQUENCIES, payloads))
                # Immutable PNG URLs must change when the rendering changes too.
                version = hashlib.sha256(RENDER_PROFILE + stamp.encode() + b''.join(payloads)).hexdigest()[:20]
                snapshot = ImageSnapshot(stamp, observed.isoformat().replace('+00:00', 'Z'),
                                         version, planes, self.renderer(planes))
                if len(snapshot.pngs) != 4:
                    raise ValueError('Incomplete rendered snapshot')
                with self.lock:
                    # The previous complete set lets in-flight browser image loads finish.
                    self.snapshots = (snapshot,) + self.snapshots[:1]
            with self.lock:
                self.last_success = self.clock()
                self.error = None
        except Exception as exc:
            with self.lock:
                self.error = ('No recent solar images are available. Waiting for the next observation.'
                              if isinstance(exc, HTTPError) and exc.code == 404 else
                              'Image source temporarily unavailable; retrying every minute.')
            self.log.warning('Fine-channel image refresh failed (%s)', type(exc).__name__)
        finally:
            self.refresh_lock.release()

    def run(self, shutdown_event):
        while not shutdown_event.is_set():
            started = time.monotonic()
            self.refresh_once()
            shutdown_event.wait(max(0, POLL_SECONDS - (time.monotonic() - started)))

    def metadata(self):
        with self.lock:
            current = self.snapshots[0] if self.snapshots else None
            stale = bool(current and (self.error or self.last_success is None
                         or self.clock() - self.last_success > 2 * POLL_SECONDS
                         or self.clock() - datetime.fromisoformat(current.observation_time.replace('Z', '+00:00')).timestamp() > 600))
            return {
                'status': 'stale' if stale else 'ready' if current else 'unavailable' if self.error else 'loading',
                'poll_interval_seconds': POLL_SECONDS,
                'observation_time': current.observation_time if current else None,
                'version': current.version if current else None,
                'checked_at': self.last_attempt,
                'fetched_at': self.last_success,
                'message': self.error,
                'images': [{
                    'target_mhz': plane.target, 'freq_mhz': plane.frequency,
                    'date_obs': plane.date_obs, 'unit': 'MK',
                    'image_url': f'/radio-images/{plane.target}.png?version={current.version}',
                } for plane in current.planes] if current else [],
            }

    def image(self, target, version=None):
        with self.lock:
            snapshot = next((item for item in self.snapshots if version is None or item.version == version), None)
            if snapshot is None or target not in FREQUENCIES:
                return None
            return snapshot.pngs[FREQUENCIES.index(target)], snapshot.timestamp, snapshot.version
