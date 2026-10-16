"""Bounded source-time spectrum selection, independent of HTTP and acquisition."""
import json
import struct
import numpy as np

HISTORY_SECONDS = 3600
MAX_RANGE_ROWS = 1200
RECENT_FRAMES = 600


def validate_range(seconds=300, start=None, end=None, max_rows=1200):
    if not np.isfinite(seconds) or not 1 <= seconds <= HISTORY_SECONDS:
        raise ValueError('seconds must be finite and between 1 and 3600')
    if not isinstance(max_rows, int) or not 16 <= max_rows <= MAX_RANGE_ROWS:
        raise ValueError('max_rows must be between 16 and 1200')
    if (start is None) != (end is None):
        raise ValueError('start and end must be supplied together')
    if start is not None and (not np.isfinite([start, end]).all() or not 0 < end-start <= HISTORY_SECONDS):
        raise ValueError('Selected range must be finite, increasing, and at most one hour')


def json_rows(values):
    if np.isfinite(values).all():
        return values.tolist()
    rows = values.astype(object)
    rows[~np.isfinite(values)] = None
    return rows.tolist()


def encode_binary(metadata, intensity, circular, timestamps):
    meta = dict(metadata, schema='lwa-spectrum-iv-v1', rows=len(timestamps), bins=768,
                dtype='float32', byteorder='little', order=['I', 'V'])
    header = json.dumps(meta, separators=(',', ':'), allow_nan=False).encode()
    header += b' ' * (-(4 + len(header)) % 8)
    return b''.join((struct.pack('<I', len(header)), header,
                     np.asarray(timestamps, '<f8').tobytes(),
                     np.asarray(intensity, '<f4').tobytes(),
                     np.asarray(circular, '<f4').tobytes()))


def average_time_bins(intensity, circular, times, start, end, max_rows):
    """UTC-aligned cells, independent I/V means, empty bins remain NaN."""
    # Reserve one cell for the partial edge so UTC-aligned bins never exceed
    # the requested row budget, even for fractional rolling window endpoints.
    width = (end-start) / (max_rows-1)
    first_bin = int(np.floor((start+1e-6)/width))
    last_bin = int(np.floor((end+1e-6)/width))
    rows = min(max_rows, last_bin-first_bin+1)
    origin = first_bin*width
    # MJD conversion can land just below an exact bin boundary.
    indices = np.minimum(rows-1, np.maximum(0, np.floor((times-origin+1e-6)/width).astype(int)))
    outputs = []
    for values in (intensity, circular):
        valid = np.isfinite(values)
        sums = np.zeros((rows, 768), np.float64)
        counts = np.zeros((rows, 768), np.int32)
        np.add.at(sums, indices, np.where(valid, values, 0))
        np.add.at(counts, indices, valid)
        result = np.full((rows, 768), np.nan, np.float32)
        np.divide(sums, counts, out=result, where=counts > 0)
        outputs.append(result)
    return *outputs, origin + (np.arange(rows)+.5)*width, width
