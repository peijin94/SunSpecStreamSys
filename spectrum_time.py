"""Shared source-time geometry for detection images and browser waterfalls."""
import numpy as np

COLUMN_SECONDS = 0.5
TIME_COLUMNS = 600


def time_raster(values, timestamps, previous_timestamp=None):
    """Low frequencies first; columns span the actual last five source minutes."""
    if len(timestamps) == 0:
        raise ValueError('No source frames')
    latest = int(np.floor(timestamps[-1] / COLUMN_SECONDS))
    first = latest - TIME_COLUMNS + 1
    raster = np.zeros((TIME_COLUMNS, values.shape[1]), dtype=np.float32)
    previous = previous_timestamp
    for row, timestamp in zip(values, timestamps):
        slot = int(np.floor(timestamp / COLUMN_SECONDS))
        covered = slot
        if previous is not None and 0 <= timestamp - previous <= COLUMN_SECONDS * 1.5:
            covered = min(slot, int(np.floor(previous / COLUMN_SECONDS)) + 1)
        for column in range(max(first, covered), min(latest, slot) + 1):
            raster[column - first] = row
        previous = timestamp
    return raster, first * COLUMN_SECONDS, (latest + 1) * COLUMN_SECONDS


def source_box(box, image_shape, window_start, window_end):
    """Convert YOLO's low-frequency-at-top image into UTC and display geometry."""
    height, width = image_shape
    x1, y1, x2, y2 = (float(value) for value in box)
    x1, x2 = sorted((max(0, min(width, x1)), max(0, min(width, x2))))
    y1, y2 = sorted((max(0, min(height, y1)), max(0, min(height, y2))))
    return {
        'bbox': [x1 / width, 1 - y2 / height, (x2 - x1) / width, (y2 - y1) / height],
        'time_start': window_start + x1 / width * (window_end - window_start),
        'time_end': window_start + x2 / width * (window_end - window_start),
    }
