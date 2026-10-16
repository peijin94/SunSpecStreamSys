"""Deployment variants for the spectrum receiver backend.

Two services share this codebase: an interactive development service with the
full one-hour window, and a restricted public service with a fixed rolling
300-second window and no interactive measurement. This module is stdlib-only
so policy can be tested without the receiver's heavy dependencies.

Production ports (wired by the service units, not by defaults here): the ZMQ
source subscriber stays on 30005, the public web service on 9527, and the
development web service on 9529. The constructor/CLI default web port remains
9527 so existing tests and single-service checkouts keep working.
"""
import math

DEV_MODE = 'dev'
PUBLIC_MODE = 'public'
MODES = (DEV_MODE, PUBLIC_MODE)

# Canonical production web ports (service units pass these explicitly).
PUBLIC_WEB_PORT = 9527
DEV_WEB_PORT = 9529
DEFAULT_WEB_PORT = 9527

HISTORY_SECONDS = {DEV_MODE: 3600, PUBLIC_MODE: 300}
PUBLIC_SECONDS = 300
INTERACTIVE = {DEV_MODE: True, PUBLIC_MODE: False}
MEASUREMENT_ENABLED = {DEV_MODE: True, PUBLIC_MODE: False}

# Minimum ring frames so the 600-new-frame archive cadence and the latest-600
# detection/AI analysis window survive even when the public API horizon is
# shorter than 600 frames at a slow source cadence.
MIN_RING_FRAMES = 601

# Browser origin allowed to use the credentialed dev APIs (Apache enforces
# Basic auth for the development subtree; the backend must not reflect
# arbitrary origins alongside credentials).
DEV_ALLOW_ORIGINS = ('https://ovsa.njit.edu',)
# Public APIs are unauthenticated, read-only, and credentialless.
PUBLIC_ALLOW_METHODS = ('GET', 'HEAD', 'OPTIONS')

# Headers that must never be reflected by /debug/request. Apache strips
# Authorization/Proxy-Authorization before proxying and enforces Basic auth
# for the development subtree; the backend redacts them as well so direct
# loopback access cannot echo credentials either.
SENSITIVE_HEADERS = frozenset({'authorization', 'proxy-authorization', 'cookie'})


def normalize_mode(mode):
    """Return 'dev' or 'public'; None/blank means dev (existing behavior)."""
    if mode is None or (isinstance(mode, str) and not mode.strip()):
        return DEV_MODE
    text = str(mode).strip().lower()
    if text in MODES:
        return text
    raise ValueError("app_mode must be 'dev' or 'public'")


def history_seconds(mode):
    """Source-time horizon in seconds for API responses."""
    return HISTORY_SECONDS[normalize_mode(mode)]


def is_public(mode):
    return normalize_mode(mode) == PUBLIC_MODE


def ring_capacity(streaming_interval, mode):
    """Default ring frames for a cadence; explicit overrides still win."""
    try:
        cadence = float(streaming_interval)
    except (TypeError, ValueError):
        raise ValueError('streaming_interval must be positive and finite')
    if not math.isfinite(cadence) or cadence <= 0:
        raise ValueError('streaming_interval must be positive and finite')
    mode = normalize_mode(mode)
    frames = int(math.ceil(history_seconds(mode) / cadence)) + 1
    if mode == PUBLIC_MODE:
        frames = max(MIN_RING_FRAMES, frames)
    return frames


def validate_web_port(port):
    try:
        number = int(port)
    except (TypeError, ValueError):
        raise ValueError('web_port must be an integer port')
    if not 1 <= number <= 65535:
        raise ValueError('web_port must be between 1 and 65535')
    return number


def validate_public_range(seconds, start, end):
    """Public /spectrum-range accepts only the fixed rolling 300 s window."""
    if start is not None or end is not None:
        raise ValueError('public range is fixed to the rolling 300 s window')
    try:
        wanted = float(seconds)
    except (TypeError, ValueError):
        raise ValueError('public range is fixed to the rolling 300 s window')
    if not math.isfinite(wanted) or wanted != PUBLIC_SECONDS:
        raise ValueError('public range is fixed to the rolling 300 s window')


def filter_debug_headers(headers):
    """Drop credential headers (case-insensitive) before JSON reflection."""
    return {key: value for key, value in dict(headers).items()
            if str(key).lower() not in SENSITIVE_HEADERS}


def cors_options(mode):
    """CORSMiddleware kwargs: credentialed dev origin vs credentialless public."""
    if is_public(mode):
        return {'allow_origins': ['*'], 'allow_credentials': False,
                'allow_methods': list(PUBLIC_ALLOW_METHODS),
                'allow_headers': ['*']}
    return {'allow_origins': list(DEV_ALLOW_ORIGINS), 'allow_credentials': True,
            'allow_methods': ['*'], 'allow_headers': ['*']}


def deployment_payload(mode, web_port):
    """Capability metadata served at GET /deployment."""
    mode = normalize_mode(mode)
    return {
        'mode': mode,
        'history_seconds': history_seconds(mode),
        'interactive': INTERACTIVE[mode],
        'measurement_enabled': MEASUREMENT_ENABLED[mode],
        'web_port': validate_web_port(web_port),
    }
